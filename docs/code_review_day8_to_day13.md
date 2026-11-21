# EvidenceFlow AI — Code Review: Days 8–13 (Tier 2: Memory & 5-Layer Cache)

**Reviewer**: Principal AI Architect  
**Date**: 2026-11-21  
**Scope**: Day 8 (Neo4j Conversational Graph Memory), Day 9–10 (5-Layer HMAC-SHA256 Scoped Cache & Deterministic Validator), Day 11 (FastMCP Memory-Cache Server with RBAC), Days 12–13 (Cache Collision Trap Tests, Ablation Study, Non-Inferiority Analysis)  
**Rule**: Read-only review. No files have been changed. All findings require developer action.

---

## Overall Assessment

The Tier 2 Memory & Cache subsystem is architecturally ambitious and largely well-constructed. The Neo4j graph client correctly enforces parameterized Cypher throughout a single ACID transaction, the five-layer cache hierarchy introduces principled scoping and HMAC-SHA256 isolation, the zero-LLM deterministic validator is elegant, and the ablation framework is statistically sound. However, **seven issues warrant developer attention before this code ships to production**, ranging from two critical defects (a string-formatted Cypher injection bypass in `graph_client.py`, and a thread-safety gap in the singleton driver) through high-priority reliability concerns (bare-module-level `logging.basicConfig` override, unprotected `_l2_semantic.pop(0)` O(N) eviction under lock, and the RBAC decorator silently falling back to positional arguments) to medium-priority correctness risks (default version fallback hiding unsupported-version bugs, `has_off_corpus_content_words` false-negative on two-character tokens, missing `l1_passed` gate for cache-hit paths in the ablation groundedness metric) and a low-priority observation on the stats counter not being reset by `clear_all()`.

---

## Severity Legend

| Symbol | Severity | Meaning |
|:---:|:---|:---|
| 🔴 | **CRITICAL** | Will cause incorrect behavior, data corruption, or production failures |
| 🟠 | **HIGH** | Significant performance or reliability issue |
| 🟡 | **MEDIUM** | Correctness risk in edge cases or maintainability concern |
| 🔵 | **LOW** | Style, naming, or minor improvement |
| 💡 | **OBSERVATION** | Not a defect — architectural note or improvement opportunity |

---

## Day 8 — Neo4j Conversational Graph Memory

### `memory/graph_client.py`

---

#### D8-1 🔴 `graph_client.py` — String-formatted Cypher in `record_claim_relationship` and `record_passage_relationship`

**File**: `memory/graph_client.py`  
**Lines**: 611–626, 688–703

```python
# graph_client.py  lines 611-626
session.run(
    f"""
    MATCH (c1:Claim {{id: $subject_id}})
    MATCH (c2:Claim {{id: $target_id}})
    MERGE (c1)-[r:{rel}]->(c2)   # <-- rel is interpolated directly into Cypher string
    SET r.reason = $reason,
        r.created_at = datetime(),
        r += $meta
    """,
    ...
)
```

**Problem**: The module-level docstring for `graph_client.py` explicitly claims *"Zero string-formatted Cypher; all inputs passed as parameters."* (line 151). That guarantee is violated for `record_claim_relationship` (line 615) and `record_passage_relationship` (line 692), where the relationship type `rel` is embedded via an f-string. While both functions do validate `rel` against an allowlist (`{"SUPERSEDES", "CONTRADICTS"}`), the structural violation means:

1. If a future developer extends the allowlist or removes the guard, Cypher injection becomes directly reachable.
2. The stated invariant in the docstring is false, which undermines trust in all security assertions made by that module.
3. Static analysis tools and security scanners will flag this correctly as a Cypher injection vector because they cannot trace the allowlist validation.

**Scenario**: A developer adds `"RELATES_TO"` to the allowlist but passes an externally-sourced value. The f-string formatting bypasses all parameterisation. Even with the current allowlist, a future change to allow caller-supplied relationship types (e.g. for a plugin) silently opens injection.

**Better Approach**: Neo4j does not support dynamic relationship type injection via parameters, but the correct mitigation is a lookup dictionary mapping to hardcoded query strings, eliminating the f-string entirely:

```python
_CLAIM_REL_QUERIES = {
    "SUPERSEDES": """
        MATCH (c1:Claim {id: $subject_id})
        MATCH (c2:Claim {id: $target_id})
        MERGE (c1)-[r:SUPERSEDES]->(c2)
        SET r.reason = $reason, r.created_at = datetime(), r += $meta
    """,
    "CONTRADICTS": """
        MATCH (c1:Claim {id: $subject_id})
        MATCH (c2:Claim {id: $target_id})
        MERGE (c1)-[r:CONTRADICTS]->(c2)
        SET r.reason = $reason, r.created_at = datetime(), r += $meta
    """,
}

rel = rel_type.strip().upper()
query = _CLAIM_REL_QUERIES.get(rel)
if query is None:
    raise ValueError(f"Invalid relationship type '{rel_type}'")
session.run(query, {"subject_id": ..., "target_id": ..., "reason": ..., "meta": ...})
```

Apply the same pattern to `record_passage_relationship`. This keeps the guarantee in the docstring true and eliminates all future injection risk.

---

#### D8-2 🔴 `connection.py` — Non-thread-safe singleton driver initialization (double-checked locking gap)

**File**: `memory/connection.py`  
**Lines**: 18–34

```python
_neo4j_driver: Optional[Driver] = None

def get_neo4j_driver() -> Driver:
    global _neo4j_driver
    if _neo4j_driver is None:                     # <-- CHECK 1 (unprotected)
        settings = get_settings()
        logger.info("Initializing Neo4j Bolt driver ...")
        _neo4j_driver = GraphDatabase.driver(     # <-- WRITE (unprotected)
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            ...
        )
    return _neo4j_driver
```

**Problem**: Under concurrent workloads (FastAPI async handlers, multiple LangGraph threads, the MCP server thread), two or more coroutines can pass the `if _neo4j_driver is None` check simultaneously. Each then creates an independent `Driver` instance. The surplus driver is orphaned—never closed—leaking Bolt connections, authentication sessions, and OS file descriptors. The Neo4j `GraphDatabase.driver()` constructor is **not** idempotent: calling it twice does not merge connection pools.

**Scenario**: Under a production load burst at startup (e.g. gunicorn pre-fork, 4 worker threads all receiving their first request simultaneously), four Driver objects are created. Only the last write to `_neo4j_driver` survives; the other three hold open connection pools indefinitely until process exit.

**Better Approach**: Protect initialization with a module-level lock:

```python
import threading

_neo4j_driver: Optional[Driver] = None
_neo4j_lock = threading.Lock()

def get_neo4j_driver() -> Driver:
    global _neo4j_driver
    if _neo4j_driver is None:
        with _neo4j_lock:
            if _neo4j_driver is None:   # double-checked locking
                settings = get_settings()
                logger.info("Initializing Neo4j Bolt driver ...")
                _neo4j_driver = GraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                    max_connection_lifetime=30 * 60,
                    max_connection_pool_size=50,
                    connection_acquisition_timeout=10.0,
                )
    return _neo4j_driver
```

---

#### D8-3 🟡 `graph_client.py` — Default `prior_version = "16"` silently suppresses unsupported-version detection

**File**: `memory/graph_client.py`  
**Lines**: 477–478

```python
prior_version = record["prior_version"] if record and record["prior_version"] else "16"
target_version = explicit_version or prior_version
```

**Problem**: When Neo4j is unreachable (the exception is caught at lines 474–475 and `record` remains `None`), `prior_version` silently defaults to `"16"`. This means a follow-up query for an obscure version-switch phrase (e.g. "and in 12?") will resolve `target_version = "12"` correctly if explicitly stated, but if no version is stated and no prior context is available, the pipeline assumes `v16` without signaling to the caller that memory was unavailable. The correctness of `is_supported_version` in the returned `resolved` dict then depends on whether `"16"` is the genuinely correct answer or a fallback mask for a connectivity failure.

**Scenario**: Neo4j is down during a planned maintenance window. All follow-up queries silently default to `v16`. A user asking "What about it in v12?" (which should return UNSUPPORTED_VERSION) might instead receive a `v16` answer if `explicit_version` extraction fails. The user has no visibility that the memory system degraded.

**Better Approach**: Return `"passthrough"` with a `reason="memory_unavailable"` and log a structured warning when `record is None` *due to exception* (distinguish from the legitimate "no prior turns" path):

```python
if _memory_query_failed:   # set True in the except block
    return {
        "status": "passthrough",
        "reason": "memory_unavailable",
        "entity_name": None,
        "target_version": explicit_version or "16",
        "rewritten_query": follow_up_query,
        "latency_ms": round(latency_ms, 3),
    }
```

Callers can then decide whether to surface a soft warning to the user.

---

#### D8-4 🟡 `corpus_settings.py` — `has_off_corpus_content_words` misses two-character content words

**File**: `memory/corpus_settings.py`  
**Lines**: 187–188

```python
tokens = re.findall(r"\b[a-z]{3,}\b", query.lower())
```

**Problem**: The regex `[a-z]{3,}` skips all tokens shorter than 3 characters. The vocabulary and stopword lists contain no entries shorter than 3 characters either (by construction), so words of length 2 (`go`, `is`, `do`, `AI`, `DB`, `PG`, `OK`) are silently skipped. While this is mostly harmless for stopwords, it means a query such as `"Is DB up?"` or `"Is PG ok?"` with off-corpus 2-letter tokens will pass `has_off_corpus_content_words` as `False` even though `DB` and `PG` are not in the technical vocabulary. The fix is relatively minor but `PG` specifically could cause a false follow-up resolution on a question about PostgreSQL abbreviations.

**Better Approach**: The minimum length guard should align with the minimum meaningful token length for the domain. Consider reducing to `[a-z]{2,}` and adding two-character tokens to `EXCLUDED_WORDS`, or at minimum document the limitation explicitly:

```python
# Note: tokens of length 1-2 are skipped intentionally. Single-letter
# variables and two-character abbreviations rarely carry domain-distinguishing
# signal and are excluded to avoid false positives.
tokens = re.findall(r"\b[a-z]{3,}\b", query.lower())
```

---

#### D8-5 🟡 `init_neo4j.py` — `logging.basicConfig` at module level may override application logging config

**File**: `memory/init_neo4j.py`  
**Lines**: 10

```python
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
```

**Problem**: `logging.basicConfig()` is a no-op if the root logger already has handlers configured. However, if this module is imported *before* the application's main logging configuration is applied (e.g. during schema validation at import time), it installs a `StreamHandler` on the root logger with a fixed format. This will override or conflict with the application-level logging configuration set up in `workflow.py` and `memory_cache_server.py`. The standard Python logging idiom is that only the **application entry point** calls `basicConfig`; library and submodule code should never call it.

**Better Approach**: Remove the `basicConfig` call from the module body. The `if __name__ == "__main__"` block at the bottom is the only legitimate caller:

```python
# Remove line 10 entirely from module body.

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    try:
        init_neo4j_schema()
        ...
    finally:
        close_neo4j_driver()
```

---

#### D8-6 💡 `graph_client.py` — `get_session_history` Cypher produces row fan-out for multi-entity, multi-claim turns

**File**: `memory/graph_client.py`  
**Lines**: 315–354

**Observation**: The `get_session_history` query uses `OPTIONAL MATCH` for entities, claims, and version without intermediate `WITH` aggregation. For a turn with 5 entities and 10 claims, the cross-join produces 50 rows per turn, which are then collapsed by `collect(DISTINCT ...)`. This is semantically correct due to `DISTINCT`, but generates 50× the network traffic for large turns. For a typical Q&A session this is negligible, but as thread history grows (e.g. 50-turn audit sessions), this can become a performance issue. Consider restructuring with `WITH` and `collect()` per path, or batching entity/claim collection separately.

---

## Day 9–10 — 5-Layer HMAC-SHA256 Scoped Cache

### `cache/key_generator.py`

---

#### D9-1 🔵 `key_generator.py` — `normalize_query` comment is misleading about underscore handling

**File**: `cache/key_generator.py`  
**Lines**: 21–22

```python
cleaned = re.sub(r"[^\w\s]", "", query.lower())
```

**Problem**: `\w` matches `[a-zA-Z0-9_]`, so underscores are **preserved** correctly. However, the comment says "Strip punctuation while keeping alphanumeric and whitespace", which is misleading—underscores are not punctuation and are intentionally kept. This creates a documentation/naming mismatch. The behavior itself is correct (underscores in GUC names like `wal_compression` survive normalization), but a developer reading the comment might incorrectly "fix" the regex to `[^a-zA-Z0-9\s]` and accidentally break GUC-name stability in cache keys.

**Better Approach**: Clarify the comment:

```python
# Strip all characters except word characters (letters, digits, underscores) and whitespace.
# Underscore preservation is intentional: PostgreSQL GUC names like wal_compression use underscores.
cleaned = re.sub(r"[^\w\s]", "", query.lower())
```

---

### `cache/manager.py`

---

#### D9-2 🟠 `manager.py` — `_l2_semantic.pop(0)` O(N) eviction holds the global `_lock` for the duration

**File**: `cache/manager.py`  
**Lines**: 366–370

```python
with self._lock:
    self._l2_semantic.append(entry)
    # Evict oldest if beyond capacity
    if len(self._l2_semantic) > self._l2_max_size:
        self._l2_semantic.pop(0)   # O(N) shift of up to 1000 elements
```

**Problem**: Python's `list.pop(0)` on a 1,000-element list requires shifting all 999 remaining elements in memory. This operation runs **inside the global `RLock`** (`self._lock`), which is shared by L1, L3, L4, and L5 operations. While the comment in `__init__` correctly notes that FIFO is used instead of LRU for L2 and links to a future fix, the problem is not just LRU vs. FIFO—it is that the O(N) shift occurs under a lock that blocks all other cache operations. Under 50 concurrent threads writing to the cache, this serializes them all during the O(N) shift.

**Scenario**: A burst of 50 concurrent L4 embedding cache writes after a cache warm-up would each trigger L2 append (the size drifts to 1001), and each of those 50 writes will pop(0) under the global lock, forcing the remaining 49 threads to wait for a list-shift of up to 1000 items.

**Better Approach**: Replace `List[SemanticCacheEntry]` with a `collections.deque(maxlen=max_semantic_entries)` for O(1) FIFO eviction without shifting:

```python
from collections import deque

# In __init__:
self._l2_semantic: deque = deque(maxlen=max_semantic_entries)

# In set_semantic (no manual eviction needed—deque enforces maxlen automatically):
with self._lock:
    self._l2_semantic.append(entry)
    # deque(maxlen=N) automatically drops the oldest element—O(1), no shift
```

This eliminates the O(N) problem. The `get_semantic` iteration over the deque remains O(N) but that is inherent to brute-force cosine search, not a locking concern.

---

#### D9-3 🔵 `manager.py` — `clear_all()` does not reset telemetry counters

**File**: `cache/manager.py`  
**Lines**: 631–638

```python
def clear_all(self) -> None:
    """Clears all 5 cache layers."""
    with self._lock:
        self._l1_exact.clear()
        self._l2_semantic.clear()
        self._l3_retrieval.clear()
        self._l4_embedding.clear()
        self._l5_claim.clear()
        # stats counters are NOT reset here
```

**Problem**: After `clear_all()`, the `stats` dictionary retains cumulative hit/miss/eviction counts from before the clear. In ablation testing (`ablation_runner.py`), a `ScopedCacheManager()` is freshly constructed per condition, so this is not a problem for the ablation. However, in integration tests or production monitoring dashboards that call `clear_all()` to reset between test cases, the stats will appear to show hits and evictions from the cleared data, misleading operators.

**Better Approach**:

```python
def clear_all(self) -> None:
    """Clears all 5 cache layers and resets telemetry counters."""
    with self._lock:
        self._l1_exact.clear()
        self._l2_semantic.clear()
        self._l3_retrieval.clear()
        self._l4_embedding.clear()
        self._l5_claim.clear()
        for key, val in self.stats.items():
            if isinstance(val, dict):
                for k in val:
                    val[k] = 0
            else:
                self.stats[key] = 0
```

---

#### D9-4 💡 `manager.py` — L4 intentionally excluded from version/passage invalidation; document rationale

**File**: `cache/manager.py`  
**Lines**: 534–580

**Observation**: `invalidate_by_version` correctly evicts L1, L2, L3, and L5 entries scoped to a version. L4 (embedding cache) is intentionally skipped because embeddings are version-agnostic (they are pure mathematical projections of text, not corpus knowledge). This design decision is correct and the comment in `get_semantic` explains it. `invalidate_by_passage` also skips L4—which is also correct because embeddings do not encode passage identity. Both omissions are architecturally sound. This observation is noted for documentation purposes so a future developer does not add L4 eviction incorrectly. Consider adding an explicit comment inside both invalidation methods:

```python
# L4 (embedding cache) is intentionally NOT invalidated: embeddings are
# version-agnostic mathematical projections of query text. They remain
# valid regardless of corpus or guardrail version changes.
```

---

### `cache/validator.py`

---

#### D10-1 🟡 `validator.py` — `validate_cache_hit` early-returns on first stale passage; subsequent stale passages go undetected

**File**: `cache/validator.py`  
**Lines**: 176–188

```python
# Chunk missing in live corpus (deleted/renamed)
if live_hash is None:
    stale_passages.append(pid)
    return False, f"PASSAGE_NOT_FOUND: ...", stale_passages   # <-- early return

# Chunk hash mismatch
if cached_hash and live_hash != cached_hash:
    stale_passages.append(pid)
    return (False, f"HASH_MISMATCH: ...", stale_passages)    # <-- early return
```

**Problem**: Both failure branches perform an **early return** after detecting the first stale passage. If a cached answer cites three passages and all three have changed, only the first stale passage is added to `stale_passages`. `evict_stale_cache_entry` then calls `invalidate_by_passage` only for that one passage. The other two stale passages remain in cache entries that cite them, and the next request for a different query citing one of those un-evicted passages will serve a stale answer.

**Scenario**: A corpus update changes passages A, B, and C. Query Q1 cites all three. Validation fails on A, evicts entries citing A. Q1's L1 entry is gone. But a different cached query Q2 cites only B—it was not evicted because B was never reported as stale. Q2 still returns stale content on the next request.

**Better Approach**: Accumulate all stale passages before returning:

```python
stale_passages = []
first_failure_reason = None

for item in cached_citations:
    pid, cached_hash = extract_citation_info(item)
    if not pid:
        return False, "INVALID_CITATION: missing passage_id", stale_passages

    live_hash = db.get_chunk_hash(pid)

    if live_hash is None:
        stale_passages.append(pid)
        if first_failure_reason is None:
            first_failure_reason = f"PASSAGE_NOT_FOUND: passage '{pid}' no longer exists"
    elif cached_hash and live_hash != cached_hash:
        stale_passages.append(pid)
        if first_failure_reason is None:
            first_failure_reason = f"HASH_MISMATCH: passage '{pid}' hash changed"

if stale_passages:
    return False, first_failure_reason, stale_passages

return True, None, []
```

---

#### D10-2 🟡 `validator.py` — `evict_stale_cache_entry` accesses private `_lock` and `_l1_exact` attributes directly

**File**: `cache/validator.py`  
**Lines**: 208–215

```python
if cache_key and total_evicted == 0:
    with cache_manager._lock:                      # <-- direct private attribute access
        if cache_key in cache_manager._l1_exact:   # <-- direct private attribute access
            del cache_manager._l1_exact[cache_key]
            cache_manager.stats["evictions"] += 1
            total_evicted += 1
```

**Problem**: `evict_stale_cache_entry` in `validator.py` reaches into the private internals (`_lock`, `_l1_exact`) of `ScopedCacheManager`. This creates a hidden coupling: if `ScopedCacheManager` renames `_l1_exact` or changes the lock strategy (e.g. switching to `asyncio.Lock`), the validator silently breaks.

**Better Approach**: Add a public method to `ScopedCacheManager`:

```python
# In ScopedCacheManager:
def evict_by_key(self, cache_key: str) -> bool:
    """Explicitly evicts a single L1 entry by its composite scope key."""
    with self._lock:
        if cache_key in self._l1_exact:
            del self._l1_exact[cache_key]
            self.stats["evictions"] += 1
            return True
    return False
```

Then in `validator.py`:

```python
if cache_key and total_evicted == 0:
    if cache_manager.evict_by_key(cache_key):
        total_evicted += 1
        logger.info("Explicitly evicted cache key '%s' from L1", cache_key)
```

---

## Day 11 — FastMCP Memory-Cache Server

### `mcp/memory_cache_server.py`

---

#### D11-1 🟠 `memory_cache_server.py` — `require_role` decorator dead fallback silently drops legitimate authorized calls

**File**: `mcp/memory_cache_server.py`  
**Lines**: 123–131

```python
def wrapper(*args, **kwargs):
    t0 = time.perf_counter()
    caller_role = kwargs.get("caller_role")
    if not caller_role and len(args) >= 1:
        # Inspect kwargs or fallback
        pass    # <-- THE FALLBACK IS EMPTY / DOES NOTHING
```

**Problem**: The decorator attempts to extract `caller_role` from `kwargs`, but falls back to a `pass` statement when it is not found in `kwargs` and args are present. If `caller_role` is passed positionally (which FastMCP might do internally when routing tool calls), `role_str` will be an empty string and `"" not in allowed_roles` evaluates to `True`, resulting in a **PermissionDenied** response for a legitimately authorized caller.

**Scenario**: FastMCP or a test framework calls `lookup_cache("What is wal_compression?", "16", "cache_gatekeeper")` with `caller_role` as the third positional argument. The decorator reads `kwargs.get("caller_role")` → `None`, then hits the dead `pass`, so `role_str = ""`, which is not in `ALLOWED_CACHE_READ_ROLES`. The call returns `{"status": "permission_denied"}` for a legitimate authorized caller.

**Better Approach**: Extract `caller_role` from both positional arguments (by inspecting the wrapped function's signature) and keyword arguments:

```python
import inspect

def wrapper(*args, **kwargs):
    t0 = time.perf_counter()
    caller_role = kwargs.get("caller_role")
    if caller_role is None:
        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            if "caller_role" in params:
                idx = params.index("caller_role")
                if idx < len(args):
                    caller_role = args[idx]
        except (ValueError, TypeError):
            pass
    ...
```

Alternatively, enforce that all MCP tools must pass `caller_role` as a keyword argument only and document this contract explicitly.

---

#### D11-2 🟡 `memory_cache_server.py` — 50ms socket probe timeout is too aggressive for cross-AZ deployments

**File**: `mcp/memory_cache_server.py`  
**Lines**: 64

```python
with socket.create_connection((s.POSTGRES_HOST, s.POSTGRES_PORT), timeout=0.05):
```

**Problem**: A 50ms TCP connect timeout is unusably tight for cross-availability-zone PostgreSQL deployments (typical cross-AZ TCP handshake is 1–5ms within the same region, but can spike to 30–80ms under GC pauses or cloud network jitter). A single transient network hiccup during GC will mark `_db_available = False` and suppress audit logging for 15 seconds (the circuit-breaker window). This means the first 15 seconds after any network blip produce no audit trail for MCP operations, which is a compliance gap.

**Better Approach**: Raise the timeout to at least 200ms, or make it configurable via settings:

```python
timeout_s = getattr(get_settings(), "MCP_DB_PROBE_TIMEOUT_S", 0.2)
with socket.create_connection((s.POSTGRES_HOST, s.POSTGRES_PORT), timeout=timeout_s):
```

---

#### D11-3 💡 `memory_cache_server.py` — `write_cache` accepts empty answer strings without validation

**File**: `mcp/memory_cache_server.py`  
**Lines**: 213–242

**Observation**: If `answer` is an empty string `""` and `citations` is non-empty, `write_cache` will store an empty answer in L1. The subsequent `validate_cache_hit` in `cache_gatekeeper_node` will accept this entry (because citations are non-empty and hashes match), and serve an empty string as the answer. A guard should be added:

```python
if not ans_text.strip():
    return {"status": "error", "error": "EMPTY_ANSWER: cannot cache an empty answer string"}
```

---

## Days 12–13 — Cache Collision Tests & Ablation Study

### `eval/verify_cache_isolation.py`

---

#### D12-1 🟡 `verify_cache_isolation.py` — Collision test only checks L1 exact cache; L2 semantic collision is untested

**File**: `eval/verify_cache_isolation.py`  
**Lines**: 93–176

```python
cm.set_exact(query=item["q_v14"], ...)   # Primes L1
res_v16_lookup = cm.lookup(query=item["q_v16"], target_versions="16")  # Checks L1
```

**Problem**: The isolation test suite correctly verifies that cross-version queries do not collide in L1 (exact HMAC key). However, the L2 semantic layer is never tested for cross-version collision. L2 uses cosine similarity at threshold 0.96 combined with version filter metadata. The test suite does not verify that two semantically near-identical queries for different versions (cosine ≥ 0.96 on the query embedding) are still correctly separated by the version scope filter in `get_semantic()`. The version filter at lines 304–307 of `manager.py` should prevent L2 collisions, but this is not empirically proven by any test in this file.

**Better Approach**: Add an L2-specific collision probe after priming `set_semantic` for `v14` with a synthetic embedding, then call `get_semantic` with the same embedding but `target_versions="16"` and verify the result is `None`:

```python
# L2 semantic cross-version trap
embedding = [0.1] * 768  # synthetic embedding
cm.set_semantic(query=item["q_v14"], query_embedding=embedding, answer=item["ans_v14"],
                citations=[...], target_versions=["14"])
l2_result = cm.get_semantic(query_embedding=embedding, target_versions="16")
if l2_result is not None:
    collisions += 1
    print(f" [L2 COLLISION] v14 semantic entry served for v16 scope on '{param}'!")
else:
    print(f" [PASS] L2 correctly blocked v14 semantic entry for v16 scope on '{param}'")
```

---

#### D12-2 🔵 `verify_cache_isolation.py` — Synthetic hash strings instead of real SHA-256 digests

**File**: `eval/verify_cache_isolation.py`  
**Lines**: 23–28

```python
"hash_v14": "hash_v14_wal_compression_off_111",
"hash_v16": "hash_v16_wal_compression_on_222",
```

**Problem**: The validator's `validate_cache_hit` function compares stored `cached_hash` against `live_hash` using string equality. Using synthetic non-SHA-256 strings means the test does not exercise the real hash format that production code will store. If the production hash format changes (e.g. `sha256:<hex>` prefix is added), the test will not catch the mismatch.

**Better Approach**: Use real `hashlib.sha256` digests in the test matrix:

```python
import hashlib

def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

"hash_v14": _h("pg14_wal_compression_off"),
"hash_v16": _h("pg16_wal_compression_on"),
```

---

### `eval/ablation_runner.py`

---

#### D13-1 🟡 `ablation_runner.py` — Cache-hit groundedness inflates `+Cache`/`+Both` distribution on repeats 2–3

**File**: `eval/ablation_runner.py`  
**Lines**: 159–162

```python
grounded = 1.0 if state.l1_passed and not state.is_abstained else 0.0
if not expected_answerable and state.is_abstained:
    grounded = 1.0  # Clean abstention is correct
groundedness_scores.append(grounded)
```

**Problem**: For cache hits, `cache_gatekeeper_node` sets `l1_passed = True` and `is_abstained = False`. Therefore, `grounded = 1.0` is awarded for every cache hit. On the first repeat pass, the answer is generated fresh and `l1_passed` reflects genuine L1 citation verification. On repeats 2 and 3 (for `+Cache` and `+Both` conditions), the same question is answered from cache, and `l1_passed = True` is set by the gatekeeper node rather than by the actual L1 citation verifier. The groundedness distribution for the `+Both` condition is therefore biased upward on repeats 2–3 relative to the baseline, potentially inflating the non-inferiority result.

**Scenario**: If the baseline groundedness is 0.80 (20% of questions fail L1) and the `+Both` groundedness after 3 repeats shows 0.93 (because repeats 2–3 always serve cache = `l1_passed=True`), the non-inferiority test passes with strong margin even if the underlying answer quality is identical. The improvement is an artifact of the measurement methodology, not genuine quality gain.

**Better Approach**: For cache hits, record groundedness separately and use only fresh-generation groundedness for the non-inferiority distribution:

```python
is_hit = getattr(state, "cache_hit", False)
if is_hit:
    cache_groundedness_scores.append(1.0)  # tracked separately
else:
    grounded = 1.0 if state.l1_passed and not state.is_abstained else 0.0
    if not expected_answerable and state.is_abstained:
        grounded = 1.0
    groundedness_scores.append(grounded)
```

Report both distributions in the final results dict so the non-inferiority test uses only fresh-generation groundedness.

---

#### D13-2 🔵 `run_ablation.py` — z-approximation used for samples potentially smaller than 30

**File**: `eval/run_ablation.py`  
**Lines**: 62–70

```python
z_crit = 1.6448536269514722
ci_upper = diff + (z_crit * se)
...
p_val = 0.5 * (1.0 + math.erf(-t_stat / math.sqrt(2.0)))
```

**Problem**: The function uses z-critical values (1.6449 for α=0.05 one-sided) derived from the standard normal distribution. This is appropriate for large samples (n ≥ 30) but the default CLI parameters (`--subset 10 --repeats 3`) yield only 30 samples per condition. For n=30, the t-distribution critical value at α=0.05 (one-sided, 29 degrees of freedom) is **1.699**, not 1.6449. Using z=1.6449 systematically underestimates the CI upper bound, making non-inferiority appear to be proven when it may not be statistically valid under the t-distribution.

**Better Approach**: For samples smaller than 60, add an explicit warning:

```python
if min(n1, n2) < 30:
    import warnings
    warnings.warn(
        f"Sample size < 30 (n1={n1}, n2={n2}). z-approximation (1.6449) is "
        "liberal; consider t-distribution (df={n1+n2-2}) critical value ~1.699.",
        stacklevel=2,
    )
```

---

#### D13-3 💡 `run_ablation.py` — JSON artifact path is hardcoded; successive runs silently overwrite previous results

**File**: `eval/run_ablation.py`  
**Lines**: 258–274

```python
json_artifact_path = repo_root / "data/benchmarks/ablation_tier2_results.json"
```

**Observation**: The `--output` CLI flag controls where the Markdown report is appended, but the JSON artifact is always written to the same hardcoded path. Successive ablation runs will silently **overwrite** this file, losing previous run results. Consider adding a `--json-output` flag or timestamping the filename:

```python
import time as _time
json_artifact_path = repo_root / f"data/benchmarks/ablation_tier2_{_time.strftime('%Y%m%d_%H%M%S')}.json"
```

---

## Summary Table

| ID | Severity | File | Issue |
|:---|:---:|:---|:---|
| D8-1 | 🔴 CRITICAL | `memory/graph_client.py` | String-formatted Cypher in `record_claim_relationship` / `record_passage_relationship` violates zero-injection invariant |
| D8-2 | 🔴 CRITICAL | `memory/connection.py` | Non-thread-safe singleton driver initialization; connection pool leak under concurrency |
| D8-3 | 🟡 MEDIUM | `memory/graph_client.py` | Default `prior_version="16"` silently masks Neo4j connectivity failures |
| D8-4 | 🟡 MEDIUM | `memory/corpus_settings.py` | `has_off_corpus_content_words` skips 2-character tokens; can produce false follow-up resolutions |
| D8-5 | 🟡 MEDIUM | `memory/init_neo4j.py` | `logging.basicConfig` at module level may override application logging configuration |
| D8-6 | 💡 OBS | `memory/graph_client.py` | `get_session_history` Cypher row fan-out for multi-entity, multi-claim turns |
| D9-1 | 🔵 LOW | `cache/key_generator.py` | Misleading comment on `normalize_query`; underscore preservation is intentional and undocumented |
| D9-2 | 🟠 HIGH | `cache/manager.py` | `_l2_semantic.pop(0)` O(N) eviction holds global `_lock`; replace with `deque(maxlen=N)` |
| D9-3 | 🔵 LOW | `cache/manager.py` | `clear_all()` does not reset telemetry counters |
| D9-4 | 💡 OBS | `cache/manager.py` | L4 intentionally excluded from version/passage invalidation; rationale should be documented inline |
| D10-1 | 🟡 MEDIUM | `cache/validator.py` | Early-return in `validate_cache_hit` leaves subsequent stale passages un-evicted |
| D10-2 | 🟡 MEDIUM | `cache/validator.py` | `evict_stale_cache_entry` accesses private `_lock` / `_l1_exact` directly; needs public API |
| D11-1 | 🟠 HIGH | `mcp/memory_cache_server.py` | `require_role` decorator dead fallback; `caller_role` from positional args silently drops to empty string, denying legitimate authorized calls |
| D11-2 | 🟡 MEDIUM | `mcp/memory_cache_server.py` | 50ms socket probe timeout too aggressive for cross-AZ deployments; suppresses audit trail on transient jitter |
| D11-3 | 💡 OBS | `mcp/memory_cache_server.py` | `write_cache` accepts empty answer strings without validation guard |
| D12-1 | 🟡 MEDIUM | `eval/verify_cache_isolation.py` | L2 semantic cross-version collision is untested; only L1 exact collision is verified |
| D12-2 | 🔵 LOW | `eval/verify_cache_isolation.py` | Synthetic hash strings instead of real SHA-256 digests; hash format changes go undetected |
| D13-1 | 🟡 MEDIUM | `eval/ablation_runner.py` | Cache-hit groundedness inflates `+Cache`/`+Both` distribution on repeats 2–3; biases non-inferiority result |
| D13-2 | 🔵 LOW | `eval/run_ablation.py` | z-approximation for small samples (n < 30); t-distribution critical value should be used |
| D13-3 | 💡 OBS | `eval/run_ablation.py` | JSON artifact path is hardcoded; successive runs silently overwrite previous results |
