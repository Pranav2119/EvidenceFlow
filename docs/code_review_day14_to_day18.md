# Code Review — Days 14–18 (Tier 3: Cross-Version Reasoning, Advanced Evaluation & CI)

**Reviewer**: Principal AI Architect  
**Date**: 2026-11-21  
**Scope**: Day 14 (Contradiction Analyst + Neo4j SUPERSEDES/CONTRADICTS edges), Day 15 (Independent L2 Judge + selective activation ADR-005), Day 16 (Head-to-head competitive benchmark), Day 17 (External FastMCP server — token bucket, Full Jitter backoff, Pydantic validation), Day 18 (CI regression gate + GitHub Actions workflow)  
**Rule**: Read-only review. No files have been changed. All findings are defects or improvement opportunities for the developer to address.

---

## Overall Assessment

The Tier 3 implementation is architecturally mature and delivers on its design goals — the contradiction analyst correctly wires SUPERSEDES/CONTRADICTS Neo4j edges, the L2 Judge's selective invocation logic is well-designed per ADR-005, the benchmark suite honestly characterises the competitors' failure modes, the MCP server implements a correct Full Jitter token-bucket pattern, and the CI gate fires as a true binary merge blocker. That said, twenty-three issues were found across the five files. The most critical is a **race condition in the async token-bucket rate limiter** that makes the limiter non-reentrant under concurrent coroutines. High-severity issues include a **Cypher injection vector** in `record_claim_relationship` / `record_passage_relationship`, a **silent abstention-accuracy parameter drop in `run_ci_gate`**, the **broken `simulate_outage` path in `get_security_advisories`** (constructing `httpx.HTTPStatusError` with `request=None`), a **live-Qdrant recall probe that always returns 0.0% in CI**, and a **groundedness metric that is always 100% for pgvector**. Medium issues cover version-extraction brittleness, the benchmark's p50/p95 latency formula, singleton key-rotation blindness, and the CI Qdrant health-check using a tool absent from the image. Findings are actionable and each carries a concrete fix.

---

## Severity Legend

| Symbol | Meaning |
|--------|---------|
| 🔴 | **CRITICAL**: Will cause incorrect behaviour, data corruption, or production failures |
| 🟠 | **HIGH**: Significant performance or reliability issue |
| 🟡 | **MEDIUM**: Correctness risk in edge cases or maintainability concern |
| 🔵 | **LOW**: Style, naming, or minor improvement |
| 💡 | **OBSERVATION**: Not a defect — architectural note or improvement opportunity |

---

## Day 14 — `contradiction.py` & `graph_client.py`

---

### D14-1 🟠 `graph_client.py` — Cypher injection via f-string relationship type

**File**: `graph_client.py`  
**Lines**: 611–626 (`record_claim_relationship`) and 688–703 (`record_passage_relationship`)

```python
# record_claim_relationship, line 612
session.run(
    f"""
    MATCH (c1:Claim {{id: $subject_id}})
    MATCH (c2:Claim {{id: $target_id}})
    MERGE (c1)-[r:{rel}]->(c2)   # ← rel inserted via f-string
    SET r.reason = $reason, ...
    """,
    { ... }
)
```

**Problem**: The relationship type `rel` is interpolated directly into the Cypher query string using an f-string even though it has already been validated to `{SUPERSEDES, CONTRADICTS}`. While the current validation guard makes exploitation theoretically impossible, the pattern is dangerous because:  
1. Future callers could bypass the in-function guard if they call `session.run` with a custom Cypher string constructed elsewhere.  
2. The pattern trains a bad habit — any copy-paste of this code without the guard would open an injection.  
3. Neo4j's Python driver does **not** support parameterised relationship-type names, but a whitelist `match` is the safer construction compared to an open f-string.

**Scenario**: A refactor removes the validation guard (or relaxes it) and a caller passes `rel_type = "SUPERSEDES}->(x:Evil) DETACH DELETE x //"`. The relationship name slot becomes a free-form injection vector causing data destruction.

**Better approach**:
```python
def _build_passage_relationship_query(rel: str) -> str:
    """Returns a hardcoded Cypher string for the given allowlisted relationship type."""
    if rel == "SUPERSEDES":
        return """
        MERGE (p1:Passage {passage_id: $subject_id})
        MERGE (p2:Passage {passage_id: $target_id})
        MERGE (p1)-[r:SUPERSEDES]->(p2)
        SET r.reason = $reason, r.created_at = datetime(), r += $meta
        """
    elif rel == "CONTRADICTS":
        return """
        MERGE (p1:Passage {passage_id: $subject_id})
        MERGE (p2:Passage {passage_id: $target_id})
        MERGE (p1)-[r:CONTRADICTS]->(p2)
        SET r.reason = $reason, r.created_at = datetime(), r += $meta
        """
    raise ValueError(f"Disallowed relationship type: {rel!r}")
```
Apply the same pattern for `record_claim_relationship`. This eliminates the f-string entirely and makes each Cypher string a static constant.

---

### D14-2 🟡 `contradiction.py` — Version sort uses `int(v) if v.isdigit()` as key, silently flattening non-numeric versions to 0

**File**: `contradiction.py`  
**Lines**: 70

```python
sorted_versions = sorted(
    passages_by_version.keys(),
    key=lambda v: int(v) if v.isdigit() else 0
)
```

**Problem**: Any non-numeric version key (e.g. `"unknown"`, `"beta"`, or the fallback `"16"` when `state.target_version` is `None`) gets mapped to `0`, making it sort *before* all numeric versions. The subsequent `v_older = sorted_versions[0]` / `v_newer = sorted_versions[-1]` selection then treats `"unknown"` as the *oldest* version. If a passage with `version=None` is grouped under `"unknown"` and another passage has `version="16"`, the comparison fires a cross-version path (`v_older="unknown"`, `v_newer="16"`) and may write a spurious SUPERSEDES edge into Neo4j.

**Scenario**: User asks a cross-version query but the retriever returns passages whose `version` payload was not set by the ingestor. Both passages land in `passages_by_version["unknown"]`. With only one key, `len(sorted_versions) < 2`, so we fall into the intra-version block — which is correct. **But** if one passage has `version="unknown"` and another has `version="16"`, sorted order is `["unknown" (key 0), "16" (key 16)]`, meaning `v_older="unknown"` and `v_newer="16"`. The comparison then extracts parameter values from an unknown-versioned passage and reports a `SUPERSEDES` from PG 16 over `"unknown"` which is then written to Neo4j as an edge.

**Better approach**:
```python
# Filter out non-numeric versions before comparing
numeric_versions = [v for v in passages_by_version if v.isdigit()]
sorted_versions = sorted(numeric_versions, key=int)
if len(sorted_versions) < 2:
    # fall through to intra-version / no-contradiction path
    ...
```

---

### D14-3 🟡 `contradiction.py` — LLM singleton ignores API-key rotation

**File**: `contradiction.py`  
**Lines**: 24–32

```python
_gemini_contradiction_client = None

def _get_contradiction_client(api_key: str):
    global _gemini_contradiction_client
    if _gemini_contradiction_client is None:
        from google import genai
        _gemini_contradiction_client = genai.Client(api_key=api_key)
    return _gemini_contradiction_client
```

**Problem**: The singleton is created with the **first** `api_key` ever passed. If the key is rotated (e.g. a secret is rolled in production, or integration tests inject a test key after a production key was used in a previous test), the new key is silently ignored. The same anti-pattern appears in `judge.py` (`_gemini_judge_client`) and `answer_writer.py` (`_gemini_writer_client`).

**Scenario**: CI rotates the `GEMINI_API_KEY` environment variable between test runs in the same process. The second test run always uses the stale key, causing `403 Invalid API key` errors that get swallowed into deterministic fallback, masking the rotation failure.

**Better approach**:
```python
_gemini_contradiction_client = None
_gemini_contradiction_key: str = ""

def _get_contradiction_client(api_key: str):
    global _gemini_contradiction_client, _gemini_contradiction_key
    if _gemini_contradiction_client is None or _gemini_contradiction_key != api_key:
        from google import genai
        _gemini_contradiction_client = genai.Client(api_key=api_key)
        _gemini_contradiction_key = api_key
    return _gemini_contradiction_client
```
Track the last key used alongside the client and invalidate when it changes.

---

### D14-4 🔵 `contradiction.py` — `analyze_cross_version_differences_deterministic` only compares the first passage per version

**File**: `contradiction.py`  
**Lines**: 96–100

```python
p_older = passages_by_version[v_older][0]   # always index [0]
p_newer = passages_by_version[v_newer][0]   # always index [0]
```

**Problem**: When multiple passages exist per version (which is common after RRF fusion returns 5–7 results), only the top-ranked passage per version is inspected. If the top passage contains no `default value is …` or `accepts …` text but a lower-ranked passage does, the deterministic analysis silently returns `NO_CONTRADICTION` instead of inspecting the richer passage.

**Better approach**: Iterate over all passages in each version bucket and pick the first one that actually contains extractable parameter information:

```python
def _best_passage_for_version(p_list):
    for p in p_list:
        info = extract_parameter_values_from_passage(p.get("content", ""))
        if info.get("default") or info.get("options"):
            return p, info
    return p_list[0], extract_parameter_values_from_passage(p_list[0].get("content", ""))

p_older, info_older = _best_passage_for_version(passages_by_version[v_older])
p_newer, info_newer = _best_passage_for_version(passages_by_version[v_newer])
```

---

## Day 15 — `judge.py` & `state.py`

---

### D15-1 🟡 `judge.py` — Trigger 5 fires on **every** query with ≥3 citations, defeating ADR-005

**File**: `judge.py`  
**Lines**: 93–96

```python
# Trigger 5: Complex multi-source synthesis with 3+ cited chunks
citations = state.citations or []
if len(citations) >= 3:
    return True
```

**Problem**: ADR-005 states the L2 Judge should activate for `< 10%` of queries. However, most real answer-writer outputs for multi-step parameters cite 3–5 chunks (e.g. `shared_buffers` has separate chunks for the value, the unit, and the context). This means virtually every detailed answer fires the L2 Judge, **doubling API costs and tripling latency** for a large fraction of non-contested queries.

**Scenario**: A user asks "What is the default value of shared_buffers in PostgreSQL 16?" The retriever returns 5 passages, the answer cites 3–4 of them. Trigger 5 fires unconditionally, calling Gemini a second time for a completely uncontested factual query that passed L1 deterministic verification without issue.

**Better approach**: Guard Trigger 5 with an additional condition that only fires when citations span *multiple different versions*:

```python
# Trigger 5: Multi-source synthesis only when sources span different versions
if len(citations) >= 3:
    cited_versions = {
        p.get("version") for p in passages
        if (p.get("chunk_id") or p.get("id")) in {
            c.chunk_id for c in citations if hasattr(c, "chunk_id")
        }
    }
    if len(cited_versions) > 1:  # Only if cross-version sources
        return True
```
Alternatively, raise the threshold to `>= 5` to match the documented `< 10%` target rate.

---

### D15-2 🟡 `judge.py` — `evaluate_l2_deterministic` returns `AFFIRMED` unconditionally when contradiction was detected, skipping actual answer-text check

**File**: `judge.py`  
**Lines**: 146–155

```python
if contradiction_details and contradiction_details.get("contradiction_detected"):
    rel = contradiction_details.get("relationship_type", "SUPERSEDES")
    if rel in {"SUPERSEDES", "CONTRADICTS"}:
        reason = contradiction_details.get("reason", "")
        return {
            "verdict": "AFFIRMED",
            "score": 0.95,
            "reasoning": f"Cross-version relationship ({rel}) accurately evaluated...",
            "revised_answer": None,
        }
```

**Problem**: The deterministic fallback judge unconditionally `AFFIRMED`s any answer where `contradiction_detected=True`, **without inspecting the draft answer at all**. This means that if the LLM API is unavailable and the answer_writer's deterministic fallback produced an `"Insufficient evidence..."` refusal string, the L2 deterministic judge would still return `AFFIRMED` with score `0.95`, causing the verifier to pass a non-answer through to the user.

**Scenario**: LLM API is down. `contradiction_detected=True` (from deterministic rule-based analysis). `answer_writer` deterministic fallback produces `"Insufficient evidence in the PostgreSQL 16 documentation."`. L2 deterministic judge sees `contradiction_detected=True`, short-circuits, returns `AFFIRMED`. Verifier passes. User receives a useless "Insufficient evidence" answer with a green verification badge.

**Better approach**:
```python
if contradiction_details and contradiction_details.get("contradiction_detected"):
    rel = contradiction_details.get("relationship_type", "SUPERSEDES")
    if rel in {"SUPERSEDES", "CONTRADICTS"}:
        # Only affirm if the draft answer is actually substantive
        if draft_answer and "insufficient evidence" not in draft_answer.lower():
            reason = contradiction_details.get("reason", "")
            return {
                "verdict": "AFFIRMED",
                "score": 0.95,
                "reasoning": f"Cross-version relationship ({rel}) evaluated. {reason}",
                "revised_answer": None,
            }
        # Fall through to full unsupported-value check if draft is a non-answer
```

---

### D15-3 🔵 `state.py` — `l2_passed` defaults to `False` but means "not yet evaluated"

**File**: `state.py`  
**Lines**: 56

```python
l2_passed: bool = Field(default=False, description="Whether the L2 Judge affirmed...")
```

**Problem**: The default value of `False` for `l2_passed` is semantically misleading. When `l2_invoked=False` (the judge was bypassed), `l2_passed=False` reads as "the judge rejected the answer" in audit trail queries or external dashboards, even though the judge was never called. `workflow.py`'s `route_after_judge` correctly gates on `state.l2_passed`, but downstream analytics that read the state directly will misinterpret the default.

**Better approach**: Use `Optional[bool]` with default `None` to represent "not evaluated":
```python
l2_passed: Optional[bool] = Field(
    default=None,
    description="Whether L2 Judge affirmed (True/False), or None if judge was not invoked"
)
```
Update `route_after_judge` to treat `None` as a pass when `l2_invoked=False`.

---

### D15-4 💡 `judge.py` — No timeout / deadline on the L2 LLM call

**File**: `judge.py`  
**Lines**: 283–290

```python
resp = client.models.generate_content(
    model=model_name,
    contents=prompt,
)
```

**Observation**: The L2 Judge LLM call has no timeout parameter, unlike the answer_writer which implements a `max_attempts=3` retry loop with progressive sleep. If the Gemini API stalls (e.g. HTTP streaming buffer full but response never ends), this call will hang indefinitely, blocking the LangGraph event loop thread. The answer_writer already demonstrates the correct retry + timeout pattern that should be replicated here.

---

## Day 16 — `comparator.py` & `benchmark_competitors.py`

---

### D16-1 🟠 `comparator.py` — `PgvectorTsvectorCompetitor.run_query` returns `is_grounded=True` whenever results are non-empty, masking answer quality

**File**: `comparator.py`  
**Lines**: 216–217

```python
is_grounded = bool(len(retrieved_ids) > 0)
is_abstained = False
```

**Problem**: The pgvector competitor's `is_grounded` is set to `True` whenever **any** chunk was retrieved, even if that chunk belongs to the wrong PostgreSQL version (which is precisely the post-filtering flaw the benchmark is designed to expose). The `calculate_competitor_metrics` function computes `groundedness_pct` from this flag:

```python
grounded_count = sum(1 for r in answerable if r.is_grounded)
groundedness_pct = (grounded_count / len(answerable)) * 100.0
```

This means pgvector's groundedness score will be nearly 100% in the aggregate metrics, **overstating its quality** and understating EvidenceFlow's comparative advantage. The benchmark's written narrative claims pgvector "fails" on version-filtered queries, but the generated metric table will show it as ~100% grounded.

**Scenario**: Running `python -m evidenceflow.eval.benchmark_competitors` and sharing the output table externally. The pgvector column shows `Groundedness: 95%`, which contradicts the architecture analysis text written in the same markdown report.

**Better approach**: Ground-truth grounding for all systems should require that at least one retrieved chunk actually matches the expected chunk IDs — the same criterion used by EvidenceFlow's L1 verifier. The harness in `run_single_competitor_benchmark` already computes `r_at_5`; use that as the grounded signal:

```python
# In run_single_competitor_benchmark, after computing r_at_5:
# Override is_grounded with the ground-truth recall signal
if item.is_answerable and item.expected_chunk_ids:
    is_grounded = r_at_5 > 0.0  # only grounded if correct chunk was retrieved
```

---

### D16-2 🟡 `comparator.py` — `_load_chunk_embeddings_cache` runs synchronously in `__init__`, bare `Exception` swallows all failures

**File**: `comparator.py`  
**Lines**: 118–132

```python
def _load_chunk_embeddings_cache(self, limit: int = 500):
    try:
        with psycopg.connect(self.conn_str, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT chunk_id, content FROM corpus_chunks LIMIT %s;", (limit,))
                rows = cur.fetchall()
        for r in rows:
            self._cached_embeddings[r["chunk_id"]] = get_embedding(r["content"][:300])
    except Exception as e:
        logger.warning("Failed to preload chunk embeddings: %s", e)
```

**Problem**: Loading 500 embeddings from PostgreSQL plus calling `get_embedding()` 500 times in the constructor blocks the calling thread for potentially tens of seconds if the embedding model is remote. The bare `Exception` catch silently leaves `_cached_embeddings` empty with no warning to the user that the competitor is now running with zero vectors, making all vector similarity scores `0.0`.

**Better approach**: Narrow the exception types and log the empty-cache consequence clearly:
```python
def _load_chunk_embeddings_cache(self, limit: int = 500):
    try:
        with psycopg.connect(self.conn_str, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT chunk_id, content FROM corpus_chunks LIMIT %s;", (limit,))
                rows = cur.fetchall()
        for r in rows:
            self._cached_embeddings[r["chunk_id"]] = get_embedding(r["content"][:300])
        logger.info("PgvectorCompetitor: preloaded %d chunk embeddings.", len(self._cached_embeddings))
    except psycopg.Error as db_err:
        logger.warning("DB error preloading embeddings (vector similarity will be 0 for all): %s", db_err)
    except Exception as embed_err:
        logger.warning("Embedding error during preload (vector similarity will be 0 for all): %s", embed_err)
```

---

### D16-3 🟡 `comparator.py` — Version inference for pgvector candidate chunks uses fragile chunk-ID substring matching

**File**: `comparator.py`  
**Lines**: 171

```python
v = "16" if "pg16" in cid else ("14" if "pg14" in cid else "15")
```

**Problem**: The version is inferred from whether the chunk ID string contains `"pg16"` or `"pg14"`, defaulting to `"15"` otherwise. Any chunk whose ID doesn't contain these exact substrings will be silently classified as version 15. This falsely elevates pgvector's version-filtered recall because misidentified chunks appear to match the target version.

**Better approach**: Read the `version` field from the actual PostgreSQL `corpus_chunks` row during the preload step:
```python
# In _load_chunk_embeddings_cache:
cur.execute("SELECT chunk_id, version, content FROM corpus_chunks LIMIT %s;", (limit,))
rows = cur.fetchall()
for r in rows:
    emb = get_embedding(r["content"][:300])
    self._cached_embeddings[r["chunk_id"]] = (emb, r["version"])

# In search_vector_naive_postfilter:
for cid, (emb, v) in self._cached_embeddings.items():
    sim = sum(a * b for a, b in zip(query_vec, emb))
    scored_chunks.append((cid, v, sim))
```

---

### D16-4 🟡 `comparator.py` — p50/p95 latency calculation uses incorrect index formula for small N

**File**: `comparator.py`  
**Lines**: 551–552

```python
latencies = sorted([r.latency_ms for r in results])
p50_latency = latencies[int(n * 0.50)]
p95_latency = latencies[min(int(n * 0.95), n - 1)]
```

**Problem**: For `n = 20` (the full benchmark), `int(20 * 0.50) = 10`, which is the **11th element** (0-indexed), not the median. The correct p50 for 20 samples should be the average of elements at indices 9 and 10. For `p95`, `int(20 * 0.95) = 19 = n - 1`, which is the maximum value, consistently overstating both latency statistics.

**Better approach**:
```python
import statistics
p50_latency = statistics.median(latencies)
p95_idx = max(0, int(math.ceil(n * 0.95)) - 1)
p95_latency = latencies[p95_idx]
```

---

### D16-5 🔵 `benchmark_competitors.py` — Report date uses local wall-clock time without timezone annotation

**File**: `benchmark_competitors.py`  
**Lines**: 30

```python
f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
```

**Problem**: `datetime.now()` returns local wall-clock time. On a CI runner (UTC) this is fine, but on a developer machine in IST (+5:30) the report embeds local time without timezone annotation, making reproducibility and cross-team comparison harder.

**Better approach**:
```python
f"**Date**: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
```

---

## Day 17 — `external_server.py`

---

### D17-1 🔴 `external_server.py` — Token bucket `acquire()` has a re-entrancy race condition

**File**: `external_server.py`  
**Lines**: 58–72

```python
async def acquire(self, tokens: float = 1.0, timeout: float = 5.0) -> bool:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        async with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
        # Sleep briefly before retrying token acquisition
        await asyncio.sleep(0.05)   # ← lock is NOT held during this sleep
    return False
```

**Problem**: The lock is **acquired and released inside the loop on every iteration**, then released before `asyncio.sleep(0.05)`. During that 50 ms sleep window the lock is **not held**. If two coroutines are simultaneously waiting in this loop, both can observe `self.tokens >= 1` after the same refill, acquire the lock in separate loop iterations, and both deduct a token — consuming two tokens when only one was available. This violates the core rate-limiting invariant.

**Concrete failure scenario**: 10 concurrent `get_release_metadata` calls arrive simultaneously. The bucket has exactly 5 tokens. All 10 enter `asyncio.sleep(0.05)` after their first check shows 0 tokens. When sleep expires, all 10 grab the lock serially, but each one runs `_refill()` and re-evaluates the token count. With the per-waiter refill accumulation (the refill is based on elapsed wall-clock time, not on lock hold time), more than the burst capacity may be consumed.

**Better approach**:
```python
async def acquire(self, tokens: float = 1.0, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        async with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            # Compute minimum wait time for enough tokens to accumulate
            deficit = tokens - self.tokens
            wait_time = deficit / self.rate

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(wait_time, remaining, 0.05))
```
This keeps the token-check and deduction atomic inside the lock, and sleeps the **computed wait time** rather than a fixed 50 ms polling interval.

---

### D17-2 🟠 `external_server.py` — `get_security_advisories` constructs `httpx.HTTPStatusError` with `request=None`, which causes `TypeError` on construction

**File**: `external_server.py`  
**Lines**: 333

```python
async def _fetch():
    if simulate_outage:
        raise httpx.HTTPStatusError(
            "503 Service Unavailable",
            request=None,       # ← httpx.HTTPStatusError requires a real Request object
            response=httpx.Response(503)
        )
```

**Problem**: `httpx.HTTPStatusError.__init__` takes `request: Request` as a required typed parameter. Passing `request=None` will cause a `TypeError` at construction time in modern `httpx` versions that enforce type annotations at runtime (or raise `AttributeError` when `execute_with_resilience` tries to log `http_err.request.url`). This is inconsistent with `get_release_metadata` which correctly uses `httpx.ConnectError` for outage simulation.

**Scenario**: A test call with `simulate_outage=True` on `get_security_advisories` raises `TypeError` on the `httpx.HTTPStatusError` constructor, which is caught by the bare `except Exception` branch in `execute_with_resilience`, gets retried `max_retries` times (each raising `TypeError`), and finally degrades to fallback — masking the bug as a network outage rather than a code error.

**Better approach**:
```python
if simulate_outage:
    raise httpx.ConnectError(
        "Simulated external API outage / connection refused for security advisories."
    )
```
Use `httpx.ConnectError` consistently across both tool handlers for outage simulation.

---

### D17-3 🟡 `external_server.py` — `execute_with_resilience` catches bare `Exception` and retries programming errors

**File**: `external_server.py`  
**Lines**: 236–238

```python
except Exception as unk_err:
    last_error = unk_err
    logger.warning("Unexpected error on attempt %d: %s", attempt + 1, unk_err)
```

**Problem**: The bare `Exception` catch retries **all** unknown errors, including programming errors (`AttributeError`, `TypeError`, `KeyError`) that will never succeed on retry. This wastes up to `max_retries * max_delay` seconds on bugs that should surface immediately. It also masks code defects behind "max retries exhausted" log messages.

**Better approach**:
```python
except Exception as unk_err:
    # Non-retryable unexpected error — surface immediately, do not retry
    logger.error("Unexpected non-retryable error on attempt %d: %s", attempt + 1, unk_err, exc_info=True)
    return None, attempt, True, f"Unexpected error: {unk_err}"
```

---

### D17-4 🟡 `external_server.py` — `check_version_eol_status` silently drops `retry_count` and `retries_exhausted` from inner response

**File**: `external_server.py`  
**Lines**: 396–401

```python
return ExternalToolResponse(
    success=True,
    data=analysis,
    is_degraded=meta_response.get("is_degraded", False),
    latency_ms=meta_response.get("latency_ms", 0.0),
    # ← retry_count and retries_exhausted silently dropped
).model_dump()
```

**Problem**: `retry_count` and `retries_exhausted` from the nested `get_release_metadata` call are silently dropped. A caller using `check_version_eol_status` has no way to know that the underlying release-metadata fetch required retries or is serving degraded fallback data.

**Better approach**:
```python
return ExternalToolResponse(
    success=True,
    data=analysis,
    is_degraded=meta_response.get("is_degraded", False),
    retries_exhausted=meta_response.get("retries_exhausted", False),
    retry_count=meta_response.get("retry_count", 0),
    latency_ms=meta_response.get("latency_ms", 0.0),
    error_notice=meta_response.get("error_notice"),
).model_dump()
```

---

### D17-5 🔵 `external_server.py` — Fallback metadata hardcodes stale minor versions and EOL dates

**File**: `external_server.py`  
**Lines**: 115–143

```python
FALLBACK_RELEASE_METADATA: Dict[str, Dict[str, Any]] = {
    "14": {"latest_minor": "14.13", "eol_date": "2026-11-12", ...},
    "15": {"latest_minor": "15.8",  "eol_date": "2027-11-11", ...},
    "16": {"latest_minor": "16.4",  "eol_date": "2028-11-09", ...},
}
```

**Problem**: PostgreSQL releases new minor versions every quarter. The hardcoded `latest_minor` and `critical_security_fixes_count` values will be stale within months. `eol_date` for PG 14 is `2026-11-12` — this date will pass, and PG 14 will show `is_supported: True` from the fallback when it should be `False`.

**Better approach**: Add a `FALLBACK_METADATA_AS_OF` timestamp and emit a staleness warning when the fallback is served:
```python
FALLBACK_METADATA_AS_OF = "2026-11-21"
# ...
error_notice=f"External API degraded: {err_msg}. Serving cached metadata as of {FALLBACK_METADATA_AS_OF}. Data may be outdated.",
```
Or load the fallback from a versioned `fallback_metadata.json` file updated by the release pipeline.

---

## Day 18 — `ci_gate.py` & `ci.yml`

---

### D18-1 🟠 `ci_gate.py` — `run_ci_gate()` ignores the `min_abstention_accuracy` parameter — CLI cannot override it

**File**: `ci_gate.py`  
**Lines**: 345–363

```python
def main():
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--groundedness-min", ...)
    parser.add_argument("--recall-min", ...)
    parser.add_argument("--max-cache-false-hits", ...)
    # ← No --abstention-min argument
    args = parser.parse_args()

    report = run_ci_gate(
        min_groundedness=min_groundedness,
        min_recall_at_5=min_recall,
        max_cache_false_hits=args.max_cache_false_hits,
        # ← min_abstention_accuracy NOT wired — always DEFAULT_MIN_ABSTENTION_ACCURACY
    )
```

**Problem**: The CLI has no `--abstention-accuracy-min` argument, and `run_ci_gate` is called without the `min_abstention_accuracy` kwarg in `main()`. This means: (1) the abstention accuracy threshold cannot be overridden from the command line; (2) the `--break-it` mode changes only `min_recall` to `0.999` but does not prove that abstention accuracy regression is catchable.

**Better approach**:
```python
parser.add_argument(
    "--abstention-min", type=float, default=DEFAULT_MIN_ABSTENTION_ACCURACY,
    help="Minimum acceptable abstention accuracy (default: 1.00)"
)
...
report = run_ci_gate(
    min_groundedness=min_groundedness,
    min_recall_at_5=min_recall,
    max_cache_false_hits=args.max_cache_false_hits,
    min_abstention_accuracy=args.abstention_min,  # ← wire it
)
```

---

### D18-2 🟠 `ci_gate.py` — `evaluate_retrieval_recall` queries live Qdrant with no corpus seeding, so CI always returns 0.0% recall

**File**: `ci_gate.py`  
**Lines**: 126–143

```python
def evaluate_retrieval_recall() -> float:
    qdrant = get_evidence_qdrant_client()
    hits_found = 0
    for item in CANONICAL_IR_REGRESSION_CASES:
        passages = qdrant.search(query=item["query"], version=item["version"], limit=5)
        retrieved_ids = {p.get("chunk_id") or p.get("id") for p in passages}
        if retrieved_ids & expected:
            hits_found += 1
    return hits_found / total if total > 0 else 0.0
```

**Problem**: The CI workflow's `ci.yml` starts a fresh Qdrant container (`qdrant/qdrant:v1.13.2`) with **no data**. No step in the workflow ingests the documentation corpus into Qdrant before running the CI gate. The Qdrant container will return empty results for all 10 regression queries, causing `evaluate_retrieval_recall()` to return `0.0`, which is `< 0.85` (the threshold), causing the gate to **always fail** regardless of code quality. This makes the CI gate useless — a permanent false-failure that developers learn to ignore.

**Scenario**: Every CI run on a fresh runner fails the Recall@5 gate with `0.0% < 85.0%`, blocking all merges not because of a regression but because the test corpus is empty.

**Better approach**: Add a corpus ingestion step to `ci.yml` before the CI gate step:
```yaml
- name: Seed CI Qdrant corpus
  env:
    QDRANT_HOST: localhost
    QDRANT_PORT: 6333
  run: uv run python -m evidenceflow.storage.ingest --subset-ci
```
Or add a `--skip-recall` flag to `ci_gate.py` that substitutes a BM25 recall evaluation (which uses an already-seeded disk index) when the Qdrant collection is empty.

---

### D18-3 🟡 `ci_gate.py` — `evaluate_cache_false_hit_rate` Probe 3 conflates cache amnesia with false hits

**File**: `ci_gate.py`  
**Lines**: 209–215

```python
# Probe 3: Legitimate hit should succeed
lookup_legit = cache_mgr.get_exact(
    query="What is work_mem?",
    target_versions="16",
    config_version="cfg_v1",
)
if lookup_legit is None:
    false_hits += 1  # Failed valid hit  ← mislabelled as false hit
```

**Problem**: A cache miss on a legitimately-cached entry is counted as a `false_hit`, which conflates two distinct failure modes: cache *collision* (a hit that should be a miss) and cache *amnesia* (a miss that should be a hit). If `ScopedCacheManager.set_exact()` at line 183 fails silently (e.g. the cache backend is in read-only mode), Probe 3 will penalise it as a false hit, inflating the false-hit rate to 25% and failing the gate for a reason unrelated to collision correctness.

**Better approach**: Separate the two concerns and only gate on collision rate:
```python
cache_amnesia = 0  # misses on valid entries (tracked but not gated)
# Probe 3: Legitimate hit should succeed
if lookup_legit is None:
    cache_amnesia += 1
    logger.warning("Cache amnesia: legitimate hit was not served (possible set() failure)")

# false_hits only counts Probes 1, 2, 4 (collision / stale leaks)
return false_hits / (total_probes - 1)  # exclude Probe 3 from gate metric
```

---

### D18-4 🟡 `ci.yml` — Qdrant health check uses `curl` which is absent from the `qdrant/qdrant:v1.13.2` image

**File**: `ci.yml`  
**Lines**: 33–37

```yaml
qdrant:
  image: qdrant/qdrant:v1.13.2
  options: >-
    --health-cmd "curl -s -f http://localhost:6333/healthz || exit 1"
```

**Problem**: The `qdrant/qdrant:v1.13.2` Docker image is built on a minimal base (Distroless or Alpine without curl). The health check command will fail with `exec: "curl": executable file not found`, causing the container to report perpetual `unhealthy` status. GitHub Actions will then wait for all `--health-retries 5` attempts (50 seconds) before continuing, and downstream steps will fail to connect.

**Scenario**: Every CI run takes 50+ additional seconds waiting for Qdrant, then all Qdrant-dependent steps fail because the container's health was never confirmed.

**Better approach**: Use a Python-based health check in a separate step (after `uv sync`) or use `wget` if available:
```yaml
options: >-
  --health-cmd "wget -qO- http://localhost:6333/healthz || exit 1"
  --health-interval 10s
  --health-timeout 5s
  --health-retries 5
```
Or add a post-install readiness wait step:
```yaml
- name: Wait for Qdrant readiness
  run: |
    for i in $(seq 1 10); do
      python -c "import urllib.request; urllib.request.urlopen('http://localhost:6333/healthz')" && break || sleep 5
    done
```

---

### D18-5 🔵 `ci.yml` — PostgreSQL and Neo4j credentials are hardcoded in the committed YAML file

**File**: `ci.yml`  
**Lines**: 17–21, 42

```yaml
env:
  POSTGRES_PASSWORD: evidenceflow_dev_secret   # ← hardcoded in git
  ...
  NEO4J_AUTH: neo4j/evidenceflow_dev_secret    # ← hardcoded in git
```

**Problem**: The passwords `evidenceflow_dev_secret` are hardcoded in a file committed to version control. While these are CI-only credentials, the pattern establishes a habit where real credentials could accidentally be added the same way. GitHub's secret scanning will also flag this in public repositories.

**Better approach**:
```yaml
env:
  POSTGRES_PASSWORD: ${{ secrets.CI_POSTGRES_PASSWORD }}
  ...
  NEO4J_AUTH: neo4j/${{ secrets.CI_NEO4J_PASSWORD }}
```
Add `CI_POSTGRES_PASSWORD: evidenceflow_dev_secret` and `CI_NEO4J_PASSWORD: evidenceflow_dev_secret` as GitHub repository secrets. This is a one-time setup that eliminates the credentials from the committed file.

---

## Summary Table

| ID | File | Severity | Title |
|---|---|---|---|
| D14-1 | `graph_client.py` | 🟠 HIGH | Cypher injection via f-string in `record_claim_relationship` / `record_passage_relationship` |
| D14-2 | `contradiction.py` | 🟡 MEDIUM | Non-numeric version keys silently sort to position 0, causing spurious SUPERSEDES edges |
| D14-3 | `contradiction.py` | 🟡 MEDIUM | LLM singleton ignores API-key rotation — stale key used after secret roll |
| D14-4 | `contradiction.py` | 🔵 LOW | Deterministic analysis only inspects `[0]` passage per version, misses richer lower-ranked passages |
| D15-1 | `judge.py` | 🟡 MEDIUM | Trigger 5 fires on ≥3 citations unconditionally, violating ADR-005 `< 10%` activation target |
| D15-2 | `judge.py` | 🟡 MEDIUM | Deterministic fallback affirms answers unconditionally when contradiction detected, without inspecting draft answer |
| D15-3 | `state.py` | 🔵 LOW | `l2_passed` defaults to `False` but semantically means "not yet evaluated" — misleads dashboards |
| D15-4 | `judge.py` | 💡 OBSERVATION | No timeout on L2 LLM call; answer_writer's retry pattern not replicated |
| D16-1 | `comparator.py` | 🟠 HIGH | pgvector `is_grounded=True` whenever results are non-empty, overstating groundedness metric |
| D16-2 | `comparator.py` | 🟡 MEDIUM | `_load_chunk_embeddings_cache` runs synchronously in `__init__`; bare `Exception` swallows all failures |
| D16-3 | `comparator.py` | 🟡 MEDIUM | Version inference from chunk ID substring is fragile — defaults non-matching IDs to version "15" |
| D16-4 | `comparator.py` | 🟡 MEDIUM | p50/p95 latency calculation uses incorrect index formula for small N |
| D16-5 | `benchmark_competitors.py` | 🔵 LOW | Report date uses local wall-clock time without timezone annotation |
| D17-1 | `external_server.py` | 🔴 CRITICAL | Token bucket `acquire()` releases lock before sleep — race condition allows over-consumption |
| D17-2 | `external_server.py` | 🟠 HIGH | `get_security_advisories` constructs `httpx.HTTPStatusError(request=None)`, raises `TypeError` on construction |
| D17-3 | `external_server.py` | 🟡 MEDIUM | Bare `Exception` catch retries programming errors (AttributeError, TypeError) instead of surfacing them |
| D17-4 | `external_server.py` | 🟡 MEDIUM | `check_version_eol_status` silently drops `retry_count` and `retries_exhausted` from inner response |
| D17-5 | `external_server.py` | 🔵 LOW | Fallback metadata hardcodes stale minor versions and EOL dates that will expire over time |
| D18-1 | `ci_gate.py` | 🟠 HIGH | `run_ci_gate()` ignores `min_abstention_accuracy` parameter — CLI cannot override it |
| D18-2 | `ci_gate.py` | 🟠 HIGH | `evaluate_retrieval_recall` queries empty CI Qdrant container — always returns 0.0%, always fails gate |
| D18-3 | `ci_gate.py` | 🟡 MEDIUM | Probe 3 conflates cache amnesia with false hits — inflates false-hit rate on backend failures |
| D18-4 | `ci.yml` | 🟡 MEDIUM | Qdrant health check uses `curl` absent from Distroless image — container stays unhealthy |
| D18-5 | `ci.yml` | 🔵 LOW | PostgreSQL and Neo4j passwords hardcoded in committed YAML instead of GitHub Secrets |

**Total findings: 23**  
🔴 Critical: 1 | 🟠 High: 5 | 🟡 Medium: 11 | 🔵 Low: 5 | 💡 Observation: 1
