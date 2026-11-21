# EvidenceFlow AI — Code Review: Days 4–7

**Reviewer:** Principal AI Architect  
**Date:** 2026-11-21  
**Scope:** Day 4 (Abstention, Input Guardrails, L1.5 Value-Binding Verifier) · Day 5 (PostgreSQL Checkpointer, ACID Audit Ledger) · Day 6 (FastMCP Corpus Server) · Day 7 (Benchmark v1 Metrics & Runner)  
**Rule:** Read-only review. No files have been changed.

---

## Overall Assessment

The Days 4–7 body of work represents a substantial, well-structured expansion of the EvidenceFlow AI system. The L1.5/L2 value-binding verifier is architecturally ambitious and largely correct; the ACID ledger design is production-aware; and the FastMCP server shows disciplined input validation. That said, the review surfaces **24 findings** across a range of severities. The most dangerous are a **race condition in the ACID step-counter** that can assign duplicate step numbers under concurrent load, **silent credential fall-back** in the connection pool that can mask misconfiguration and elevate privileges in production, a **degenerate citation-recall implementation** in `metrics.py` that always returns `1.0` regardless of actual coverage, a **`NameError` crash** in the benchmark runner on any `should_abstain=True` question, and **direct mutation of the global `Settings` singleton** that is not thread-safe. The benchmark runner also lacks a schema-migrations tracking table, meaning all SQL files re-execute on every invocation.

---

## Severity Legend

| Symbol | Meaning |
|--------|---------|
| 🔴 | **CRITICAL** – Will cause incorrect behaviour, data corruption, or production failures |
| 🟠 | **HIGH** – Significant performance or reliability issue |
| 🟡 | **MEDIUM** – Correctness risk in edge cases or maintainability concern |
| 🔵 | **LOW** – Style, naming, or minor improvement |
| 💡 | **OBSERVATION** – Not a defect; architectural note or improvement opportunity |

---

## Day 4: Guardrails & Abstention

### `guardrails/input_guardrail.py`

---

#### D4-1 🟡 `input_guardrail.py` — Version regex over-matches single-digit numerals in prose

**File:** `guardrails/input_guardrail.py` · Lines 57–66

```python
version_match = re.search(
    r"\b(?:postgres(?:ql)?|pg|v|version|release)\s*(\d{1,2})\b",
    cleaned, re.IGNORECASE
)
if version_match:
    requested_v = version_match.group(1)
    if requested_v not in SUPPORTED_VERSIONS:
        return (
            False,
            "UNSUPPORTED_VERSION",
            f"Unsupported Version: PostgreSQL version {requested_v} is not indexed. ..."
        )
```

**Problem:** The version group `(\d{1,2})` captures any 1–2 digit number that follows a triggering keyword. The trigger includes the bare letter `v`, which appears in everyday prose. A query like *"What is the WAV format for pg_basebackup output on v9 boxes?"* would extract `9` and block with `UNSUPPORTED_VERSION`. Worse, the pattern `release\s*(\d{1,2})` can match a release-note section number such as "Release 8" in copied documentation text.

**Scenario:** A DBA pastes a snippet from an old migration guide asking about behaviour that has carried forward into version 16. The snippet mentions `pg 9.6` compatibility notes. The query is legitimate for pg 16 but is blocked because "pg" + "9" matches the version regex, returning `UNSUPPORTED_VERSION` with a misleading message.

**Better approach:**
```python
# Require more specific version markers — exclude lone "v"
version_match = re.search(
    r"\b(?:postgres(?:ql)?\s*(?:version\s*)?|pg\s*v?|postgresql\s+v?)(\d{1,2})\b",
    cleaned, re.IGNORECASE
)
# Also require the number to be in a plausible DB-version range (8–17)
if version_match:
    requested_v = version_match.group(1)
    if int(requested_v) in range(8, 18) and requested_v not in SUPPORTED_VERSIONS:
        return (False, "UNSUPPORTED_VERSION", ...)
```

---

#### D4-2 🔵 `input_guardrail.py` — Duplicate casual-pattern compilation across two functions

**File:** `guardrails/input_guardrail.py` · Lines 80–86 and 113–119

```python
# In evaluate_input_guardrails (lines 80-86):
casual_patterns = [
    re.compile(r"^(?:hello|hi|hey|good\s+morning|good\s+evening)\b", re.IGNORECASE),
    re.compile(r"\b(?:what\s+is\s+the\s+meaning\s+of\s+life|who\s+are\s+you)\b", re.IGNORECASE),
]

# In check_pre_resolution_guards (lines 113-119): identical block
casual_patterns = [
    re.compile(r"^(?:hello|hi|hey|good\s+morning|good\s+evening)\b", re.IGNORECASE),
    re.compile(r"\b(?:what\s+is\s+the\s+meaning\s+of\s+life|who\s+are\s+you)\b", re.IGNORECASE),
]
```

**Problem:** The same two patterns are compiled fresh inside each function call. This wastes CPU on every request and creates a maintenance divergence risk — a future change to one block may not be mirrored in the other.

**Better approach:** Promote to module-level constants alongside `INJECTION_PATTERNS`:
```python
CASUAL_PATTERNS = [
    re.compile(r"^(?:hello|hi|hey|good\s+morning|good\s+evening)\b", re.IGNORECASE),
    re.compile(r"\b(?:what\s+is\s+the\s+meaning\s+of\s+life|who\s+are\s+you)\b", re.IGNORECASE),
]
```

---

### `graph/nodes/abstention.py`

---

#### D4-3 🟡 `abstention.py` — `l1_passed: True` on abstention is semantically misleading

**File:** `graph/nodes/abstention.py` · Line 87

```python
return {
    "draft_answer": formatted_answer,
    "citations": [],
    "is_abstained": True,
    "l1_passed": True,  # Abstention is safe, compliant termination
    ...
}
```

**Problem:** `l1_passed` has the documented meaning: *"Whether L1 deterministic citation-existence check passed"* (`state.py` L49). Setting it to `True` on an abstention path where no citation check was actually performed is semantically incorrect. Any downstream consumer that branches on `l1_passed and not is_abstained` is robust, but audit queries against `step_ledger` or `claim_audit` that filter `WHERE l1_passed = TRUE` will include abstention rows, inflating the "pass" rate and confusing operational dashboards.

**Scenario:** An ops team builds a Grafana panel showing `l1_passed` percentage as a proxy for answer quality. Abstentions — which carry no citations at all — are counted as "passed" citation checks, making the dashboard mislead operations into thinking groundedness is higher than reality.

**Better approach:** Set `l1_passed` to `False` for abstentions, relying on `is_abstained` to route:
```python
return {
    "draft_answer": formatted_answer,
    "citations": [],
    "is_abstained": True,
    "l1_passed": False,   # No L1 check was run; abstention is not a citation pass
    ...
}
```
If the routing condition in `workflow.py` (`not state.l1_passed or state.is_abstained`) is updated accordingly, behaviour is unchanged while semantics are correct.

---

### `graph/nodes/verifier.py`

---

#### D4-4 🟡 `verifier.py` — Multi-chunk value verification exits on first failure even when another cited chunk supports the value

**File:** `graph/nodes/verifier.py` · Lines 539–562

```python
for cid in cids_to_verify:
    passage_content = passage_map.get(cid, "")
    missing, err_detail = verify_statement_value_presence(stmt_text, passage_content)
    if missing or err_detail:
        fail_reason = (
            f"L1.5/L2 Value Verification Failed: {err_detail or missing} in source passage '{cid}'."
        )
        ...
        return {
            "l1_passed": False,
            "is_abstained": True,
            "abstention_code": "VALUE_MISMATCH",
            ...
        }
```

**Problem:** When a statement cites multiple chunks (e.g. `[^pg16-fsync, ^pg16-wal-sync-method]`), `cids_to_verify` is built from targeted matching, but the loop still returns `VALUE_MISMATCH` on the **first** chunk that does not contain the value — even though the second chunk might contain it. The correct semantics for multi-chunk citation is: the value should be verifiable in **at least one** of the cited chunks.

**Scenario:** A correct answer says *"fsync defaults to `on`"* and cites `[^pg16-fsync, ^pg16-wal-level]`. `pg16-wal-level` is targeted first (or first in `cids_to_verify`) and doesn't mention `on` as a boolean default for `fsync`. The verifier fails with `VALUE_MISMATCH` on a correct, grounded answer.

**Better approach:**
```python
# For multi-chunk citations, the value only needs to be present in ONE cited chunk
any_pass = False
last_fail_detail = None
for cid in cids_to_verify:
    passage_content = passage_map.get(cid, "")
    missing, err_detail = verify_statement_value_presence(stmt_text, passage_content)
    if not missing and not err_detail:
        any_pass = True
        break
    last_fail_detail = err_detail or str(missing)

if not any_pass and last_fail_detail:
    # all cited chunks failed — trigger VALUE_MISMATCH
    ...
```

---

#### D4-5 🔵 `verifier.py` — `split_sentences_safely` lookbehind references `\bp` inconsistently with the other lookbehinds

**File:** `graph/nodes/verifier.py` · Line 151

```python
split_pattern = re.compile(
    r"(?:\n\s*\n+|(?<!\d)(?<!\be\.g)(?<!\bi\.e)(?<!\betc)(?<!\bvs)(?<!\bp)\.(?!\d)..."
    ...
)
```

**Problem:** The lookbehind `(?<!\bp)` is intended to avoid splitting on abbreviations starting with `p`, but `\b` inside a fixed-width lookbehind in Python `re` checks a word-boundary position before `p`. The other lookbehinds — `(?<!\be\.g)`, `(?<!\bi\.e)` — omit the final character of the abbreviation to work around fixed-width constraints. `(?<!\bp)` is inconsistent in structure: it protects only the single character `p`, not abbreviations like `pp.` or `p.s.`. This is a subtle regex correctness issue that may cause incorrect sentence splits on multi-letter abbreviations starting with `p`.

**Better approach:** Replace `(?<!\bp)` with `(?<!p)` (without `\b`, consistent with the fixed-width constraint) or add specific abbreviations. Consider replacing the custom splitter with `nltk.sent_tokenize` for correctness.

---

#### D4-6 💡 `verifier.py` — `get_equivalent_quantities` uses `elif` for memory branch chained to time ladder

**File:** `graph/nodes/verifier.py` · Lines 66–79

```python
        # Time equivalences
        if unit == "min" and int_num is not None:
            ...
        elif unit == "s" and int_num is not None:
            ...
        elif unit == "ms" and int_num is not None:
            ...

        # Memory equivalences (Postgres base 1024)
        elif unit == "mb" and int_num is not None:
```

**Observation:** The memory branch (`elif unit == "mb"`) is chained to the time `elif` ladder. This is functionally correct (since a unit cannot be both `ms` and `mb`), but the mid-chain comment `# Memory equivalences` creates reader confusion and increases the risk of breakage if a new time unit is inserted. Splitting into a separate `if`/`elif` block for memory would improve clarity.

---

## Day 5: PostgreSQL Checkpointer & Ledger

### `storage/ledger.py`

---

#### D5-1 🔴 `ledger.py` — `get_next_step_number` is not atomic — race condition under concurrent requests

**File:** `storage/ledger.py` · Lines 107–125

```python
def get_next_step_number(thread_id: str) -> int:
    try:
        pool = get_connection_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COALESCE(MAX(step_number), 0) + 1 AS next_step FROM step_ledger WHERE thread_id = %s;",
                    (thread_id,)
                )
                row = cur.fetchone()
                if row:
                    return int(row["next_step"]) if isinstance(row, dict) else int(row[0])
                return 1
    except Exception:
        return 1
```

**Problem:** This is a classic time-of-check / time-of-use (TOCTOU) race. Between the `SELECT MAX()` and the subsequent `INSERT` in `record_step`, another concurrent request on the same `thread_id` can read the same `MAX` value and generate the same `step_number`. Both rows will be inserted with the same `step_number`, violating the "strictly monotonic counter" guarantee stated in the docstring. The bare `except Exception: return 1` also silently swallows database errors (connection failures, permission errors) and resets the counter to `1`, causing step number collisions if rows already exist.

**Scenario:** An API server handles two simultaneous queries on the same `thread_id`. Both `audit_wrapper.py` calls compute `step_number = 5`. Both `INSERT` statements succeed (no unique constraint exists on `(thread_id, step_number)` in SQL). The `step_ledger` now has two rows with `step_number = 5`, making audit replay non-deterministic.

**Better approach:** Use a PostgreSQL sequence or advisory lock:
```sql
-- In migration SQL: add a global or per-thread sequence
CREATE SEQUENCE IF NOT EXISTS step_ledger_step_seq;
```
```python
# In record_step, use nextval instead of SELECT MAX()+1:
INSERT INTO step_ledger (step_id, turn_id, thread_id, step_number, ...)
VALUES (%s, %s, %s, nextval('step_ledger_step_seq'), ...)
```
Or add a `UNIQUE(thread_id, step_number)` constraint and handle `UniqueViolation` with a retry loop.

---

#### D5-2 🔴 `ledger.py` — Silent credential fall-back elevates privileges to DDL owner role in production

**File:** `storage/ledger.py` · Lines 50–62

```python
try:
    _pool_instance = ConnectionPool(
        conninfo=conn_str,     # limited app role
        ...
    )
except Exception as e:
    if use_app_role:
        logger.warning("Failed to connect using limited app role (%s), falling back to owner role.", e)
        conn_str = get_postgres_connection_string(use_app_role=False, dbname=dbname)
        _pool_instance = ConnectionPool(
            conninfo=conn_str,  # OWNER / DDL role
            ...
        )
    else:
        raise e
```

**Problem:** When the limited `evidenceflow_app` role fails to connect (e.g., due to a misconfiguration or a temporary network blip), the pool **silently falls back to the full DDL owner role**. The RBAC boundary between the runtime app role and the migration owner role collapses silently. The `logger.warning` will be missed in a noisy log stream.

**Scenario:** An ops engineer rotates the `evidenceflow_app` password but forgets to update the secret in the deployment config. All production API pods immediately fall back to the owner role with full DDL privileges. If compromised, an attacker gains full database write access; if an operator bug triggers `DROP TABLE`, there is no protection.

**Better approach:** Remove the automatic fall-back entirely. If the app role fails, raise immediately:
```python
try:
    _pool_instance = ConnectionPool(conninfo=conn_str, ...)
except Exception as e:
    logger.critical(
        "FATAL: Cannot connect to PostgreSQL as app role '%s': %s — "
        "Refusing to fall back to owner role in production.",
        get_settings().POSTGRES_APP_USER, e
    )
    raise RuntimeError(
        "Database connection failed for app role. Check POSTGRES_APP_USER/PASSWORD."
    ) from e
```

---

#### D5-3 🟠 `ledger.py` — `record_turn` return value is misleading when `ON CONFLICT DO NOTHING` fires

**File:** `storage/ledger.py` · Lines 142–166

```python
cur.execute(
    """
    INSERT INTO turns (turn_id, ...) VALUES (%s, ...)
    ON CONFLICT (turn_id) DO NOTHING
    RETURNING turn_id;
    """,
    (turn_id, ...)
)
# No fetchone() call — RETURNING is ignored
return turn_id   # always returns the input turn_id
```

**Problem:** The `RETURNING turn_id` clause is never read (no `fetchone()`). When `ON CONFLICT DO NOTHING` fires (duplicate `turn_id`), `RETURNING` returns no rows and the Python code unconditionally returns `turn_id` as if the insert succeeded. The caller in `workflow.py` has no way to distinguish a first-time insert from a silently-skipped duplicate.

**Better approach:**
```python
cur.execute("INSERT INTO turns (...) VALUES (...) ON CONFLICT (turn_id) DO NOTHING RETURNING turn_id;", (...))
row = cur.fetchone()
if row is None:
    logger.debug("turn_id %s already exists in turns table, skipping duplicate insert.", turn_id)
return turn_id
```

---

#### D5-4 🟠 `ledger.py` — `sanitize_payload` token-protection heuristic can pass through sensitive keys containing the word "token"

**File:** `storage/ledger.py` · Lines 93–95

```python
if k_str in token_count_keys or (("token" in k_str or "count" in k_str) and isinstance(v, (int, float))):
    clean[k] = v
```

**Problem:** The heuristic `"token" in k_str` to protect numeric token counts also protects any key containing the word "token" with an integer value from redaction. A payload key like `"access_token_expiry_seconds": 3600` would be passed through unredacted because `"token" in "access_token_expiry_seconds"` is `True` and `v` is an `int`.

**Better approach:** Match only the explicit set `token_count_keys` by exact name, not substring:
```python
if k_str in token_count_keys:
    clean[k] = v
elif any(p in k_str for p in sensitive_patterns):
    clean[k] = "[REDACTED]"
else:
    clean[k] = sanitize_payload(v)
```

---

#### D5-5 🟡 `ledger.py` — `fetch_turns`, `fetch_steps`, `fetch_claims` expose `SELECT *` without column allowlist

**File:** `storage/ledger.py` · Lines 264–309

```python
def fetch_turns(thread_id: str) -> List[Dict[str, Any]]:
    ...
    cur.execute(
        "SELECT * FROM turns WHERE thread_id = %s ORDER BY turn_number ASC, created_at ASC;",
        (thread_id,),
    )
    return cur.fetchall()
```

**Problem:** `SELECT *` returns all columns including any future columns added by migrations, which may include sensitive data. If a migration adds (e.g.) an `ip_address` or `user_agent` column, it is automatically included in all API responses that call these fetch functions, without any review gate.

**Better approach:** Enumerate the columns explicitly in all three `fetch_*` functions.

---

### `storage/migrate_postgres.py`

---

#### D5-6 🟠 `migrate_postgres.py` — Migration runner applies all `.sql` files with `autocommit=True`, no idempotency tracking

**File:** `storage/migrate_postgres.py` · Lines 63–70

```python
sql_files = sorted(m_dir.glob("*.sql"))
with psycopg.connect(conn_str, autocommit=True) as conn:
    with conn.cursor() as cur:
        for sql_file in sql_files:
            logger.info("Applying migration: %s ...", sql_file.name)
            sql_content = sql_file.read_text(encoding="utf-8")
            cur.execute(sql_content)
            applied.append(sql_file.name)
```

**Problem:** There is no migration tracking table (e.g., `schema_migrations`). Every call to `run_migrations()` re-executes **all** SQL files from scratch. With `autocommit=True`, there is no transaction rollback when a mid-script error occurs — later files are skipped but earlier files in the same run have already been committed permanently.

**Scenario:** A developer runs `migrate_postgres.py` on a live database after adding a new migration file. The first nine migration files re-execute; file 3 contains `ALTER TABLE turns ADD COLUMN user_agent TEXT`, which fails with `column already exists`. Because `autocommit=True`, columns added by files 1–2 in this run are already committed. The process exits with an error mid-way with partially applied state.

**Better approach:** Implement a `schema_migrations` tracking table:
```sql
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ DEFAULT NOW()
);
```
```python
cur.execute("SELECT filename FROM schema_migrations;")
applied_set = {row["filename"] for row in cur.fetchall()}
for sql_file in sql_files:
    if sql_file.name in applied_set:
        logger.info("Skipping already-applied migration: %s", sql_file.name)
        continue
    cur.execute(sql_content)
    cur.execute("INSERT INTO schema_migrations (filename) VALUES (%s);", (sql_file.name,))
```

---

#### D5-7 🟡 `migrate_postgres.py` — `drop_all_tables` logs `actual_connected_db` which may be unbound if `fetchone` returns None

**File:** `storage/migrate_postgres.py` · Lines 93–117

```python
cur.execute("SELECT current_database();")
row = cur.fetchone()
actual_connected_db = str(row[0]) if row else ""
if not actual_connected_db.endswith("_test"):
    raise RuntimeError(...)

# ... (drops happen) ...

logger.info("All EvidenceFlow PostgreSQL tables dropped on test database '%s'.", actual_connected_db)
```

**Problem:** If `fetchone()` returns `None` (structurally possible), `actual_connected_db` is `""` and the safety check `"".endswith("_test")` is `False`, which raises — but the log at the end references `actual_connected_db` in the success path. The logic works in practice but is structurally fragile.

**Better approach:**
```python
cur.execute("SELECT current_database();")
row = cur.fetchone()
if not row:
    raise RuntimeError("Could not determine current database from PostgreSQL session.")
actual_connected_db = str(row[0])
if not actual_connected_db.endswith("_test"):
    raise RuntimeError(...)
```

---

## Day 6: FastMCP Corpus Server

### `mcp/corpus_server.py`

---

#### D6-1 🟠 `corpus_server.py` — `search_passages` performs redundant length validation after Pydantic already enforces `max_length=500`

**File:** `mcp/corpus_server.py` · Lines 104–109

```python
query_clean = query.strip()
if len(query_clean) > 500:
    return {"status": "error", "error": f"Query length ({len(query_clean)}) exceeds maximum allowed (500 chars)."}
```

**Problem:** `QueryStr` is `Annotated[str, Field(max_length=500, ...)]`. FastMCP validates this at the tool boundary before `search_passages` body executes. The manual length check is dead code for all well-formed MCP calls and creates a subtle asymmetry: Pydantic's `max_length` applies to the raw input, but the manual check applies *after* `.strip()`. This signals that the developer is not trusting their own schema, which may lead to future duplicated or inconsistent validation logic.

**Better approach:** Remove the redundant manual check and rely on Pydantic. Add a comment explaining that validation is handled by the annotated type.

---

#### D6-2 🟠 `corpus_server.py` — `diff_sections` does not validate that `v1 != v2`, enabling no-op diffs

**File:** `mcp/corpus_server.py` · Lines 257–262

```python
v1_str, v2_str = str(v1).strip(), str(v2).strip()
if v1_str not in VALID_VERSIONS or v2_str not in VALID_VERSIONS:
    return {
        "status": "error",
        "error": f"Invalid version pair ...",
    }
```

**Problem:** There is no check that `v1 != v2`. A caller can invoke `diff_sections(setting="work_mem", v1="16", v2="16")`. Both chunks will be fetched for the same version, the content will be identical, `changed` will be `False`, and the tool will return a successful "unchanged" result. This is misleading — the user has compared a parameter to itself.

**Better approach:**
```python
if v1_str == v2_str:
    return {
        "status": "error",
        "error": f"v1 and v2 must be different versions. Both are '{v1_str}'.",
    }
```

---

#### D6-3 🟡 `corpus_server.py` — `_find_parameter_chunk` swallows all Qdrant errors silently

**File:** `mcp/corpus_server.py` · Lines 51–84

```python
def _find_parameter_chunk(setting: str, version: str) -> Optional[Dict[str, Any]]:
    client = get_evidence_qdrant_client()
    ...
    candidates = client.search(query=clean_setting, version=version, limit=5)
    if not candidates:
        return None
```

**Problem:** If `client.search()` raises an exception (Qdrant connection failure, timeout, authentication error), the exception propagates unhandled into `diff_sections`, which calls `_find_parameter_chunk` without a try/except. `diff_sections` itself also has no try/except. The FastMCP framework may output a Python traceback to stdout, corrupting the JSON-RPC framing and breaking all subsequent tool calls in the session.

**Better approach:**
```python
def _find_parameter_chunk(setting: str, version: str) -> Optional[Dict[str, Any]]:
    try:
        client = get_evidence_qdrant_client()
        candidates = client.search(query=setting.strip().lower(), version=version, limit=5)
        ...
    except Exception as e:
        logger.error("_find_parameter_chunk failed for setting='%s' version='%s': %s", setting, version, e)
        return None
```
And wrap `diff_sections` body in a try/except that returns `{"status": "error", "error": str(e)}`.

---

#### D6-4 🔵 `corpus_server.py` — `list_versions` hardcodes version details instead of deriving from `VALID_VERSIONS`

**File:** `mcp/corpus_server.py` · Lines 215–230

```python
VALID_VERSIONS = {"14", "15", "16"}
...
def list_versions() -> Dict[str, Any]:
    return {
        "supported_versions": sorted(list(VALID_VERSIONS)),
        "version_details": {
            "14": {"status": "indexed", ...},
            "15": {"status": "indexed", ...},
            "16": {"status": "indexed", ...},
        },
    }
```

**Problem:** `version_details` is hardcoded with keys `"14"`, `"15"`, `"16"`. When `VALID_VERSIONS` is updated (e.g., adding `"17"`), `version_details` will silently omit the new version from the dict, while `supported_versions` will include it.

**Better approach:**
```python
def list_versions() -> Dict[str, Any]:
    version_details = {
        v: {"status": "indexed", "scope": f"PostgreSQL {v}.x official documentation"}
        for v in VALID_VERSIONS
    }
    return {
        "supported_versions": sorted(list(VALID_VERSIONS)),
        "default_version": max(VALID_VERSIONS),
        "version_details": version_details,
    }
```

---

#### D6-5 💡 `corpus_server.py` — `diff_sections` unified diff output is unbounded in size

**File:** `mcp/corpus_server.py` · Lines 346–354

```python
diff_lines = list(
    difflib.unified_diff(
        content_v1.splitlines(keepends=True),
        content_v2.splitlines(keepends=True),
        ...
    )
)
diff_text = "".join(diff_lines)
```

**Observation:** For large documentation sections, `diff_text` could be many kilobytes. MCP responses are framed over stdio as JSON; very large `diff` values will work but add latency and may overwhelm IDE clients with weak JSON parsers. Consider capping `diff_text` to a maximum length (e.g., 4096 chars) and appending a truncation notice.

---

## Day 7: Benchmark v1

### `eval/metrics.py`

---

#### D7-1 🔴 `metrics.py` — `calculate_citation_recall` is a stub that always returns `1.0`

**File:** `eval/metrics.py` · Lines 129–149

```python
def calculate_citation_recall(
    draft_answer: str,
    citations: List[Any],
) -> float:
    """
    Computes citation recall: fraction of factual answer assertions backed by valid citations.
    """
    if not draft_answer or not draft_answer.strip():
        return 0.0
    if not citations:
        return 0.0

    markers = re.findall(r"\[\^([^\]]+)\]", draft_answer)
    if not markers:
        return 0.0

    # Answer has valid citations backing its assertions
    return 1.0
```

**Problem:** This function claims to compute *"fraction of factual answer assertions backed by valid citations"* but actually returns a binary `0.0` or `1.0` — specifically `1.0` as long as any citation marker exists. It does not count the number of assertions, nor the fraction that are cited. The benchmark summary metric `citation_recall_pct` is therefore meaningless as reported — it will approach 100% for any non-trivial answer, providing zero signal about citation density.

**Scenario:** The benchmark reports `citation_recall_pct: 98.0%` even though many generated answers have only one citation marker for three factual claims. The metric is used in a go/no-go gate for model release. The team approves a model that is actually poorly grounded.

**Better approach:**
```python
def calculate_citation_recall(
    draft_answer: str,
    citations: List[Any],
) -> float:
    if not draft_answer or not draft_answer.strip():
        return 0.0
    if not citations:
        return 0.0

    # Split into sentences; count how many have at least one citation marker
    sentences = re.split(r"(?<=[.!?])\s+", draft_answer.strip())
    if not sentences:
        return 0.0

    cited_count = sum(
        1 for s in sentences
        if re.search(r"\[\^[^\]]+\]", s)
    )
    return cited_count / len(sentences)
```

---

#### D7-2 🟠 `metrics.py` — `calculate_recall_at_k` matcher fallback creates a recall asymmetry

**File:** `eval/metrics.py` · Lines 69–85

```python
if gt_chunks:
    retrieved_ids = {
        c.get("chunk_id") or c.get("id")
        for c in top_k
        if (c.get("chunk_id") or c.get("id"))
    }
    matched_gt = sum(1 for gid in gt_chunks if gid in retrieved_ids)
    if matched_gt > 0:
        return matched_gt / len(gt_chunks)

# Fallback to composite matcher
if gt_matcher:
    for chunk in top_k:
        if matches_ground_truth_passage(chunk, gt_chunks, gt_matcher):
            return 1.0

return 0.0
```

**Problem:** When `gt_chunks` is provided and `matched_gt == 0`, the function falls through to the `gt_matcher` block. The `gt_chunks` path returns a true recall fraction (`matched_gt / len(gt_chunks)`); the `gt_matcher` path always returns `1.0` on any structural match. This asymmetry means that the matcher path over-reports recall as perfect (100%) when it matches, regardless of how many ground-truth chunks were actually retrieved.

**Better approach:** Document the semantics explicitly with inline comments. If a true recall fraction is desired for the matcher path, count the number of matcher-matching chunks out of `len(top_k)`.

---

#### D7-3 🟡 `metrics.py` — `NUMBER_WORD_MAP` is redundant with `verifier.NUMBER_WORDS` and covers only 0–5

**File:** `eval/metrics.py` · Lines 152–165

```python
NUMBER_WORD_MAP = {
    "0": "zero", "zero": "0",
    "1": "one",  "one": "1",
    ...
    "5": "five", "five": "5",
}
```

**Problem:** `verifier.py` already defines `NUMBER_WORDS` (Lines 217–220) mapping word→digit covering 0–10. `metrics.py` defines `NUMBER_WORD_MAP` locally covering only 0–5. `check_value_accuracy` imports `get_equivalent_quantities` from `verifier.py` for unit normalization but re-implements number-word handling locally. The coverage gap means `check_value_accuracy` will not recognize *"six seconds"* as equivalent to `"6s"`.

**Better approach:** Import and reuse the `NUMBER_WORDS` dict from `verifier.py` in `metrics.py`, or move both to a shared `constants.py` module.

---

### `eval/run_benchmark.py`

---

#### D7-4 🔴 `run_benchmark.py` — Direct mutation of the `Settings` singleton is not thread-safe

**File:** `eval/run_benchmark.py` · Lines 94–96, 307

```python
settings = get_settings()
original_gemini_key = settings.GEMINI_API_KEY
if not gemini_enabled:
    settings.GEMINI_API_KEY = ""
...
# Restore settings
settings.GEMINI_API_KEY = original_gemini_key
```

**Problem:** `get_settings()` returns a shared singleton. Mutating `GEMINI_API_KEY` on the shared singleton directly is not thread-safe. If the benchmark is called from a multi-threaded context (e.g., a FastAPI endpoint, or `pytest-xdist` parallel workers), the mutation will bleed into concurrent pipeline calls, disabling Gemini for unrelated requests.

**Scenario:** A CI pipeline runs benchmark tests in parallel using `pytest-xdist`. Thread A sets `GEMINI_API_KEY = ""` while Thread B's `run_pipeline` call reads the same singleton and fires an LLM call with an empty key, causing an authentication failure that cascades into a wrong abstention on a question that should be answered.

**Better approach:** Pass `gemini_enabled` as a parameter through `run_pipeline` without mutating global settings, or use a context-local settings override pattern:
```python
# In run_pipeline, accept gemini_enabled as a parameter
# and pass it to the answer_writer_node at construction time
```

---

#### D7-5 🟠 `run_benchmark.py` — `q_draft_val_acc` referenced before assignment when `should_abstain=True`

**File:** `eval/run_benchmark.py` · Lines 236–301

```python
if should_abstain:
    if is_abstained:
        correct_refusals.append(qid)
    else:
        error_class = "WRONG_ANSWER"
        wrong_answers.append(qid)
else:
    ...
    q_draft_val_acc = check_value_accuracy(draft_answer, expected_value)
    ...

question_results.append({
    ...
    "draft_value_accuracy": q_draft_val_acc,   # ← NameError if should_abstain=True
    ...
})
```

**Problem:** `q_draft_val_acc` is only assigned in the `else` branch (`not should_abstain`). When `should_abstain=True`, the `question_results.append` block at line ~281 references `q_draft_val_acc` unconditionally, raising `NameError: name 'q_draft_val_acc' is not defined`. This crashes the entire benchmark run at the first abstention test case, discarding all results.

**Better approach:** Initialize alongside the other `q_*` variables at line ~230:
```python
q_draft_val_acc: Optional[bool] = None   # add this alongside q_recall, q_mrr, etc.
```

---

#### D7-6 🟠 `run_benchmark.py` — Global latency percentiles mix answered and abstained question populations without disclosure

**File:** `eval/run_benchmark.py` · Lines 136, 208, 341

```python
measured_latencies_all: List[float] = []
...
measured_latencies_all.extend(timed_latencies)   # includes abstention runs
...
lat_percentiles = calculate_percentiles(measured_latencies_all, [50, 95])
```

**Problem:** `measured_latencies_all` is used to compute the global `latency_median_ms` and `latency_p95_ms` in the final summary table, mixing fast abstention runs (typically <100ms) with full pipeline runs (typically 2–5s). The printed table at line 406 shows the blended median as the main latency metric, while IR and quality rows cover only answerable questions. This creates a comparison mismatch that understates true answer-path latency.

**Better approach:** Report latency separately for answered vs. abstained, or add a note in the printed table footer clarifying which rows cover all questions vs. answerable-only.

---

#### D7-7 🟡 `run_benchmark.py` — Relative `file_path` resolves from CWD, not repo root

**File:** `eval/run_benchmark.py` · Lines 101–104

```python
elif split == "test":
    file_path = Path("data/benchmarks/benchmark_test_frozen_20.json")
else:
    file_path = Path("data/benchmarks/benchmark_dev_40.json")
```

**Problem:** These are relative paths. `Path("data/benchmarks/...")` resolves relative to the **current working directory** at process start, not the repository root. Running the benchmark from a different directory (e.g., `python -m evidenceflow.eval.run_benchmark` from the `src/` subdirectory) will fail with `FileNotFoundError`.

**Better approach:**
```python
_REPO_ROOT = Path(__file__).resolve().parents[4]  # adjust depth to repo root
...
file_path = _REPO_ROOT / "data/benchmarks/benchmark_test_frozen_20.json"
```

---

#### D7-8 🔵 `run_benchmark.py` — `--tier` argument is accepted but completely unused

**File:** `eval/run_benchmark.py` · Lines 417, 425–433

```python
parser.add_argument("--tier", type=int, default=1, help="Evaluation tier")
...
# args.tier is never referenced
results = run_benchmark(
    split=args.split,
    ...
)
```

**Problem:** `--tier` is documented in the CLI help as `"Evaluation tier"` but `args.tier` is never read or passed to `run_benchmark`. Users who try `--tier 2` expecting different behaviour will get identical results to `--tier 1` with no warning. This is dead interface.

**Better approach:** Either implement tiered evaluation logic or remove the argument.

---

## `graph/audit_wrapper.py` (Context File)

---

#### DX-1 🟡 `audit_wrapper.py` — All audit DB write failures are silently swallowed with no cumulative tracking

**File:** `graph/audit_wrapper.py` · Lines 129–130, 161–162

```python
except Exception as db_err:
    logger.error("Failed to record step to step_ledger: %s", db_err)

# ... and in claim loop:
except Exception as db_err:
    logger.error("Failed to record claim to claim_audit: %s", db_err)
```

**Problem:** All PostgreSQL write failures during audit recording are caught and logged but the pipeline continues. If the database is unreachable for an extended period, every single step and claim insert fails silently, the audit ledger is left with gaps, and no alert is raised. Post-incident forensics would be impossible for that window. There is no mechanism to track the cumulative audit failure rate.

**Better approach:** Increment a process-level counter on audit failures and expose it as a Prometheus metric or at minimum periodically emit a summary warning:
```python
_AUDIT_FAIL_COUNTER = 0
...
except Exception as db_err:
    global _AUDIT_FAIL_COUNTER
    _AUDIT_FAIL_COUNTER += 1
    logger.error(
        "Audit write failure #%d: Failed to record step to step_ledger: %s",
        _AUDIT_FAIL_COUNTER, db_err
    )
```

---

## Summary Table

| ID | Severity | File | Issue |
|:---|:---:|:---|:---|
| D4-1 | 🟡 MEDIUM | `input_guardrail.py` | Version regex over-matches single-digit numerals in prose |
| D4-2 | 🔵 LOW | `input_guardrail.py` | Duplicate casual-pattern compilation across two functions |
| D4-3 | 🟡 MEDIUM | `abstention.py` | `l1_passed: True` on abstention is semantically misleading |
| D4-4 | 🟡 MEDIUM | `verifier.py` | Multi-chunk value verification exits on first failure even when another chunk supports the value |
| D4-5 | 🔵 LOW | `verifier.py` | `split_sentences_safely` lookbehind references `\bp` inconsistently |
| D4-6 | 💡 OBS | `verifier.py` | `get_equivalent_quantities` `elif` memory branch chained to time ladder |
| D5-1 | 🔴 CRITICAL | `ledger.py` | `get_next_step_number` is not atomic — race condition under concurrent requests |
| D5-2 | 🔴 CRITICAL | `ledger.py` | Silent credential fall-back elevates privileges to DDL owner role in production |
| D5-3 | 🟠 HIGH | `ledger.py` | `record_turn` return value misleading when `ON CONFLICT DO NOTHING` fires |
| D5-4 | 🟠 HIGH | `ledger.py` | `sanitize_payload` token-protection heuristic passes through `access_token_expiry_seconds` |
| D5-5 | 🟡 MEDIUM | `ledger.py` | `fetch_*` functions expose `SELECT *` without column allowlist |
| D5-6 | 🟠 HIGH | `migrate_postgres.py` | Migration runner has no idempotency tracking table — re-applies all migrations on each run |
| D5-7 | 🟡 MEDIUM | `migrate_postgres.py` | `drop_all_tables` logs potentially unbound `actual_connected_db` |
| D6-1 | 🟠 HIGH | `corpus_server.py` | `search_passages` performs redundant length validation after Pydantic already enforces `max_length` |
| D6-2 | 🟠 HIGH | `corpus_server.py` | `diff_sections` does not validate `v1 != v2`, enabling no-op diffs |
| D6-3 | 🟡 MEDIUM | `corpus_server.py` | `_find_parameter_chunk` swallows all Qdrant errors silently |
| D6-4 | 🔵 LOW | `corpus_server.py` | `list_versions` hardcodes version details instead of deriving from `VALID_VERSIONS` |
| D6-5 | 💡 OBS | `corpus_server.py` | `diff_sections` unified diff output is unbounded in size |
| D7-1 | 🔴 CRITICAL | `metrics.py` | `calculate_citation_recall` is a stub that always returns `1.0` |
| D7-2 | 🟠 HIGH | `metrics.py` | `calculate_recall_at_k` matcher fallback creates recall asymmetry (always 1.0 on match) |
| D7-3 | 🟡 MEDIUM | `metrics.py` | `NUMBER_WORD_MAP` is redundant with `verifier.NUMBER_WORDS` and covers only 0–5 |
| D7-4 | 🔴 CRITICAL | `run_benchmark.py` | Direct mutation of the `Settings` singleton is not thread-safe |
| D7-5 | 🟠 HIGH | `run_benchmark.py` | `q_draft_val_acc` referenced before assignment when `should_abstain=True` → `NameError` crash |
| D7-6 | 🟠 HIGH | `run_benchmark.py` | Global latency P95 mixes answered and abstained question populations without disclosure |
| D7-7 | 🟡 MEDIUM | `run_benchmark.py` | Relative `file_path` resolves from CWD, not repo root |
| D7-8 | 🔵 LOW | `run_benchmark.py` | `--tier` argument is accepted but completely unused |
| DX-1 | 🟡 MEDIUM | `audit_wrapper.py` | All audit DB write failures silently swallowed with no cumulative tracking |

**Findings by severity:** 🔴 4 Critical · 🟠 8 High · 🟡 7 Medium · 🔵 4 Low · 💡 2 Observations = **25 total findings**
