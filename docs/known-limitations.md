# EvidenceFlow AI — Known Limitations & Fragility Analysis

This document records the empirical failure modes, edge cases, and architectural weaknesses identified during Day 4 Guardrail and Retrieval benchmarking.

---

## 1. Adversarial False Premises: Fake Features Next to Real Settings

### Symptom & Empirical Trace
When a query pairs a real PostgreSQL parameter with a completely fabricated or unsupported feature, the system risks answering the query rather than refusing the false premise.
- **Example (`h19`)**: *"How do I set shared_buffers to enable quantum caching in Postgres 16?"*
- **Observed Behavior**:
  - `shared_buffers` is a core PostgreSQL setting with extensive documentation.
  - Dense cosine similarity exceeds `0.78`, and BM25 matches `shared_buffers` at rank #1.
  - The lexical grounding check passes because the primary distinctive parameter exists in the retrieved evidence.
  - In offline fallback mode (or if an LLM hallucinates), an answer is synthesized from the `shared_buffers` chunk, citing `pg16-runtime-config-resource-shared-buffers`.
  - Only an ungrounded LLM declaration of "insufficient evidence" prevented a live hallucination; the deterministic rule layer alone did **not** stop the question.

### Root Cause
The retrieval grounding guardrail enforces an **existential** check rather than a **universal** check: it verifies that *at least one* recognized technical parameter exists in the corpus. Unrecognized modifiers (`quantum`, `turbo`, `hyperscale`) that are not formatted as snake_case parameters slip through when attached to legitimate setting names.

### Architectural Mitigation Plan
- **Corpus Vocabulary Verification**: Extract all content tokens from the user query. If a content token (excluding common English stopwords) does not exist anywhere in the documentation vocabulary for that PostgreSQL version, trigger a deterministic abstention:
  ```json
  {
    "action": "abstained_unrecognized_term",
    "code": "MISSING_KEYWORD",
    "unrecognized_terms": ["quantum"]
  }
  ```
- **Partial Grounding Response**: Alternatively, partition the answer into two parts: answer the grounded parameter (`shared_buffers`), while explicitly warning that the modifier (`quantum caching`) does not exist in PostgreSQL documentation.

---

## 2. Thin Retrieval Margin on Descriptive Paraphrases

### Symptom & Empirical Trace
When users ask conceptual or descriptive questions without mentioning the exact parameter name, retrieval operates with very thin safety margins.
- **Example (`h10`)**: *"Which setting controls how much memory a sort can use before it writes to temporary files in Postgres 16, and what is its default?"*
- **Observed Behavior**:
  - **Dense-Only Search**: Semantic drift caused broad disk and memory neighbors (`temp_file_limit`, `max_stack_depth`, `maintenance_work_mem`) to crowd the top ranks. The target setting (`work_mem`) was pushed to rank **#17** (cosine score `0.7486`), completely missing `RETRIEVAL_LIMIT=5`.
  - **Hybrid Dense + BM25 Search**: BM25 ranked `work_mem` at **#1** due to exact lexical matches on *"memory"*, *"sort"*, and *"temporary files"*. Fused via Reciprocal Rank Fusion (RRF), `work_mem` reached rank **#4**.
  - **The Fragility**: Rank #4 of 5 leaves a safety margin of only **1 rank position**. A minor increase in corpus size or slightly altered phrasing could drop `work_mem` to rank #6, causing an ungrounded abstention.

### Root Cause
- Dense embeddings on technical configuration docs cluster closely: cosine scores for related memory parameters fall within a narrow band of `0.74 - 0.79`.
- Pure semantic distance struggles to distinguish between "temporary file limit on disk" and "memory allowed before spilling to temporary files".

### Architectural Mitigation Plan
- **Query Classification & Adaptive Retrieval Limits**:
  - If a query contains an explicit parameter name (`max_connections`), keep `RETRIEVAL_LIMIT = 5`.
  - If a query is descriptive/paraphrased (no exact parameter name), expand `RETRIEVAL_LIMIT = 8` or `10` before passing candidates to the reranker and answer writer.
- **Dense Cosine Threshold Stability**: Ensure the `0.60` threshold continues to evaluate raw dense cosine similarity, but calibrate a separate RRF rank-presence threshold to avoid reliance on arbitrary cosine cutoffs when BM25 provides strong lexical anchors.

---

## 3. Release Notes vs. Runtime Configuration Cross-Document Bleed

### Symptom & Empirical Trace
When a parameter is removed in a newer PostgreSQL version, querying about it can surface release notes rather than triggering a version abstention.
- **Example (`h14`)**: *"What is vacuum_defer_cleanup_age in Postgres 16 and what is its default?"*
- **Observed Behavior**:
  - The parameter `vacuum_defer_cleanup_age` existed in PostgreSQL 14 and 15, but was removed in PostgreSQL 16.
  - The benchmark ground truth expected an abstention (`"Setting was removed in 16. Verify on real docs."`).
  - However, the PostgreSQL 16 index includes both `runtime-config-*.html` and `release-16.html`.
  - Qdrant chunk `pg16-release-16-96b6fa3f68` explicitly states: *"Remove server parameter vacuum_defer_cleanup_age"*.
  - The retriever surfaced this release note chunk, and the model answered factually that PostgreSQL 16 removed the parameter.
  - Under strict test harness labels, this was marked as a "wrong answer" (answer generated instead of abstaining), despite the answer being 100% grounded in official PostgreSQL 16 documentation.

### Root Cause
Single-namespace ingestion: `runtime-config` (current configuration state) and `release-notes` (historical change logs) are stored in the same Qdrant collection under `version="16"`. The retriever treats change-log documentation as equivalent to active configuration documentation.

### Architectural Mitigation Plan
- **Document Type Tagging**:
  Add a `doc_type` payload metadata tag during scraping/chunking:
  - `doc_type: "configuration"` for `runtime-config-*.html`
  - `doc_type: "release_notes"` for `release-*.html`
- **Scoped Query Routing**:
  - Configuration queries ("What is the default of X?", "What is the minimum value?") filter by `must: [{version: target_version}, {doc_type: "configuration"}]`.
  - When the setting is missing in configuration docs, the cross-version check inspects prior versions (`VERSION_MISSING`).
  - Deprecation and release queries ("What changed in version 16?", "Was X removed?") explicitly search `doc_type: "release_notes"`.

---

## 4. Fallback Mode Section Number Extraction as Value (`g11`)

### Symptom & Empirical Trace
In zero-LLM deterministic fallback mode, the verifier may treat structural documentation metadata (such as chapter or section numbers) as a claimed configuration value, causing a false-positive rejection.
- **Example (`g11`)**: *"What is autovacuum_naptime in Postgres 16?"*
- **Observed Behavior**:
  - When the Gemini API quota is exhausted, the offline fallback synthesizes:
    *"According to PostgreSQL 16 documentation on Section 20.4: Specifies the minimum delay between autovacuum runs on any given database..."*
  - The verifier detects the attribute word `"minimum"`.
  - The numeric value extractor captures the numeral `"20"` from the section prefix (`Section 20.4`) and attempts to bind it to the parameter's minimum allowed value.
  - Verification fails closed with code `VALUE_MISMATCH`:
    `L1.5/L2 Value Verification Failed: Value '20' appears in passage, but does not match claimed attribute 'minimum'. in source passage 'pg16-runtime-config-autovacuum-autovacuum-naptime'.`

### Root Cause
The fallback text template prefixes the answer with the documentation section name/number, and the regex value extraction passes all integers in the sentence into attribute binding without filtering out section/chapter metadata numerals.

### Architectural Mitigation Plan
- Strip `Section X.Y` or `Chapter X` prefixes before extracting candidate values in `verifier.py`.
- Ensure deterministic fallback templates omit numeric section identifiers from synthesized sentences or format them cleanly outside the factual claim boundary.

---

## 5. Multi-Turn Follow-Up Reasoning & Entity Resolution (Deferred to Tier 2)

### Symptom & Current Scope Boundary
When a user asks a follow-up question that relies on implicit context from an earlier turn (e.g. Turn 1: *"What is the default of shared_buffers in PostgreSQL 16?"*, Turn 2: *"What about version 15?"*), the Day 5 pipeline recovers previous thread state from the checkpointer, but does **not** perform conversational query rewriting or cross-turn entity resolution.
- **Observed Behavior**:
  - The PostgreSQL checkpointer safely persists and recovers graph state across server restarts for a given `thread_id`.
  - However, the follow-up query *"What about version 15?"* contains no explicit parameter name. Without a conversational entity resolution layer, the query retriever searches literally for generic tokens, missing the intended target (`shared_buffers`).
- **Scope Justification**:
  - Day 5 specifically implements thread isolation, process-restart state recovery, and immutable audit logging in PostgreSQL.
  - Conversational query rewriting, entity resolution across turns, and semantic turn graph memory (`:Turn`, `:Entity`, `:Claim`) are scheduled as core components of **Tier 2 (Neo4j Conversational Graph Memory)**.

---

## Summary Matrix

| Limitation | Severity | Trigger Condition | Current Safeguard | Planned Architectural Fix |
| :--- | :--- | :--- | :--- | :--- |
| **Fake Features (`h19`)** | High | Real setting paired with non-existent technical modifier | LLM abstention declaration | Corpus-wide vocabulary check; reject non-existent content tokens |
| **Thin Retrieval Margin (`h10`)** | Medium | Paraphrased question without setting name | Dense + BM25 RRF fusion (rank #4) | Adaptive `RETRIEVAL_LIMIT` (expand to 8-10 for descriptive queries) |
| **Release Note Bleed (`h14`)** | Low / Label Discrepancy | Inquiring about a removed or deprecated setting | Model cites release note | `doc_type` metadata partitioning (`configuration` vs `release_notes`) |
| **Section Num as Value (`g11`)** | Medium / Fallback Only | Deterministic fallback synthesizes sentence containing section number | Fail-closed verifier | Strip section/chapter numbers from candidate claim value extraction |
| **Multi-Turn Entity Resolution** | Deferred / Tier 2 | Follow-up questions with implicit entities (*"what about v15?"*) | Checkpointer recovers state; query requires explicit parameter | Tier 2 Neo4j conversational graph memory & query rewriting |
