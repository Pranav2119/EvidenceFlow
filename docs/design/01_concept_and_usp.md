# Stage 1: Concept, USP, and Architectural Justifications

## 1. Metric & Terminology Glossary (Plain English)
- **Groundedness**: Every assertion made in an answer is directly supported by retrieved source text, with zero unverified assumptions or outside fabrications.
- **Recall@k**: The fraction of all relevant source passages that appear anywhere in the top $k$ retrieved results.
- **MRR (Mean Reciprocal Rank)**: A measure of retrieval quality that scores how high up the first truly relevant chunk was placed (rank 1 = 1.0, rank 2 = 0.5, rank 3 = 0.33).
- **Non-inferiority**: A statistical test proving an optimization (like caching or context pruning) does not degrade output quality by more than a pre-agreed margin (e.g., 2%).
- **Confidence Interval**: A range of values derived from sample data that is likely to contain the true value with a specified probability (e.g., 95% certainty).

---

## 2. Core Problem: The Version-Bleed Failure Mode in Technical Docs
Most RAG systems fail when applied to real-world software engineering or legal corpora because documentation **evolves**:
- In **PostgreSQL 14**, a query configuration or default parameter might be set to $X$.
- In **PostgreSQL 16**, that default parameter was changed to $Y$, or deprecated altogether.
- Naive vector search embeds both passages into a high-dimensional vector space where semantic similarity is near 0.95. A user asking *"What is the default value of `vacuum_cost_limit` in PostgreSQL 16?"* will frequently retrieve the chunk from PostgreSQL 14 if it happens to have closer token phrasing.

**EvidenceFlow AI** treats technical documents as **versioned assertions**, where facts are strictly scoped to a release version and can explicitly *supersede* or *contradict* prior facts.

---

## 3. Unique Selling Proposition (USP) vs. Toy RAG
| Dimension | Toy RAG Systems | EvidenceFlow AI |
| :--- | :--- | :--- |
| **Verification** | Blind trust in model completion | Deterministic quote/containment verification + entailment check before delivery |
| **Abstention** | Hallucinates or guesses when evidence is missing | Standardized inspectable abstention with reason codes |
| **Caching** | Unsafe raw-string query caching (cross-version collisions) | 5-layer scoped cache with HMAC composite keys and zero-LLM deterministic hit validation |
| **Conversational Memory**| Unbounded context stuffing (high cost, prompt dilution) | Dual-tier memory: Postgres thread checkpoints + Neo4j relational entity-claim graph |
| **Evaluation** | Single metric (e.g. LLM-as-a-judge score) | Decoupled component-level metrics (Recall@k, Verifier precision, Cache false-hit rate, Non-inferiority) |

---

## 4. Architectural Defense

### "Why not just pgvector + Postgres for everything?"
1. **The Index Combining Problem**: In Postgres, combining `tsvector` (GIN) + `pgvector` (HNSW) + version filters (`WHERE version = '16'`) often forces full sequential table scans or poor query plans because the cost estimator struggles to combine disparate index structures.
2. **Qdrant's Filtered Single-Stage HNSW**: Qdrant applies payload filters *during* the graph traversal, avoiding post-filtering and preserving recall and latency at scale.
3. **Graph Traversal Complexity**: Querying multi-hop conversational relations (`Turn -> Question -> Entity <- Supersedes - Version`) via Recursive Common Table Expressions (CTEs) in SQL leads to unmaintainable, brittle queries that scale poorly. Neo4j Cypher handles these pattern matches natively.
4. **Operational Isolation**: Postgres guarantees ACID transactions for step ledgers and audit records without competing for CPU/RAM against vector similarity computations.

### "Why is this cache safe?"
1. **Composite Scope Key**: Keys incorporate `(Intent || Corpus_Version || Guardrail_Hash || Config_Hash)`. Queries targeting different versions never share cache slots.
2. **Deterministic Hit Validation**: Every hit verifies that cited chunk hashes still match the current database snapshot. If a doc is patched, the cache entry is immediately evicted.
3. **Upstream Guardrails**: Malicious prompts are rejected *before* reaching the cache layer; injected text is never stored in cache.
4. **Zero-LLM Hit Path**: Executed completely in deterministic Python code ($0 tokens, minimal latency).
