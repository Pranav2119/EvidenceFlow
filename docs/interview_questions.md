# EvidenceFlow AI: Technical Interview & Explain-Back Questions

This document compiles all technical explain-back and architectural interview questions across every stage and day of the EvidenceFlow AI project. Use these questions to test your conceptual clarity and interview readiness. Answers are intentionally omitted so you can practice explaining each concept in your own words.

---

## Tier 1: Measure First (Baseline Pipeline & Corpus Ingestion)

### Day 0: Environment, Tooling, & Container Orchestration
1. Why do we run Qdrant, PostgreSQL, and Neo4j simultaneously instead of just using PostgreSQL with `pgvector` and recursive relational SQL joins?
2. What happens if a developer commits their `.env` file to GitHub, and what is the exact blast radius if Gemini or OpenAI keys are leaked?
3. How does Docker port mapping `host:container` (e.g. `6333:6333`) route TCP packets to isolated container namespaces?

### Day 1: Documentation Parsing, Hierarchy-Aware Chunking, & Hashes
1. What is "version bleed" in RAG systems, and why can semantic search alone not resolve it?
2. Why is SHA-256 preferred over simple MD5 or integer IDs for document chunk hashing?
3. Why does splitting by markdown/HTML DOM headers preserve more semantic context than fixed 512-character token slicing?

### Day 2: Qdrant Vector Store Setup & Version-Scoped Retrieval
1. Why does hybrid retrieval (BM25 + Dense) beat pure dense vector search for database configuration documentation? *(Defense case study: Recall the wal_level trace where dense search ranked the exact chunk 5th out of 5 behind broad WAL neighbors, risking recall failure if top-k < 5).*
2. How does Qdrant's filterable HNSW (where the filter is applied during graph traversal) differ from standard pre-filtering or post-filtering, and why does post-filtering cause "recall collapse"?
3. What is the difference between Cosine Distance, Dot Product, and Euclidean Distance when comparing normalized embeddings?

### Day 3: Baseline LangGraph State Machine
1. Why is Pydantic v2 `BaseModel` superior to plain Python dictionaries or `TypedDict` for LangGraph state passing?
2. What is the responsibility of the `L1Verifier` node, and why must it run before the answer is delivered to the user?
3. What happens to state fields if a node does not return them during a StateGraph transition?

### Day 4: Abstention Mechanism & Input/Output Guardrails
1. Why must the abstention logic be deterministic code rather than asking an LLM: *"Did you hallucinate?"*?
2. What are the three conditions that must trigger an immediate abstention in EvidenceFlow AI?
3. What is the architectural difference between input guardrails (pre-execution) and output guardrails (post-generation)?

### Day 5: PostgreSQL Checkpointer, Step Ledger, & Audit Trail
1. What is the difference between LangGraph's short-term thread state checkpointer and our application-level step ledger table?
2. Why is an immutable claim-evidence audit trail essential for enterprise AI compliance?
3. How does PostgreSQL write-ahead logging (WAL) prevent state corruption during agent execution crashes?

### Day 6: FastMCP Corpus Server
1. What is the Model Context Protocol (MCP), and why build custom FastMCP servers instead of hardcoding database queries directly into agent nodes?
2. What is the difference between an MCP Tool, an MCP Resource, and an MCP Prompt?
3. How does FastMCP enforce role-based permission boundaries (e.g., read-only tools vs write tools)?

### Day 7: Benchmark v1 & Deterministic Evaluation
1. What is the difference between Retrieval Metrics (Recall@k, MRR) and Generation Metrics (Groundedness, Citation Precision)?
2. Why is using release-note diffs for benchmark ground truth superior to LLM-generated synthetic QA datasets?
3. How do you measure citation precision vs citation recall on a generated technical response?

---

## Tier 2: The Differentiator (Memory & 5-Layer Scoped Cache)

### Day 8: Neo4j Conversational Graph Memory
1. What is "index-free adjacency" in graph databases, and why does Neo4j traverse multi-turn conversation relationships in $O(k)$ time compared to relational SQL `JOIN`s?
2. Why does flattening multi-turn conversation history into a sliding-window text prompt fail on complex technical follow-up questions?
3. What nodes and relationships form the conversational memory schema in EvidenceFlow AI?

### Day 9: 5-Layer Scoped Cache & Composite HMAC Keys
1. Walk through the 5 cache layers in EvidenceFlow AI (exact-match, semantic, retrieval, embedding, verified-claim). When is each layer checked?
2. Why must a cache scope key include the documentation version, user intent, and guardrail config hash?
3. How does HMAC-SHA256 prevent cross-version cache collisions when two queries use the identical prompt string?

### Day 10: Zero-LLM Cache Hit Validator & Document Patch Staleness
1. What is the Zero-LLM Cache Gatekeeper, and how does it validate a cache hit in under 3ms with 0 tokens consumed?
2. How does the validator detect if a documentation chunk has been updated since the cached answer was generated?
3. What is the false-hit rate, and why is reporting false-hit rate mandatory when defending a RAG cache in an interview?

### Day 11: Memory-Cache FastMCP Server
1. Why isolate cache and memory write permissions exclusively to the Cache Gatekeeper and Memory Curator agents?
2. How does the server prevent unauthorized agent nodes from injecting poisoned claims into the verified cache?

### Day 12: Adversarial Collision & Staleness Tests
1. What is a "cache collision trap test", and how do you write an automated test that attempts to trick the cache with conflicting versions?
2. What test proves that modifying a single SHA-256 chunk hash causes an immediate cache miss on the next query?

### Day 13: Tier 2 Ablation Study & Non-Inferiority Testing
1. What is non-inferiority testing in AI system evaluation, and what does the $\Delta$ margin represent?
2. Why must an ablation study test Baseline vs +Memory vs +Cache vs +Both across identical test splits?
3. If caching reduces latency by 70% and tokens by 80%, how do you prove statistically that answer quality did not degrade?

---

## Tier 3: Stand Out (Cross-Version Reasoning & Production Hardening)

### Day 14: Contradiction Analyst Agent
1. How does the Contradiction Analyst detect when a default setting in PostgreSQL 16 directly contradicts or supersedes a default in PostgreSQL 14?
2. How are `SUPERSEDES` and `CONTRADICTS` graph edges structured in Neo4j?

### Day 15: Independent L2 Judge for Contested Claims
1. Why must the L2 Judge belong to a different model family than the primary generator model?
2. Why should the L2 Judge only run on 5–10% of contested claims rather than 100% of queries?

### Day 16: Comparative Benchmark (EvidenceFlow vs pgvector vs Recursive SQL)
1. What causes `pgvector` HNSW index-scan degradation under concurrent write workloads compared to Qdrant?
2. Why do recursive SQL Common Table Expressions (CTEs) suffer from combinatorial blowup on multi-hop entity graphs compared to Neo4j Cypher?

### Day 17: External API FastMCP Server
1. How do you design an external API MCP server with rate-limiting, exponential backoff retries, and strict JSON Schema validation?
2. How does the agent handle external API downtime without crashing the user session?

### Day 18: CI Regression Test Gate
1. What is a CI regression test gate, and what specific metrics (Recall@k, Groundedness) must block a pull request if degraded?
2. How do you prevent benchmark test set contamination in continuous integration?

---

## Top 10 Portfolio Interview Defense Questions

1. *"Why didn't you just use PostgreSQL with pgvector and full-text search for everything?"*
2. *"How do you prove your cache doesn't return stale or hallucinated answers when documentation updates?"*
3. *"Why use Neo4j for memory instead of storing chat history in PostgreSQL or Redis?"*
4. *"What is version bleed, and how does your architecture mathematically prevent it?"*
5. *"Why use LangGraph instead of standard LangChain chains or AutoGen/CrewAI?"*
6. *"How do you evaluate RAG components independently instead of just scoring the end-to-end output?"*
7. *"What is non-inferiority testing, and what was your $\Delta$ margin?"*
8. *"Why FastMCP? What does MCP give you that regular Python tool functions don't?"*
9. *"How does your system decide when to abstain instead of answering?"*
10. *"What is the exact latency and token reduction achieved by your 5-layer cache and graph memory?"*
