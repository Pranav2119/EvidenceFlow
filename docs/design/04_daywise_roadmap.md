# Stage 4: Day-Wise Learning Mode Roadmap

This document outlines the step-by-step, day-by-day implementation plan for **EvidenceFlow AI** across all three tiers. Every day follows the strict **Learning Mode** contract:
1. **Goal**: Concrete deliverable for the day.
2. **Concept to Understand First**: 5-minute foundational breakdown before writing any code.
3. **Exact Commands**: PowerShell commands with a 1-line explanation and its exact undo command.
4. **Ownership Split**: What you write yourself (core logic) vs. what Gemini may generate (boilerplate).
5. **Verification**: How to verify that today's work functions correctly.
6. **Break It on Purpose**: An intentional sabotage exercise to observe failure modes and error logs.
7. **Explain-Back Questions**: 3 technical questions to test interview readiness.
8. **Git Commit Message**: Conventional, clear commit message.
9. **Daily Shutdown Routine**: Graceful teardown of background processes.

---

## Metric & Terminology Glossary
- **Precision**: The fraction of retrieved items or generated claims that are actually relevant or correct (True Positives / (True Positives + False Positives)).
- **Recall**: The fraction of all existing relevant items or facts that were successfully captured (True Positives / (True Positives + False Negatives)).
- **False-Hit Rate (Cache)**: The percentage of cache hits where the cached content did *not* actually match the semantic intent or version scope of the query.
- **P-Value**: The probability that a measured improvement (e.g., lower latency) occurred purely by random chance rather than an actual architectural enhancement.

---

# Tier 1: Measure First (Baseline Ingestion & Core Flow)

## Day 0: Tooling, Repositories, Environment, & Container Infrastructure

### 1. Goal
Initialize the development environment with pinned tooling, configure `.env` secrets, establish container orchestration for PostgreSQL, Qdrant, and Neo4j, and make the repository's initial commit.

### 2. Concept to Understand First (5 minutes)
- **Container Isolation & Host Port Binding**: Why running database servers in Docker Compose (`5432:5432`, `6333:6333`, `7687:7687`) decouples application runtime dependencies from Windows host clutter, ensuring full reproducibility.
- **Environment Parity**: Why API keys and database credentials must never exist in code or commit history, but must be validated at application startup using Pydantic `BaseSettings`.

### 3. Exact Commands
- **Check uv installation**:
  ```powershell
  uv --version
  ```
  *Explanation*: Verifies that the pinned fast Python package manager `uv` is installed and reachable.  
  *Undo*: `winget uninstall astral-sh.uv`
- **Create local `.env` from template**:
  ```powershell
  Copy-Item .env.example .env
  ```
  *Explanation*: Creates local configuration file for private API keys and connection strings.  
  *Undo*: `Remove-Item .env`
- **Spin up local multi-database cluster**:
  ```powershell
  docker compose up -d
  ```
  *Explanation*: Starts PostgreSQL, Qdrant, and Neo4j in background daemon mode.  
  *Undo*: `docker compose down`
- **Check container health**:
  ```powershell
  docker compose ps
  ```
  *Explanation*: Confirms all three database containers are in the `Up` / healthy status.  
  *Undo*: N/A (read-only command)

### 4. Ownership Split
- **Gemini Generates**: `docker-compose.yml` (database container configurations, volume mounts, health checks), `.gitignore`, `.env.example`.
- **You Write**: `src/evidenceflow/config.py` (Pydantic `BaseSettings` mapping environment variables with type hints and default connection strings).

### 5. Verification
Run a verification script testing connection reachability to all 3 ports:
```powershell
uv run python -c "import socket; [socket.create_connection(('localhost', p), timeout=2) for p in (5432, 6333, 7687)]; print('All database ports active!')"
```

### 6. Break It on Purpose
Change the Neo4j port in `docker-compose.yml` from `7687:7687` to `7689:7687`, run `docker compose up -d`, and execute the python port-check command above.
- *What breaks*: The connection attempt throws a `ConnectionRefusedError: [WinError 10061]`.
- *Why*: The container is listening on host port 7689, but the client tried connecting to default port 7687. Revert port back to `7687:7687`.

### 7. Explain-Back Questions
1. Why do we run Qdrant, PostgreSQL, and Neo4j simultaneously instead of just using PostgreSQL with `pgvector` and recursive relational joins?
2. What happens if a developer commits their `.env` file to GitHub, and what is the exact blast radius if Gemini or OpenAI keys are leaked?
3. How does Docker port mapping `host:container` (e.g. `6333:6333`) route TCP packets to isolated container namespaces?

### 8. Git Commit Message
`chore(infra): initialize Day 0 docker compose, env validation, and tooling configs`

### 9. Daily Shutdown Routine
```powershell
docker compose stop
```

---

## Day 1: Versioned PostgreSQL Documentation Parser & Chunking Pipeline

### 1. Goal
Download and parse official PostgreSQL documentation (v14, v15, v16) into structured, metadata-rich markdown chunks tagged with document section, version tag, and SHA-256 content hashes.

### 2. Concept to Understand First (5 minutes)
- **Hierarchy-Aware Chunking**: Splitting raw HTML/Markdown documentation blindly by character count (e.g., 500 tokens) severs configuration options from their section headers. Chunking by markdown headers (`#`, `##`, `###`) ensures configuration parameters retain their structural context.
- **Deterministic Content Hashes**: Storing a SHA-256 hash for every chunk enables instant staleness detection and deduplication.

### 3. Exact Commands
- **Run documentation ingestion pipeline**:
  ```powershell
  uv run python -m evidenceflow.ingestion.download_docs --versions 14 15 16
  ```
  *Explanation*: Downloads official PostgreSQL HTML/manual documentation for specified versions.  
  *Undo*: `Remove-Item -Recurse -Force data/raw`
- **Run parser and chunker**:
  ```powershell
  uv run python -m evidenceflow.ingestion.chunker
  ```
  *Explanation*: Extracts sections, computes SHA-256 hashes, and outputs JSON lines chunks.  
  *Undo*: `Remove-Item -Recurse -Force data/processed`

### 4. Ownership Split
- **Gemini Generates**: Beautiful soup / HTML-to-markdown extraction scripts and directory scaffolding.
- **You Write**: `src/evidenceflow/ingestion/chunker.py` (Header-aware chunk splitting logic, version metadata tagging, and SHA-256 hash assignment).

### 5. Verification
```powershell
uv run python -c "import json; chunk = json.loads(open('data/processed/chunks.jsonl').readline()); assert 'sha256' in chunk and 'version' in chunk; print('Valid chunk:', chunk['title'], 'v' + chunk['version'])"
```

### 6. Break It on Purpose
Modify the chunker so it omits the `version` field from the dictionary output, then rerun the test. Observe the `AssertionError`. In RAG, untagged data causes version bleed because the retriever cannot filter out legacy configurations.

### 7. Explain-Back Questions
1. What is "version bleed" in RAG systems, and why can semantic search alone not resolve it?
2. Why is SHA-256 preferred over simple MD5 or integer IDs for document chunk hashing?
3. Why does splitting by markdown headers preserve more semantic context than fixed 512-character token slicing?

### 8. Git Commit Message
`feat(ingestion): implement versioned documentation parser with sha256 metadata hashing`

### 9. Daily Shutdown Routine
```powershell
docker compose stop
```

---

## Day 2: Qdrant Hybrid Vector Store Setup with Dense & BM25 Payloads

### 1. Goal
Initialize Qdrant collections with hybrid indexing (Dense Gemini vector embeddings + Sparse BM25 lexical vectors) and configure indexed payload fields for version-constrained single-stage filtering.

### 2. Concept to Understand First (5 minutes)
- **Hybrid Search (Dense + Sparse)**: Dense embeddings excel at finding conceptual synonyms ("increase throughput"), but fail on exact technical keywords ("`wal_compression = zstd`"). BM25 lexical search finds exact keyword tokens. Combining them via Reciprocal Rank Fusion (RRF) yields state-of-the-art technical retrieval.
- **Single-Stage Filtered Traversal**: In Qdrant, payload filters (`version == '16'`) are evaluated *during* HNSW graph traversal rather than after, eliminating recall collapse.

### 3. Exact Commands
- **Create Qdrant collection and payload indexes**:
  ```powershell
  uv run python -m evidenceflow.storage.init_qdrant
  ```
  *Explanation*: Creates `evidenceflow_chunks` collection with cosine distance and indexed fields.  
  *Undo*: `uv run python -c "from qdrant_client import QdrantClient; QdrantClient('http://localhost:6333').delete_collection('evidenceflow_chunks')"`
- **Index ingested chunks into Qdrant**:
  ```powershell
  uv run python -m evidenceflow.ingestion.index_qdrant --input data/processed/chunks.jsonl
  ```
  *Explanation*: Generates embeddings using Gemini API and batch-uploads chunks into Qdrant.  
  *Undo*: Re-run init command above to purge collection.

### 4. Ownership Split
- **Gemini Generates**: Batching utilities and Qdrant connection setup boilerplate.
- **You Write**: `src/evidenceflow/storage/qdrant_client.py` (Hybrid search query construction using Qdrant client, applying hard payload filters for documentation version).

### 5. Verification
Execute a test query filtered strictly to PostgreSQL 16:
```powershell
uv run python -m evidenceflow.storage.qdrant_client --query "vacuum_cost_limit" --version "16"
```
Verify that 100% of the returned chunks contain `version: "16"` in their payload.

### 6. Break It on Purpose
Temporarily remove the `payload_index` on the `version` field in `init_qdrant.py`, re-upload, and run a benchmark query with a filter. Observe the increased search latency in Qdrant logs when it has to do unindexed payload evaluations.

### 7. Explain-Back Questions
1. Why does hybrid retrieval (BM25 + Dense) beat pure dense vector search for database configuration documentation?
2. How does Qdrant's single-stage filtered HNSW search differ from standard pre-filtering or post-filtering?
3. What is the difference between Cosine Distance, Dot Product, and Euclidean Distance when comparing normalized embeddings?

### 8. Git Commit Message
`feat(storage): initialize qdrant collection with hybrid vectors and version payload indexes`

### 9. Daily Shutdown Routine
```powershell
docker compose stop
```

---

## Day 3: Baseline LangGraph State Machine

### 1. Goal
Wire and compile the baseline LangGraph state machine with four core nodes: `Planner` -> `Retriever` -> `AnswerWriter` -> `L1Verifier`, passing a strongly-typed Pydantic state.

### 2. Concept to Understand First (5 minutes)
- **LangGraph StateGraph**: A directed graph where state flows immutably from node to node. Each node is a pure function or agent step that takes `State`, performs work, and returns updated keys.
- **Fail-Fast Node Transitions**: Instead of monolithic chain prompts, separating planning from retrieval and verification ensures each sub-task is inspectable and testable in isolation.

### 3. Exact Commands
- **Run the baseline LangGraph execution test**:
  ```powershell
  uv run python -m evidenceflow.graph.workflow --query "What is wal_level in Postgres 16?" --version "16"
  ```
  *Explanation*: Runs an end-to-end question through Planner, Retriever, AnswerWriter, and L1Verifier.  
  *Undo*: N/A (stateless run)

### 4. Ownership Split
- **Gemini Generates**: Node logging decorators, CLI argument parsers, and prompt formatting templates.
- **You Write**: `src/evidenceflow/graph/workflow.py` and `src/evidenceflow/state.py` (The Pydantic state definition, node additions, edge connections, and conditional router).

### 5. Verification
Check that the returned dictionary contains:
- `plan`: A non-empty search query string.
- `retrieved_passages`: At least 1 passage with `version: '16'`.
- `answer`: An answer containing markdown citations `[^chunk_id]`.
- `l1_verification_passed`: Boolean `True`.

### 6. Break It on Purpose
Introduce a bug in `workflow.py` by misnaming an edge (`builder.add_edge("planner", "retriever_typo")`). Run the workflow and observe LangGraph's graph compilation validation error (`GraphValidationError: Edge targets nonexistent node`).

### 7. Explain-Back Questions
1. Why is Pydantic v2 `BaseModel` superior to Python dictionaries or `TypedDict` for LangGraph state passing?
2. What is the responsibility of the `L1Verifier` node, and why must it run before the answer is delivered to the user?
3. What happens to state fields if a node does not return them?

### 8. Git Commit Message
`feat(graph): assemble baseline 4-node langgraph workflow with pydantic v2 state schema`

### 9. Daily Shutdown Routine
```powershell
docker compose stop
```

---

## Day 4: Standardized Abstention Mechanism & Input/Output Guardrails v1

### 1. Goal
Implement deterministic guardrails: input scope checks (rejecting out-of-domain queries and prompt injections), retrieval threshold guardrails, and deterministic citation verifiers that trigger clean abstention when evidence is insufficient.

### 2. Concept to Understand First (5 minutes)
- **Controlled Abstention**: A high-integrity enterprise AI must state *"I cannot find sufficient evidence in the verified documentation for PostgreSQL 16 to answer your question"* rather than hallucinate plausible parameters.
- **Deterministic Citation Entailment**: Verifying that every citation ID in the answer exists in `retrieved_passages` and that quoted text is a verbatim substring of the source passage.

### 3. Exact Commands
- **Run the guardrail test suite**:
  ```powershell
  uv run pytest tests/test_guardrails.py -v
  ```
  *Explanation*: Runs automated tests asserting that prompt injections, irrelevant questions, and ungrounded claims trigger abstention.  
  *Undo*: N/A (test run)

### 4. Ownership Split
- **Gemini Generates**: Pytest fixtures with adversarial prompt injection strings.
- **You Write**: `src/evidenceflow/graph/nodes/verifier.py` and `src/evidenceflow/graph/nodes/abstention.py` (Plain-code substring citation check and structured abstention response builder).

### 5. Verification
Execute a query about an unanswerable question or a made-up parameter:
```powershell
uv run python -m evidenceflow.graph.workflow --query "How do I configure quantum hyperthreading in Postgres 16?" --version "16"
```
Verify that the output has `is_abstained: true` and the answer contains zero hallucinations.

### 6. Break It on Purpose
Disable the score threshold check in `retriever.py` (allow chunks with similarity score < 0.20 to pass). Run the quantum query again and observe how the LLM attempts to generate a hallucinated answer because low-quality noise was fed into its context window.

### 7. Explain-Back Questions
1. Why must the abstention logic be deterministic code rather than asking the LLM: *"Did you hallucinate?"*?
2. What are the three conditions that must trigger an immediate abstention in EvidenceFlow AI?
3. What is the difference between input guardrails (pre-execution) and output guardrails (post-generation)?

### 8. Git Commit Message
`feat(guardrails): implement deterministic citation verification and structured abstention node`

### 9. Daily Shutdown Routine
```powershell
docker compose stop
```

---

## Day 5: PostgreSQL Checkpointer, Step Ledger, & Audit Trail Tables

### 1. Goal
Configure PostgreSQL to persist LangGraph conversation state across turns, store an immutable step ledger recording every intermediate node execution, and record a claim-evidence audit trail.

### 2. Concept to Understand First (5 minutes)
- **ACID Step Ledger**: Why AI systems require relational transaction logs. If an answer claims a database configuration can cause data loss, an auditor must be able to inspect the exact query, the retrieved passage hashes, and the verifier verdict associated with that answer UUID.
- **LangGraph Checkpointing**: Writing graph thread state to PostgreSQL enables multi-turn conversations to survive server restarts.

### 3. Exact Commands
- **Run database migrations for state and audit schema**:
  ```powershell
  uv run python -m evidenceflow.storage.migrate_postgres
  ```
  *Explanation*: Creates `checkpoints`, `step_ledger`, and `claim_audit` tables in PostgreSQL.  
  *Undo*: `uv run python -c "from evidenceflow.storage.postgres import drop_all_tables; drop_all_tables()"`
- **Test persistent thread execution**:
  ```powershell
  uv run python -m evidenceflow.graph.workflow --thread-id "test-session-001" --query "What is shared_buffers default in v16?"
  ```
  *Explanation*: Runs workflow and writes checkpoints into PostgreSQL.  
  *Undo*: N/A

### 4. Ownership Split
- **Gemini Generates**: SQLAlchemy / asyncpg table schemas and connection pooling boilerplate.
- **You Write**: `src/evidenceflow/storage/ledger.py` (Methods recording node inputs, outputs, execution latency, token counts, and citation links into PostgreSQL).

### 5. Verification
Query the PostgreSQL audit table directly to confirm recorded execution steps:
```powershell
uv run python -c "from evidenceflow.storage.ledger import fetch_recent_steps; steps = fetch_recent_steps('test-session-001'); print(f'Recorded {len(steps)} steps in PostgreSQL ledger!')"
```

### 6. Break It on Purpose
Pass an un-serializable object (like an open file handle or raw lambda function) into the LangGraph state. Run the workflow and observe the checkpointer serialize error. This will prove why Pydantic v2 schemas are critical for checkpoint persistence.

### 7. Explain-Back Questions
1. Why can't we store conversation audit trails in Qdrant or Neo4j? What does PostgreSQL provide that they cannot?
2. What is the purpose of the `thread_id` in LangGraph checkpointing?
3. How does recording token counts and execution latency per node in a step ledger help identify performance bottlenecks?

### 8. Git Commit Message
`feat(storage): integrate postgresql checkpointer and claim-evidence audit ledger`

### 9. Daily Shutdown Routine
```powershell
docker compose stop
```

---

## Day 6: FastMCP Corpus Server (`corpus-mcp`)

### 1. Goal
Implement a standalone, read-only FastMCP server exposing tools to inspect, search, and diff documentation sections across versions.

### 2. Concept to Understand First (5 minutes)
- **Model Context Protocol (FastMCP)**: A standardized JSON-RPC protocol allowing AI clients (like Claude, Gemini, or IDE sidecars) to discover and invoke tools with strict schema validation.
- **Cross-Version Diffing**: The `diff_sections` tool allows agents to pull identical sections across two versions (e.g., PostgreSQL 14 vs 16) and view a unified diff of what parameters were modified or deprecated.

### 3. Exact Commands
- **Run the FastMCP corpus server**:
  ```powershell
  uv run python -m evidenceflow.mcp.corpus_server
  ```
  *Explanation*: Starts the FastMCP server listening via stdio or SSE transport.  
  *Undo*: Press `Ctrl+C` in the terminal to stop the process.
- **Run FastMCP tool unit tests**:
  ```powershell
  uv run pytest tests/test_corpus_mcp.py -v
  ```
  *Explanation*: Tests all registered MCP tools (`search_passages`, `get_passage`, `diff_sections`).  
  *Undo*: N/A

### 4. Ownership Split
- **Gemini Generates**: FastMCP server initialization and CLI launcher.
- **You Write**: `src/evidenceflow/mcp/corpus_server.py` (The tool functions `@mcp.tool()`: `search_passages`, `get_passage`, `list_versions`, and `diff_sections`).

### 5. Verification
Invoke `diff_sections` via Python script to inspect differences between Postgres 14 and 16 for `vacuum_cost_limit`:
```powershell
uv run python -c "from evidenceflow.mcp.corpus_server import diff_sections; print(diff_sections('vacuum_cost_limit', '14', '16'))"
```
Verify that the output displays lines indicating changed default values.

### 6. Break It on Purpose
Call `get_passage` with a malformed chunk ID that does not exist in the database. Ensure that rather than crashing with an unhandled 500 error, the FastMCP tool gracefully returns a clean JSON error response (`status: 'not_found'`).

### 7. Explain-Back Questions
1. What is the Model Context Protocol (MCP), and why is building tools with FastMCP more extensible than hardcoded Python functions?
2. Why is `corpus-mcp` strictly read-only, and what security risks emerge if an agent tool has write permissions to raw corpus docs?
3. How does the `diff_sections` tool help an agent discover cross-version parameter changes without flooding context windows with full documentation pages?

### 8. Git Commit Message
`feat(mcp): implement read-only FastMCP corpus server with cross-version diff tool`

### 9. Daily Shutdown Routine
```powershell
docker compose stop
```

---

## Day 7: Benchmark v1 & Baseline Measurement Recording

### 1. Goal
Execute the Tier 1 evaluation suite across a curated dataset of versioned technical questions, calculate baseline metrics (Recall@5, MRR, Groundedness, Latency), and commit the empirical baseline into `docs/experiments.md`.

### 2. Concept to Understand First (5 minutes)
- **Measure First Principle**: In AI engineering, you cannot claim an optimization (like a cache or memory graph) made your system better unless you have hard, reproducible baseline numbers measured before those optimizations were introduced.
- **MRR (Mean Reciprocal Rank)**: A retrieval metric where higher is better. If the ground-truth document is retrieved at rank 1, reciprocal rank is 1.0; at rank 2, 0.5; at rank 3, 0.33.

### 3. Exact Commands
- **Run the Tier 1 baseline benchmark**:
  ```powershell
  uv run python -m evidenceflow.eval.run_benchmark --tier 1 --output docs/experiments.md
  ```
  *Explanation*: Evaluates 30 versioned questions, computing groundedness, latency, and retrieval accuracy.  
  *Undo*: N/A (generates benchmark results)

### 4. Ownership Split
- **Gemini Generates**: Benchmark runner execution scaffolding and markdown table formatting utilities.
- **You Write**: `src/evidenceflow/eval/metrics.py` (Deterministic scoring formulas for Citation Recall, Precision, and Groundedness).

### 5. Verification
Open `docs/experiments.md` and verify that the Tier 1 Baseline table contains:
- Total evaluated queries
- Mean Recall@5 and MRR
- L1 Groundedness percentage
- Average latency per query (in milliseconds)
- Average token usage per query

### 6. Break It on Purpose
Artificially tamper with the scoring function in `metrics.py` to divide by zero when no citations exist. Run the benchmark to catch the edge case and add proper guard checking (`if total == 0: return 0.0`).

### 7. Explain-Back Questions
1. Why must the Tier 1 benchmark run on a dedicated *dev* split rather than the *test* split?
2. If our baseline Recall@5 is 0.85 and MRR is 0.72, what does that indicate about our retriever's ranking performance?
3. Why do we record latency and token counts alongside answer quality metrics?

### 8. Git Commit Message
`test(eval): execute tier 1 baseline benchmark and record empirical metrics in experiments.md`

### 9. Daily Shutdown Routine
```powershell
docker compose stop
```

---

# Tier 2: The Differentiator (Memory & 5-Layer Scoped Cache)

## Day 8: Neo4j Conversational Graph Memory

### 1. Goal
Model and implement the conversational memory graph in Neo4j with `:Turn`, `:Question`, `:Entity`, `:Claim`, and `:Version` nodes, enabling multi-turn follow-up resolution without re-ingesting conversation histories into prompt tokens.

### 2. Concept to Understand First (5 minutes)
- **Conversational Memory Graphs**: In technical dialogues, users ask follow-up questions like *"What about in version 16?"* or *"Does that parameter require a server restart?"*. A graph resolves *"that parameter"* by traversing `(:Turn)-[:MENTIONED]->(:Entity {name: 'wal_compression'})`.
- **Relational Memory vs. Flat Chat Logs**: Flat chat logs grow linearly in token size and confuse LLMs when past conversation turns discuss older versions. A graph maintains explicit semantic edges.

### 3. Exact Commands
- **Apply Neo4j graph schema constraints and indexes**:
  ```powershell
  uv run python -m evidenceflow.memory.init_neo4j
  ```
  *Explanation*: Creates uniqueness constraints on `:Entity(name)`, `:Version(tag)`, and indexes on `:Turn(session_id)`.  
  *Undo*: `uv run python -c "from evidenceflow.memory.graph_client import drop_graph; drop_graph()"`
- **Test multi-turn graph insertion**:
  ```powershell
  uv run python -m evidenceflow.memory.test_insert
  ```
  *Explanation*: Inserts a mock 2-turn conversation and verifies edge creation.  
  *Undo*: Clears test session nodes.

### 4. Ownership Split
- **Gemini Generates**: Neo4j driver connection boilerplate and Cypher query execution helpers.
- **You Write**: `src/evidenceflow/memory/graph_client.py` (Cypher queries creating `:Turn`, `:Question`, `:Entity` nodes and traversing `FOLLOWS_UP` and `SCOPED_TO` relationships).

### 5. Verification
Run Cypher query in python to verify follow-up resolution:
```powershell
uv run python -c "from evidenceflow.memory.graph_client import resolve_entity_for_turn; print('Resolved Entity:', resolve_entity_for_turn('test-session', 'What about in v16?'))"
```
Verify that it returns the entity from turn 1 (`wal_compression`).

### 6. Break It on Purpose
Drop the uniqueness constraint on `:Entity(name)` and insert the same entity twice with different cases (`WAL_Compression` vs `wal_compression`). Observe graph fragmentation when queries fail to link related turns to the same canonical entity.

### 7. Explain-Back Questions
1. Why is a graph database (Neo4j) superior to vector databases for resolving pronouns like "it", "that setting", and "those options"?
2. What are the key node types and edge types in EvidenceFlow AI's conversational memory schema?
3. How does graph-based entity tracking protect against cross-turn context contamination?

### 8. Git Commit Message
`feat(memory): implement neo4j conversational graph schema and entity resolution queries`

### 9. Daily Shutdown Routine
```powershell
docker compose stop
```

---

## Day 9: 5-Layer Scoped Cache Architecture

### 1. Goal
Implement the 5-layer cache (Exact Match, Semantic, Retrieval, Embedding, Verified-Claim) using composite HMAC-SHA256 scope keys combining Intent, Version Scope, Config Hash, and Normalized Query.

### 2. Concept to Understand First (5 minutes)
- **Composite Cache Keys**: Caching LLM outputs with naive query string keys is dangerous. If the corpus version changes or guardrail rules update, stale or unsafe cached answers will leak. A composite scope key binds the exact version scope, configuration hash, and intent.
- **Layered Hierarchy**: Fast exact match (L1, memory) -> Vector semantic cache (L2, Qdrant) -> Retrieved passage ID cache (L3) -> Embedding cache (L4) -> Verified claims (L5).

### 3. Exact Commands
- **Run cache key generator unit tests**:
  ```powershell
  uv run pytest tests/test_cache_keys.py -v
  ```
  *Explanation*: Asserts that identical questions with different version scopes generate different hash keys.  
  *Undo*: N/A

### 4. Ownership Split
- **Gemini Generates**: In-memory LRU cache wrappers and redis/memory data structures.
- **You Write**: `src/evidenceflow/cache/key_generator.py` and `src/evidenceflow/cache/manager.py` (Composite key hashing formula and 5-layer cache lookup/write orchestration).

### 5. Verification
Run the key generation test script:
```powershell
uv run python -c "from evidenceflow.cache.key_generator import generate_scope_key; k1 = generate_scope_key('query', '14'); k2 = generate_scope_key('query', '16'); assert k1 != k2; print('Key isolation verified!')"
```

### 6. Break It on Purpose
Remove `version_scope` from the hash concatenation in `key_generator.py`. Run the verification test and observe the assertion failure (`k1 == k2`). This demonstrates how naive caching causes cross-version cache corruption.

### 7. Explain-Back Questions
1. Why must the cache key incorporate the configuration and guardrail version hashes in addition to the user query?
2. What is the difference between Layer 2 (Semantic Cache) and Layer 5 (Verified-Claim Cache)?
3. What is the computational cost of an HMAC-SHA256 hash calculation compared to an LLM API call?

### 8. Git Commit Message
`feat(cache): implement 5-layer scoped cache manager with composite HMAC-SHA256 keys`

### 9. Daily Shutdown Routine
```powershell
docker compose stop
```

---

## Day 10: Zero-LLM Deterministic Cache Hit Validator & Stale Chunk Eviction

### 1. Goal
Build the `CacheGatekeeper` node that validates cache entries deterministically—checking chunk SHA-256 hashes against live database records—with ZERO LLM calls on cache hit, and evicting stale entries automatically.

### 2. Concept to Understand First (5 minutes)
- **Zero-LLM Cache Gate**: If a cache hit required an LLM call to verify its accuracy, caching would save no money or latency. The validation must run entirely in deterministic plain code (hash matching, TTL checks, and metadata verification) in sub-millisecond time.
- **Silent Invalidation**: When a document chunk is updated in the corpus, its SHA-256 hash changes. Any cache entry referencing the old hash is instantly evicted on lookup.

### 3. Exact Commands
- **Test the deterministic validator**:
  ```powershell
  uv run pytest tests/test_cache_validator.py -v
  ```
  *Explanation*: Tests instant hit validation and automatic invalidation on chunk hash mismatch.  
  *Undo*: N/A

### 4. Ownership Split
- **Gemini Generates**: Mock database chunk repositories for fast testing.
- **You Write**: `src/evidenceflow/cache/validator.py` and `src/evidenceflow/graph/nodes/cache_gatekeeper.py` (The deterministic hash verification logic and LangGraph gatekeeper node).

### 5. Verification
Run a workflow test with cache enabled twice:
```powershell
uv run python -m evidenceflow.graph.workflow --query "What is max_connections in v16?" --version "16"
uv run python -m evidenceflow.graph.workflow --query "What is max_connections in v16?" --version "16"
```
Verify that the second execution completes in < 5ms with `cache_hit: true` and 0 LLM tokens consumed.

### 6. Break It on Purpose
Simulate a documentation patch: artificially change a single character in the source chunk's stored SHA-256 hash in PostgreSQL. Run the query again and verify that the Cache Gatekeeper identifies the hash mismatch, rejects the cached entry, and falls back to a fresh retrieval.

### 7. Explain-Back Questions
1. How does EvidenceFlow AI guarantee that a cache hit never consumes LLM tokens?
2. If a technical documentation typo is corrected in the database, how does the system know the cached answer is stale without clearing the entire cache?
3. What is the difference between a cache miss and a cache rejection?

### 8. Git Commit Message
`feat(cache): implement zero-LLM deterministic cache validator and staleness eviction`

### 9. Daily Shutdown Routine
```powershell
docker compose stop
```

---

## Day 11: FastMCP Memory & Cache Server (`memory-cache-mcp`)

### 1. Goal
Build a dedicated FastMCP server providing isolated, role-based tools for memory and cache operations, enforcing that only authorized agent roles can write to memory or cache.

### 2. Concept to Understand First (5 minutes)
- **Role-Based Tool Permissions**: An AnswerWriter agent should never be allowed to write directly to conversational memory or the cache. Only the `MemoryCurator` may commit to graph memory, and only the `CacheGatekeeper` may commit validated answers to the cache.
- **Audit Logging in Tool Execution**: Every MCP tool execution must log caller identity, parameters, and timestamps to the PostgreSQL audit ledger.

### 3. Exact Commands
- **Start the memory-cache MCP server**:
  ```powershell
  uv run python -m evidenceflow.mcp.memory_cache_server
  ```
  *Explanation*: Starts the FastMCP memory and cache service.  
  *Undo*: Press `Ctrl+C`.
- **Run MCP security isolation tests**:
  ```powershell
  uv run pytest tests/test_memory_cache_mcp.py -v
  ```
  *Explanation*: Verifies role enforcement and permission rejections.  
  *Undo*: N/A

### 4. Ownership Split
- **Gemini Generates**: FastMCP authorization headers and middleware decorators.
- **You Write**: `src/evidenceflow/mcp/memory_cache_server.py` (Tool functions for `query_entity_graph`, `commit_turn_memory`, `lookup_cache`, and `write_cache` with role checks).

### 5. Verification
Attempt an unauthorized write to the cache from a simulated `Retriever` client:
```powershell
uv run python -c "from evidenceflow.mcp.memory_cache_server import write_cache; res = write_cache('key', {}, caller_role='retriever'); print('Result:', res)"
```
Verify that it returns `{"error": "PermissionDenied: retriever cannot write to cache"}`.

### 6. Break It on Purpose
Remove the role validation check in `write_cache`. Observe that unverified intermediate retrievals can corrupt the cache with unverified assertions. Restore the role guard.

### 7. Explain-Back Questions
1. Why should tool access be partitioned by agent role rather than giving every agent unrestricted access to all MCP tools?
2. How does using an MCP server for memory decouple the storage backend (Neo4j) from the agent orchestration framework (LangGraph)?
3. What happens if the `memory-cache-mcp` server crashes during a workflow turn? How does the agent handle the failure?

### 8. Git Commit Message
`feat(mcp): build memory-cache-mcp server with strict role-based write isolation`

### 9. Daily Shutdown Routine
```powershell
docker compose stop
```

---

## Day 12: Cache Collision Traps & Staleness Failure Tests

### 1. Goal
Construct and execute an automated adversarial failure-test suite specifically testing cache-collision traps, cross-version isolation, prompt injection containment, and patch invalidation.

### 2. Concept to Understand First (5 minutes)
- **Adversarial Failure Testing**: High-performing AI engineers don't just write "happy path" tests. They write tests specifically designed to exploit cache collisions (asking the exact same question for version 14 and version 16 in alternating succession) and measure the **False-Hit Rate**.
- **False-Hit Rate Target**: The false-hit rate must be strictly 0.0%.

### 3. Exact Commands
- **Run the adversarial cache test suite**:
  ```powershell
  uv run pytest tests/failure/test_cache_traps.py -v
  ```
  *Explanation*: Runs 25 alternating cross-version queries and verifies zero collisions occur.  
  *Undo*: N/A

### 4. Ownership Split
- **Gemini Generates**: Adversarial question generator pairing identical parameter queries across versions 14, 15, and 16.
- **You Write**: `tests/failure/test_cache_traps.py` (The test assertions verifying version payload matches, hash integrity, and zero cross-version leaks).

### 5. Verification
Run the collision detector script:
```powershell
uv run python -m evidenceflow.eval.verify_cache_isolation
```
Ensure output states: `Total Collisions: 0 | False Hit Rate: 0.00%`.

### 6. Break It on Purpose
Temporarily hardcode the `version_scope` parameter inside the cache key generator to a static string `"default"`. Run `test_cache_traps.py`. Watch the tests fail immediately with detailed assertions displaying Postgres 14 answers returned for Postgres 16 queries.

### 7. Explain-Back Questions
1. What is a cache-collision trap, and why are LLM systems with semantic caching particularly vulnerable to it?
2. Why is a non-zero False-Hit Rate unacceptable in technical documentation RAG?
3. What test guarantees that a prompt injection embedded inside an unverified document is never written to the verified-claim cache?

### 8. Git Commit Message
`test(cache): implement automated cache-collision trap suite and staleness failure tests`

### 9. Daily Shutdown Routine
```powershell
docker compose stop
```

---

## Day 13: Tier 2 Ablation Study & Non-Inferiority Analysis

### 1. Goal
Execute a multi-condition ablation study across 4 configurations: **Baseline**, **+Memory**, **+Cache**, and **+Both**, running each condition 3 times to measure cost/token reductions, latency gains, and prove **Non-Inferiority** of groundedness.

### 2. Concept to Understand First (5 minutes)
- **Ablation Study**: Systematically disabling individual components to quantify their exact contribution to the system's performance.
- **Non-Inferiority Testing**: A statistical proof demonstrating that an optimization (here, caching and memory retrieval) does not degrade output quality (Groundedness and Citation Precision) by more than a pre-specified margin $\Delta$ (e.g., 2.0%).

### 3. Exact Commands
- **Run the full 4-condition ablation benchmark**:
  ```powershell
  uv run python -m evidenceflow.eval.run_ablation --repeats 3 --output docs/experiments.md
  ```
  *Explanation*: Runs 4 pipeline configurations x 3 repetitions across the dev benchmark split.  
  *Undo*: N/A

### 4. Ownership Split
- **Gemini Generates**: Statistical calculation scripts (mean, standard deviation, confidence intervals, p-value calculations).
- **You Write**: `src/evidenceflow/eval/ablation_runner.py` (Pipeline toggling logic dynamically disabling/enabling memory and cache nodes).

### 5. Verification
Inspect the results table in `docs/experiments.md` to confirm:
- Latency drop for cached repeated queries (> 90% reduction).
- Token reduction across multi-turn sessions with memory graph.
- Groundedness score remains within the non-inferiority margin ($\Delta \le 2\%$) compared to baseline.

### 6. Break It on Purpose
Set the non-inferiority margin in the script to an impossible `0.00001%` and observe the statistical check flag a non-inferiority warning due to standard LLM generation variance. Understand why confidence intervals and realistic margins are necessary.

### 7. Explain-Back Questions
1. In an ablation study, why do we run each experimental condition 3 times rather than once?
2. How do you explain "non-inferiority" to an engineering hiring manager in one sentence?
3. If caching reduced latency by 85% and tokens by 80%, but dropped citation precision from 98% to 91%, would you deploy it to production? Explain why or why not.

### 8. Git Commit Message
`experiment(eval): conduct 4-way ablation study proving cache non-inferiority and token savings`

### 9. Daily Shutdown Routine
```powershell
docker compose stop
```

---

# Tier 3: Stand Out (Cross-Version Reasoning & Advanced Evaluation)

## Day 14: Contradiction Analyst & Cross-Version Graph Reasoning

### 1. Goal
Implement the `ContradictionAnalyst` agent node and enrich the Neo4j graph with `:Claim` nodes connected by `CONTRADICTS` and `SUPERSEDES` edges to explicitly reason over deprecated configurations and changed defaults.

### 2. Concept to Understand First (5 minutes)
- **Cross-Version Contradiction Detection**: When a user asks: *"Did the default of vacuum_cost_limit change between Postgres 14 and 16?"*, the system must retrieve both passages, identify the semantic contradiction in their parameter values, and emit explicit warnings with both versions cited.
- **Supercedence Edges in Knowledge Graphs**: Linking `(:Claim {version: '16'})-[:SUPERSEDES]->(:Claim {version: '14'})` provides structural reasoning capabilities that standard flat vector search cannot achieve.

### 3. Exact Commands
- **Test contradiction detection logic**:
  ```powershell
  uv run pytest tests/test_contradiction.py -v
  ```
  *Explanation*: Asserts that opposing parameter claims across versions are flagged as contradictory.  
  *Undo*: N/A

### 4. Ownership Split
- **Gemini Generates**: Few-shot prompt templates for cross-version contradiction extraction.
- **You Write**: `src/evidenceflow/graph/nodes/contradiction.py` and Neo4j edge generation functions linking superseded claims.

### 5. Verification
Run a cross-version comparative query:
```powershell
uv run python -m evidenceflow.graph.workflow --query "Compare vacuum_cost_limit in Postgres 14 vs 16"
```
Verify the answer explicitly cites both versions, outlines the change, and flags the `SUPERSEDES` relation.

### 6. Break It on Purpose
Feed two identical, non-contradictory passages into the contradiction analyst. Verify that the agent does *not* hallucinate a conflict and returns `contradiction_detected: false`.

### 7. Explain-Back Questions
1. Why does traditional RAG struggle with cross-version comparative questions?
2. How do `SUPERSEDES` edges in Neo4j simplify answering migration and upgrade queries?
3. What is the difference between an outright contradiction (two passages asserting opposing facts about the same version) and an evolutionary change (a setting changed across versions)?

### 8. Git Commit Message
`feat(reasoning): implement contradiction analyst node with neo4j supersedes edge wiring`

### 9. Daily Shutdown Routine
```powershell
docker compose stop
```

---

## Day 15: Independent L2 Judge for Contested Claims

### 1. Goal
Implement an independent L2 Judge node using an alternative model family (or specialized strict evaluator prompt) invoked selectively *only* when the L1 verifier detects low entailment confidence or cross-version contradictions.

### 2. Concept to Understand First (5 minutes)
- **Tiered Evaluation (L1 vs L2)**: Running an expensive judge model on every query is cost-prohibitive. EvidenceFlow AI uses a 2-tier architecture: cheap deterministic L1 verification on 100% of queries, and an independent L2 Judge invoked *only* on the 5-10% of queries where claims are contested or ambiguous.
- **Model Independence**: Avoiding self-evaluation bias by ensuring the Judge prompt or model is isolated from the generation prompt.

### 3. Exact Commands
- **Run the L2 Judge test suite**:
  ```powershell
  uv run pytest tests/test_l2_judge.py -v
  ```
  *Explanation*: Tests selective invocation of L2 Judge and verdict arbitration.  
  *Undo*: N/A

### 4. Ownership Split
- **Gemini Generates**: Scoring rubric definitions and JSON output parsing utilities.
- **You Write**: `src/evidenceflow/graph/nodes/judge.py` (Conditional routing logic deciding when to invoke L2, and the arbitration verdict generator).

### 5. Verification
Trigger a low-confidence claim query and verify in the step ledger logs that the L2 Judge node was activated and recorded an arbitration verdict.

### 6. Break It on Purpose
Configure the router so that the L2 Judge is invoked on *every* single query. Run a 20-query batch, measure the 300% surge in latency and cost in `experiments.md`, and verify why the tiered selective invocation is required.

### 7. Explain-Back Questions
1. Why is having an LLM judge its own output considered an architectural anti-pattern?
2. What are the specific triggers that route a query to the L2 Judge in EvidenceFlow AI?
3. How does selective L2 invocation maintain high groundedness while keeping LLM costs near zero for standard queries?

### 8. Git Commit Message
`feat(eval): add independent L2 judge node with selective conditional triggering`

### 9. Daily Shutdown Routine
```powershell
docker compose stop
```

---

## Day 16: Competitive Architectural Benchmark: EvidenceFlow vs. Baselines

### 1. Goal
Execute a head-to-head empirical benchmark comparing **EvidenceFlow AI** against two industry baselines: (1) Naive `pgvector` + full-text search, and (2) PostgreSQL recursive SQL joins, measuring Recall@k, Groundedness, and Latency.

### 2. Concept to Understand First (5 minutes)
- **Honest Architectural Comparison**: An enterprise AI portfolio must prove why the chosen stack (Qdrant + Neo4j + Scoped Cache) outperforms a single monolithic Postgres database with `pgvector`.
- **Measuring Trade-offs**: Documenting where Postgres excels (e.g. simple single-table lookups) vs. where EvidenceFlow dominates (multi-turn entity tracking and filtered vector recall).

### 3. Exact Commands
- **Run the competitive comparative benchmark**:
  ```powershell
  uv run python -m evidenceflow.eval.benchmark_competitors --output docs/experiments.md
  ```
  *Explanation*: Runs identical test suites through EvidenceFlow, pgvector baseline, and recursive SQL.  
  *Undo*: N/A

### 4. Ownership Split
- **Gemini Generates**: Raw pgvector schema scripts and SQL execution harnesses.
- **You Write**: `src/evidenceflow/eval/comparator.py` (The benchmarking harness executing queries across all three engines and tabulating metrics).

### 5. Verification
Open `docs/experiments.md` and confirm the comparative table contains:
- Recall@5 across all three systems on version-filtered queries
- Latency comparison across single-turn and multi-turn questions
- Abstention accuracy when faced with unanswerable questions

### 6. Break It on Purpose
Run a version-filtered query on `pgvector` with a highly selective filter (`WHERE version = '16'`). Observe the query planner fall back to a slow sequential filter scan or drop relevant items, contrasting with Qdrant's single-stage filtered HNSW.

### 7. Explain-Back Questions
1. Under what specific conditions does single-stage filtered HNSW in Qdrant outperform `pgvector` index scans?
2. Why does recursive SQL fail when queries involve imprecise or semantic entity follow-ups?
3. How do you defend the operational complexity of managing 3 database technologies to a CTO?

### 8. Git Commit Message
`benchmark(eval): record competitive results comparing evidenceflow against pgvector and recursive sql`

### 9. Daily Shutdown Routine
```powershell
docker compose stop
```

---

## Day 17: External API FastMCP Server with Rate Limiting & Resilience

### 1. Goal
Build a resilient external-API FastMCP server wrapping official PostgreSQL release metadata APIs, featuring token bucket rate-limiting, exponential backoff retries, and schema validation.

### 2. Concept to Understand First (5 minutes)
- **Production Tool Resilience**: External APIs fail, experience network jitter, or rate-limit clients with HTTP 429. An enterprise AI system must wrap external tools with deterministic rate-limiters, exponential retries, and fallback degradation rather than crashing the agent workflow.
- **Schema Validation with Pydantic**: Validating external API JSON payloads against strict schemas before passing them into LLM context.

### 3. Exact Commands
- **Start external API MCP server**:
  ```powershell
  uv run python -m evidenceflow.mcp.external_server
  ```
  *Explanation*: Launches external metadata FastMCP server with rate limiting.  
  *Undo*: Press `Ctrl+C`.
- **Run resilience and fault injection tests**:
  ```powershell
  uv run pytest tests/test_external_mcp_resilience.py -v
  ```
  *Explanation*: Injects HTTP 429 and 500 errors to verify retry and backoff behavior.  
  *Undo*: N/A

### 4. Ownership Split
- **Gemini Generates**: Mock external HTTP server endpoints for fault testing.
- **You Write**: `src/evidenceflow/mcp/external_server.py` (Rate-limiter decorator, retry loop with jitter, and Pydantic response models).

### 5. Verification
Simulate an external API outage and verify that the workflow handles the failure gracefully by returning a structured notice rather than an unhandled crash.

### 6. Break It on Purpose
Temporarily set the timeout to `0.001` seconds to force a `TimeoutException`. Verify that the retry loop triggers, exhausts retries, and emits a clean degradation response.

### 7. Explain-Back Questions
1. Why is exponential backoff with jitter superior to fixed-interval retries when dealing with rate-limited APIs?
2. What happens to the user experience if an external MCP tool crashes without error handling in LangGraph?
3. Why should external API responses always be parsed into Pydantic models before being injected into the agent's prompt?

### 8. Git Commit Message
`feat(mcp): implement external api fastmcp server with rate limiting and retry resilience`

### 9. Daily Shutdown Routine
```powershell
docker compose stop
```

---

## Day 18: CI Regression Test Gate & Release Packaging

### 1. Goal
Establish a continuous integration (CI) test gate that runs on every pull request, enforcing that Groundedness ($\ge 95\%$), Recall@5 ($\ge 85\%$), and Cache False-Hit Rate ($= 0.0\%$) do not regress, and package the final release.

### 2. Concept to Understand First (5 minutes)
- **CI Quality Gates in AI**: Software engineering uses unit test coverage gates (e.g. 80% coverage). AI engineering requires metric regression gates: if a prompt change or chunker adjustment causes Groundedness to drop below 95%, the CI build fails and blocks the PR.
- **One-Command Reproducibility**: The entire project—from container boot to benchmark completion—must execute with a single script for interview reviewers.

### 3. Exact Commands
- **Run the complete CI evaluation gate locally**:
  ```powershell
  uv run python -m evidenceflow.eval.ci_gate
  ```
  *Explanation*: Runs the automated gate asserting quality thresholds. Exits with code 0 on pass, 1 on regression.  
  *Undo*: N/A
- **Run the full test suite across the repo**:
  ```powershell
  uv run pytest tests/ -v
  ```
  *Explanation*: Executes all unit, integration, and failure tests across all modules.  
  *Undo*: N/A

### 4. Ownership Split
- **Gemini Generates**: GitHub Actions workflow YAML (`.github/workflows/ci.yml`) and Docker build configurations.
- **You Write**: `src/evidenceflow/eval/ci_gate.py` (The regression threshold evaluation script and exit-code enforcer).

### 5. Verification
Verify that `ci_gate.py` outputs green checkmarks across all metric thresholds and returns exit code `0`:
```powershell
echo $LASTEXITCODE
```
(Expected: `0`)

### 6. Break It on Purpose
Raise the Groundedness threshold in `ci_gate.py` to `101%`. Run the script and observe it exit with code `1` and a red CI failure notification explaining the regression. Revert back to `95%`.

### 7. Explain-Back Questions
1. How does an AI regression gate differ from a standard unit test suite?
2. Why is having a single-command reproducible benchmark run crucial for portfolio credibility?
3. If a new prompt boosts retrieval recall from 85% to 92%, but drops citation precision from 98% to 89%, should the CI gate pass or fail? Explain the trade-off.

### 8. Git Commit Message
`ci(eval): implement automated regression gate and complete tier 3 release packaging`

### 9. Daily Shutdown Routine
```powershell
docker compose stop
```
