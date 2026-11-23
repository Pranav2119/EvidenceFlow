# EvidenceFlow AI

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Qdrant](https://img.shields.io/badge/vector--db-Qdrant-red.svg)](https://qdrant.tech/)
[![PostgreSQL](https://img.shields.io/badge/database-PostgreSQL--16-blue.svg)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**EvidenceFlow AI** is an inspectable research-assistance prototype for complex, multi-turn technical questions over versioned PostgreSQL documentation (versions 14, 15, and 16).

> [!IMPORTANT]
> **Readiness review is in progress.** [CI at commit `aa83738`](https://github.com/Pranav2119/EvidenceFlow/actions/runs/45172813930) passed 249 tests and the configured metric gates. That verifies those checks, not production readiness or every historical performance claim. Evaluation repairs have passed 41 focused local tests; fresh end-to-end measurements and remaining correctness repairs are still required. See the [review roadmap](docs/ROADMAP.md) and [experiment validity notice](docs/experiments.md).

The baseline path combines:
1. **Rule-Based Guardrails & Abstention Routing**: Checks inputs and retrieved evidence, with refusal paths for detected unsafe or unsupported requests.
2. **Hybrid Retrieval (Dense + BM25)**: Combines 384-dimensional Qdrant dense retrieval with local Okapi BM25 rankings through Reciprocal Rank Fusion (RRF).
3. **L1 & L1.5 Citation and Value Checks**: Checks cited passages and nearby quantity/setting matches. Known validation gaps remain under review; passing these checks is not proof that every claim is supported.
4. **Step Ledger & PostgreSQL Checkpointing**: Stores conversation checkpoints through `PostgresSaver` and records step and claim-evidence data in PostgreSQL. Token-accounting coverage still needs validation before total-cost claims.

---

## Architecture Overview

```
User Query
   │
   ▼
[input_guardrail] ──(Injections / Out-of-Scope)──► [abstention_node] ──► END
   │ (Safe)
   ▼
[planner_node]
   │ (Sub-queries)
   ▼
[retriever_node] ──(Dense + BM25 Hybrid RRF)──► (Low Score / Missing Keyword) ──► [abstention_node] ──► END
   │ (Passages Retrieved)
   ▼
[answer_writer_node] (Gemini Flash-Lite / Fail-Closed Deterministic Fallback)
   │ (Draft Answer + Inline Citations)
   ▼
[verifier_node] ──(L1 Hallucinated Citation / L1.5 Value Mismatch)──► [abstention_node] ──► END
   │ (PASS)
   ▼
[END] (Answer Delivered + Turns & Step Ledger Persisted in PostgreSQL)
```

---

## Infrastructure & Setup Guide

### 1. Prerequisites
- **Python 3.13+**
- **uv** (Ultra-fast Python package installer): `pip install uv` or `winget install astral-sh.uv`
- **Docker Desktop** (for containerized PostgreSQL, Qdrant, and Neo4j)

### 2. Environment Configuration
Create a `.env` file in the project root:
```env
# Gemini API Key (Optional; system gracefully falls back to deterministic offline synthesis)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_EMBEDDING_MODEL=text-embedding-004

# Embedding Provider: 'fastembed' (local 384-dim, zero API key) or 'gemini'
EMBEDDING_PROVIDER=fastembed

# PostgreSQL Settings (Checkpointer, Step Ledger & Claim Audit)
POSTGRES_USER=evidenceflow
POSTGRES_PASSWORD=evidenceflow_dev_secret
POSTGRES_DB=evidenceflow_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Qdrant Vector Store Settings
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334

# Neo4j Settings (Tier 2 Conversational Graph Memory)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=evidenceflow_graph_secret
```

> [!WARNING] Production Password Security & Refusal Guard
> The `*_secret` passwords (`evidenceflow_dev_secret`, `evidenceflow_app_secret`, `evidenceflow_graph_secret`) shown above are **local-dev placeholders only**. When `APP_ENV=production` (or `ENV=production`), the application strictly refuses to start if any of these placeholder passwords remain configured in `.env` or system environment variables.

### 3. Start Database Services via Docker Compose
EvidenceFlow includes pre-configured services with persistent volumes and healthchecks in `docker-compose.yml`:
```powershell
# Start PostgreSQL, Qdrant, and Neo4j in the background
docker compose up -d

# Verify all services are healthy
docker compose ps
```

Services exposed:
- **PostgreSQL 16**: Port `5432` (`evidenceflow-postgres`)
- **Qdrant Vector DB**: Port `6333` (REST) & `6334` (gRPC) (`evidenceflow-qdrant`)
- **Neo4j 5**: Port `7474` (Browser UI) & `7687` (Bolt) (`evidenceflow-neo4j`)

### 4. Install Dependencies
Run from the repository root. This creates or updates `.venv` using the versions in
[uv.lock](uv.lock), without upgrading or rewriting the lockfile.

```powershell
uv sync --frozen
```

### 5. Run Database Migrations
Initialize LangGraph checkpointer schemas, immutable audit tables, and append-only database triggers:
```powershell
uv run --frozen python -m evidenceflow.storage.migrate_postgres
```

### 6. Initialize the Corpus and Memory Schema
For a fresh setup, index the committed [corpus chunks](data/processed/chunks.jsonl)
into Qdrant and initialize Neo4j before running queries or integration tests:

```powershell
uv run --frozen python -m evidenceflow.ingestion.index_qdrant
uv run --frozen python -m evidenceflow.memory.init_neo4j
```

With `EMBEDDING_PROVIDER=fastembed`, the first indexing run downloads the local
embedding model; it does not need a Gemini API key. The indexing command creates
a missing collection or upserts stable passage IDs. It does not recreate an
existing collection.

### Updating an Existing Local Checkout
After changes are merged into the branch you use locally, pull from a clean
working tree and synchronize the virtual environment:

```powershell
git pull --ff-only
uv sync --frozen
docker compose up -d
docker compose ps
```

Keep your existing `.env`, embedding provider, and Docker volumes. These CI
configuration changes do not require rebuilding an already initialized local
database or corpus. Use steps 5-6 for a fresh setup or missing database resources.
Do not use `docker compose down -v`, `--drop`, or `--recreate` for a routine pull.
Restart any running EvidenceFlow Python/MCP process after pulling application-code
updates so it loads the new code; keep the existing database volumes.

To stop the local services without deleting their data:

```powershell
docker compose stop
```

---

## Usage

### Run a Single Query
```powershell
uv run python -m evidenceflow.graph.workflow --query "What is the default of shared_buffers in Postgres 16?" --version "16"
```

### Run Multi-Turn with Process-Restart Checkpointing
```powershell
# Turn 1 on thread 'dba-session-1'
uv run python -m evidenceflow.graph.workflow --thread-id "dba-session-1" --query "What is max_connections in Postgres 16?" --use-checkpoint

# Graph state is persisted to PostgreSQL and isolated from other threads.
```

---

### FastMCP Read-Only Corpus Server (`corpus-mcp`)
EvidenceFlow exposes an inspectable, read-only Model Context Protocol (MCP) server over the versioned PostgreSQL documentation corpus:
- **Stdio Transport**: Clean standard I/O communication using JSON-RPC. All application and diagnostic logging is strictly routed to `sys.stderr` to prevent wire protocol corruption.
- **Read-Only Invariant**: All registered tools are strictly read-only. Mutation operations are prohibited.
  - *Storage Note*: Qdrant runs without an API key in local development; read-only enforcement at this layer is architectural (only search/point query APIs are invoked; no mutation handlers exist).
  - *Security Note*: Name-based tool validation is a pipeline contract guardrail, not a cryptographic or OS-level security boundary.
- **The Four Front-Desk Tools**:
  1. `list_versions`: Discovers indexed versions (`14`, `15`, `16`), chunk counts, and Qdrant storage health.
  2. `search_passages`: Hybrid dense + lexical retrieval with version filtering, query length limits (<= 500 chars), and result clamping (1 <= limit <= 20).
  3. `get_passage`: Direct point lookup from Qdrant by validated `chunk_id` (`^pg(14|15|16)-[a-z0-9_-]+$`).
  4. `diff_sections`: Side-by-side cross-version configuration comparison returning structured fields (`found_v1`, `found_v2`, `changed`, `default_v1`, `default_v2`) and a readable unified diff.

**Start the MCP Server**:
```powershell
uv run python -m evidenceflow.mcp.corpus_server
```

---

## Automated Testing & Benchmarks

### 1. Run the Full Test Suite
Runs unit, integration, and failure tests covering prompt injections, out-of-scope detection, value verifier normalization, RBAC role restrictions, append-only triggers, process-restart recovery, and thread isolation. Integration tests require the initialized services from the setup steps above:
```powershell
uv run --frozen pytest tests -v
```

The [GitHub Actions workflow](.github/workflows/ci.yml) uses uv `0.13.5`,
Python 3.13, and fresh PostgreSQL, Qdrant, and Neo4j service containers. It runs
migrations, indexes the committed corpus, and initializes the memory schema
before the full test suite and metric gate. CI selects local FastEmbed embeddings
and disables Gemini calls; these job-level settings do not change your local
`.env`. Test failures and metric regressions still fail the job.

CI caches the configured embedding model using the lockfile and embedding module
as the cache key inputs. Model preparation allows three attempts with five-minute
pauses to handle upstream download rate limits. The model cache is saved before
tests, after an offline load and vector-size check; indexing and tests then load
it with Hugging Face offline mode enabled.
This cache does not include corpus data, conversation history, or API keys.

### 2. CI Development Baseline (Gemini Off)

CI runs the 40-question development dataset after database initialization and
before the tests can alter database contents. Each question gets one discarded
warm-up and three timed executions: 40 warm-ups and 120 timed observations.
Gemini, conversational memory, the scoped answer cache, and checkpointing are
disabled. Normal model/connection warm-up is retained.

Download `dev-baseline-<run-id>-<attempt>` from the workflow run's **Artifacts**
section. Its `dev-baseline.json` records the commit, configuration, per-repeat
draft answers, citations, ranked retrieved passage IDs, observed version scope,
refusal information, and cache-hit flags. Draft text on a refused turn is audit
data, not an answer delivered to the user. An observed cache hit invalidates this
uncached benchmark; a missing report fails the upload step.

These are deterministic diagnostic results, not Gemini synthesis measurements.
The legacy `overall_accuracy_pct` field measures answer/refusal decisions, not
factual correctness. Quality aggregates use the last timed repeat per question;
all timed observations are retained for inspection. L1 flags and citation markers
are not independent proof that claims are supported. CI latency is not a
production SLA or proof of a performance improvement across different runners.
Artifacts follow repository retention settings; retain the selected baseline
when recording comparisons in [experiments.md](docs/experiments.md).

### 3. Run the 40-Question Guardrail Benchmark Exam
Evaluates the entire pipeline against legitimate questions, prompt injections, fake technical features, and out-of-domain queries:
```powershell
uv run python -m evidenceflow.eval.run_guardrail_exam --delay 1.5
```

---

## Auditability & Append-Only Invariants

Every conversation turn, node step, and verified claim is stored in PostgreSQL:
- `turns`: Stores final user queries, draft answers, turn numbers, and abstention codes.
- `step_ledger`: Stores node transitions, actions, status, decided_by (`rule`, `llm`, `fallback`), execution latency in milliseconds, token counts, and sanitized details. **`step_number` is a thread-wide sequential counter** that monotonically increments across all turns within a conversation thread (e.g. Turn 1: steps 1–5, Turn 2: steps 6–10).
- `claim_audit`: Stores the exact statement, cited chunk ID, chunk cryptographic SHA-256 hash, documented version, claimed values, and verifier code.

**Database-Level Immutability & Defense-in-Depth**:
- **Layer 1 (RBAC Least-Privilege Role)**: Application runtime connects as `evidenceflow_app` which possesses only `SELECT` and `INSERT` permissions on audit tables; `UPDATE`, `DELETE`, and `TRUNCATE` are revoked at the PostgreSQL engine privilege level.
- **Layer 2 (PL/pgSQL Engine Triggers)**: Row-level and statement-level triggers block all `UPDATE`, `DELETE`, and `TRUNCATE` operations on `turns`, `step_ledger`, and `claim_audit`, raising security violation exceptions if an operation is ever attempted.

---

## Known Limitations & Roadmap
- For documented edge cases, boundary sensitivity, and deferred items, see [docs/known-limitations.md](docs/known-limitations.md).
- For milestone tracking and progress, see [docs/ROADMAP.md](docs/ROADMAP.md).
- For evaluation benchmarks and threshold tuning experiments, see [docs/experiments.md](docs/experiments.md).
