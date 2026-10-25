# EvidenceFlow AI

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Qdrant](https://img.shields.io/badge/vector--db-Qdrant-red.svg)](https://qdrant.tech/)
[![PostgreSQL](https://img.shields.io/badge/database-PostgreSQL--16-blue.svg)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**EvidenceFlow AI** is an inspectable, deterministic research-assistance platform for complex, multi-turn technical questions over versioned PostgreSQL documentation (versions 14, 15, and 16).

Unlike conventional RAG chatbots that produce ungrounded hallucinations when features change or queries ask for non-existent settings, EvidenceFlow enforces:
1. **Deterministic Guardrails & Abstention Routing**: Intercepts prompt injections, out-of-scope domain queries, ungrounded technical tokens, and cross-version discrepancies before token generation.
2. **True Hybrid Retrieval (Dense + BM25)**: Fuses 384-dimensional dense semantic embeddings with lexical Okapi BM25 rankings via Reciprocal Rank Fusion (RRF), ensuring exact technical keywords and parameter names rank at #1.
3. **L1 & L1.5 Deterministic Verification**: Every factual statement is verified against the cited documentation chunk before output delivery. Quantities, units, and configuration bounds (`default`, `minimum`, `maximum`) are strictly validated in the same sentence as the claim.
4. **ACID Step Ledger & PostgreSQL Checkpointing**: Conversation threads survive process restarts via `PostgresSaver`. Every node transition, model latency, token count, and claim SHA-256 hash is recorded in an immutable, append-only PostgreSQL ledger.

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
```powershell
uv sync
```

### 5. Run Database Migrations
Initialize LangGraph checkpointer schemas, immutable audit tables, and append-only database triggers:
```powershell
uv run python -m evidenceflow.storage.migrate_postgres
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

## Automated Testing & Benchmarks

### 1. Run the Full Test Suite
Executes 38 automated tests covering prompt injections, out-of-scope detection, value verifier normalization, RBAC role restrictions, append-only triggers, process-restart recovery, and thread isolation:
```powershell
uv run pytest tests/ -v
```

### 2. Run the 40-Question Guardrail Benchmark Exam
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
