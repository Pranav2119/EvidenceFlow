# EvidenceFlow AI: Step-by-Step Implementation Log

This document records every concrete step, command executed, files modified/created, rationale, and verification output across the lifecycle of EvidenceFlow AI.

---

## Day 0: Tooling, Repositories, Environment, & Container Infrastructure

### Step 0.1: System Diagnostics & Tooling Verification
- **Goal**: Ensure Python, `uv`, Git, and Docker Desktop are operational.
- **Commands Executed**:
  ```powershell
  uv --version       # Verified uv version 0.13.5
  python --version   # Verified Python 3.13.5
  git status         # Verified repository on branch main
  ```
- **Docker Setup**: Installed Docker Desktop with WSL 2 backend. Docker daemon (`v29.9.2`) and Docker Compose (`v5.6.1`) verified.

### Step 0.2: Credential & Artifact Isolation (`.gitignore`)
- **Goal**: Prevent committing API keys, virtual environment binaries, and database volumes to version control.
- **File Created**: [`.gitignore`](file:///.gitignore)
- **Ignored Paths**: `.env`, `.env.*`, `.venv/`, `__pycache__/`, `postgres_data/`, `qdrant_data/`, `neo4j_data/`.

### Step 0.3: Environment Variable Parity (`.env.example` & `.env`)
- **Goal**: Provide a public template with placeholder keys while maintaining local credentials.
- **File Created**: [`.env.example`](file:///.env.example)
- **Command Executed**:
  ```powershell
  Copy-Item .env.example .env
  ```
- **Keys Configured**: `GEMINI_API_KEY`, `POSTGRES_*` (`5432`), `QDRANT_*` (`6333`, `6334`), `NEO4J_*` (`7474`, `7687`).

### Step 0.4: Dependency Management (`pydantic-settings`)
- **Goal**: Add runtime settings validation library for Pydantic v2.
- **Command Executed**:
  ```powershell
  uv add pydantic-settings
  ```
- **Files Updated**: `pyproject.toml`, `uv.lock`.

### Step 0.5: Multi-Database Container Orchestration (`docker-compose.yml`)
- **Goal**: Provision PostgreSQL 16, Qdrant v1.13.2, and Neo4j v5.26.0 with health checks.
- **File Created**: [`docker-compose.yml`](file:///docker-compose.yml)
- **Command Executed**:
  ```powershell
  docker compose up -d
  docker compose ps
  ```
- **Verification**: All 3 containers reported `healthy` and active.
- **Port Socket Check**:
  ```powershell
  uv run python -c "import socket; [socket.create_connection(('localhost', p), timeout=2) for p in (5432, 6333, 7687)]; print('All database ports active!')"
  ```
  *Result*: `All database ports active!`

### Step 0.6: Centralized Configuration Schema (`src/evidenceflow/config.py`)
- **Goal**: Parse `.env` with strict type annotations, default connection strings, and fail-fast startup validation.
- **File Created**: [`src/evidenceflow/config.py`](file:///src/evidenceflow/config.py)
- **Verification**:
  ```powershell
  uv run python -c "from evidenceflow.config import settings; print('Config loaded successfully!'); print(f'Postgres: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}'); print(f'Qdrant: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}'); print(f'Neo4j: {settings.NEO4J_URI}')"
  ```
  *Result*: All connection strings successfully printed.

### Step 0.7: Git Commit (Day 0)
- **Commit**: `chore(infra): initialize Day 0 docker compose, env validation, and tooling configs` (Commit `65d4a48`).

---

## Day 1: Versioned PostgreSQL Documentation Parser & Chunking Pipeline

### Step 1.1: Dependency Installation (`beautifulsoup4`)
- **Goal**: Add robust HTML/XML parsing library.
- **Command Executed**:
  ```powershell
  uv add beautifulsoup4
  ```
- **Files Updated**: `pyproject.toml`, `uv.lock`.

### Step 1.2: Raw Documentation Downloader (`download_docs.py`)
- **Goal**: Fetch official PostgreSQL documentation pages for versions 14, 15, and 16 into `data/raw/{version}/`.
- **Files Created**: [`src/evidenceflow/ingestion/download_docs.py`](file:///src/evidenceflow/ingestion/download_docs.py)
- **Command Executed**:
  ```powershell
  uv run python -m evidenceflow.ingestion.download_docs --versions 14 15 16
  ```
- **Verification**: 18 files downloaded across `data/raw/14`, `data/raw/15`, and `data/raw/16` (release notes and server runtime configs).

### Step 1.3: Hierarchy-Aware Chunker & Breadcrumbs (`chunker.py`)
- **Goal**: Parse official PostgreSQL HTML manuals, extract section breadcrumbs (e.g. `Server Configuration > Automatic Vacuuming` or `20.4. Resource Consumption > 20.4.4. Cost-based Vacuum Delay`), handle consecutive `<dt>` elements sharing a single `<dd>`, and embed the breadcrumb into both the searchable content text and discrete metadata.
- **File Updated**: [`src/evidenceflow/ingestion/chunker.py`](file:///src/evidenceflow/ingestion/chunker.py)
- **Command Executed**:
  ```powershell
  uv run python -m evidenceflow.ingestion.chunker
  ```
- **Verification**: Generated **1,261 structured chunks** in `data/processed/chunks.jsonl`. Verified by eye on `autovacuum_naptime`, `vacuum_cost_limit`, `shared_buffers`, and release notes across PostgreSQL 14, 15, and 16.

### Step 1.4: Deterministic Fingerprint Diff & Staleness Tool (`diff_chunks.py`)
- **Goal**: Put the SHA-256 chunk fingerprints to work by comparing revisions or versions without calling an LLM. Detects unchanged chunks (0-token cache hits), changed definitions/defaults (staleness triggers), and additions/deprecations.
- **File Created**: [`src/evidenceflow/ingestion/diff_chunks.py`](file:///src/evidenceflow/ingestion/diff_chunks.py)
- **Command Executed**:
  ```powershell
  uv run python -m evidenceflow.ingestion.diff_chunks --cross-version 15 16
  ```
- **Verification Result**:
  - Unchanged parameters: 200 (Safe in cache)
  - Changed parameters: 8 (Trigger cache eviction: `vacuum_freeze_min_age`, `wal_sync_method`, `archive_command`, etc.)
  - Added in PG16: 10 (`createrole_self_grant`, `icu_validation_level`, etc.)
  - Removed/Deprecated in PG16: 1 (`force_parallel_mode`)

### Step 1.5: Principal Architect Code Review & Hardening Refactor
- **Goal**: Audit all code files against Staff/Principal AI Engineering standards before proceeding to vector indexing.
- **Review Document**: [`docs/code_review_day1.md`](file:///docs/code_review_day1.md)
- **Key Fixes Implemented Across 5 Files**:
  1. `chunker.py`: Stable GUC-derived chunk IDs (e.g. `pg16-runtime-config-resource-shared-buffers`), NFC normalization, per-`<dt>` breadcrumb calculation, child list item decoupling in release notes, and standard logging.
  2. `config.py`: Dynamic `DATABASE_URL` computation via Pydantic `@computed_field`, lazy singleton `get_settings()` with `@lru_cache`, and `require_gemini_key()` fail-fast helper.
  3. `download_docs.py`: Exponential backoff retry loop (3 attempts), `#docContent` validation instead of magic byte threshold, and expanded corpus coverage to 33 pages (including connection, logging, locks, and replication configs).
  4. `diff_chunks.py`: Cross-version diff key includes section to prevent title collisions; removed dead imports.
  5. `pyproject.toml` & `.env.example`: Production portfolio description, Neo4j 5.x password requirements noted.
- **Verification Commands Executed**:
  ```powershell
  # 1. Download expanded 33-page corpus with retry verification
  uv run python -m evidenceflow.ingestion.download_docs --versions 14 15 16

  # 2. Run hardened chunker with stable IDs and NFC normalization
  uv run python -m evidenceflow.ingestion.chunker
  # Result: 1,581 structured chunks generated in data/processed/chunks.jsonl

  # 3. Re-run fingerprint cross-version diff on expanded corpus
  uv run python -m evidenceflow.ingestion.diff_chunks --cross-version 15 16
  # Result: 265 unchanged parameters, 19 changed parameters, 43 added, 31 removed
  ```
