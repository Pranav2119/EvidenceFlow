# EvidenceFlow AI: Project Roadmap & Progress Tracker

## Design & Specification Stages
- [x] **Stage 1**: Concept, USP, and Architectural Justifications (`docs/design/01_concept_and_usp.md`)
- [x] **Stage 2**: System Architecture, Multi-turn Flow, Neo4j Graph, and 5-Layer Cache (`docs/design/02_architecture.md`)
- [x] **Stage 3**: File-Level Repo Structure & Core Skeletons (Pydantic schemas, Workflow, FastMCP, Cache Validator) (`docs/design/03_repo_structure_and_skeletons.md`)
- [x] **Stage 4**: Day-wise Roadmap per Tier in Learning Mode Format (`docs/design/04_daywise_roadmap.md`)
- [x] **Stage 5**: Evaluation Plan (Metrics, Splits, Ablation, Non-inferiority, Failure-tests) (`docs/design/05_evaluation_plan.md`)
- [x] **Stage 6**: Portfolio Packaging (README, ADRs, 2-minute Demo Script, Top 10 Interview Q&A) (`docs/design/06_portfolio_packaging.md`)
- [x] **Stage 7**: Resume Bullet Templates & ATS Keywords (`docs/design/07_resume_and_ats.md`)

---

## Build Tiers Overview

### Tier 1: Measure First (Baseline Pipeline & Corpus Ingestion)
- [x] Day 0: Tooling installs (pinned uv/Python, Docker Compose, Git init, .env validation)
- [x] Day 1: Versioned PostgreSQL Documentation parser & chunker with SHA-256 hashes
- [x] Day 2: Qdrant vector store setup with dense + BM25 hybrid search & version payload filters
- [x] Day 3: Baseline LangGraph state machine (`Planner` -> `Retriever` -> `AnswerWriter` -> `L1Verifier`)
- [x] Day 4: Standardized Abstention mechanism & input/output guardrails v1
- [ ] Day 5: PostgreSQL checkpointer, step ledger, and claim-evidence audit trail tables
- [ ] Day 6: FastMCP `corpus-mcp` read-only server
- [ ] Day 7: Benchmark v1 with deterministic scoring & baseline measurements recorded in `experiments.md`

### Tier 2: The Differentiator (Memory & 5-Layer Cache)
- [ ] Neo4j conversational graph memory (`:Turn`, `:Question`, `:Entity`, `:Claim`, `:Version`)
- [ ] 5-Layer scoped cache implementation with composite HMAC keys
- [ ] Zero-LLM deterministic cache hit validator & automatic stale-chunk eviction
- [ ] FastMCP `memory-cache-mcp` server with role-based write isolation
- [ ] Cache collision trap tests & document patch staleness tests
- [ ] Tier 2 Ablation study: Baseline vs. +Memory vs. +Cache vs. +Both (Non-inferiority analysis)

### Tier 3: Stand Out (Cross-Version Reasoning & Advanced Evaluation)
- [ ] Contradiction Analyst agent with `SUPERSEDES` and `CONTRADICTS` graph edges
- [ ] Independent L2 Judge for contested claims
- [ ] Benchmark comparison: EvidenceFlow vs. pgvector + full-text vs. recursive SQL
- [ ] External API FastMCP server with rate-limiting & schema validation
- [ ] CI regression test gate (preventing recall/groundedness regressions)
