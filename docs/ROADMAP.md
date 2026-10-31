# EvidenceFlow AI: Project Roadmap & Progress Tracker

## Learning Mode & Execution Protocol
Every Day/Task strictly follows this 3-phase cadence:
1. **Upfront Grounding (Start)**: 3-5 line concept explanation + concrete real-world scenario BEFORE writing code or running shell commands.
2. **Step-by-Step Verifiable Build**: Isolated feature branches (`feat/day-X-feature-name`), surgical diffs, and automated tests.
3. **Closing Retrospective (End)**:
   - 2-3 Explain-back questions for interview defense.
   - Comprehensive detailed summary with real-world examples, input/output traces, and failure modes caught.
   - Push to feature branch and open GitHub PR.

---

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
- [x] Day 2: Qdrant vector store with dense search and version payload filters (BM25 sparse leg was wired in during Day 4 Fix 3; Day 2 originally ran dense-only)
- [x] Day 3: Baseline LangGraph state machine (`Planner` -> `Retriever` -> `AnswerWriter` -> `L1Verifier`)
- [x] Day 4: Abstention mechanism and guardrails v1: input guard, retrieval keyword/score guard, per-citation value-binding verifier (L1.5), fail-closed fallback, standardized abstention routing. Tuning set: 0 wrong answers / 0 wrong refusals (optimistic, tuned on). Fresh held-out set: 4/15 wrong refusals and 1/9 wrong answers before fixes. Known weak spots in docs/known-limitations.
- [x] Day 5: PostgreSQL checkpointer, step ledger, and claim-evidence audit trail tables
- [x] Day 6: FastMCP `corpus-mcp` read-only server
- [x] Day 7: Benchmark v1 with deterministic scoring & baseline measurements recorded in `experiments.md` (Tier 1 Certified)

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
