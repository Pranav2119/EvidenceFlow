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

> Checked historical milestones record implementation progress, not current certification of every capability or benchmark claim. The correctness review below takes precedence for readiness.

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
- [x] Day 8: Neo4j conversational graph memory (`:Turn`, `:Question`, `:Entity`, `:Claim`, `:Version`), discrete ACID transactions, explicit setting precedence, ambiguity detection, and non-inferiority evaluation against single-turn dev benchmark
- [x] Day 9: 5-Layer scoped cache implementation with composite HMAC keys (L1 Exact Match, L2 Semantic, L3 Retrieval, L4 Embedding, L5 Verified-Claim)
- [x] Day 10: Zero-LLM deterministic cache hit validator & automatic stale-chunk eviction
- [x] Day 11: FastMCP `memory-cache-mcp` server with role-based write isolation
- [x] Day 12: Cache collision trap tests & document patch staleness tests
- [x] Day 13: Tier 2 Ablation study: Baseline vs. +Memory vs. +Cache vs. +Both (Non-inferiority analysis)

### Tier 3: Stand Out (Cross-Version Reasoning & Advanced Evaluation)

- [x] Day 14: Contradiction Analyst agent with `SUPERSEDES` and `CONTRADICTS` graph edges
- [x] Day 15: Independent L2 Judge for contested claims (selective activation via ADR-005)
- [x] Day 16: Benchmark comparison: EvidenceFlow vs. pgvector + full-text vs. recursive SQL
- [x] Day 17: External API FastMCP server with rate-limiting & schema validation
- [x] Day 18: CI regression test gate (preventing recall/groundedness regressions) & release packaging

### CI Reproducibility Follow-up (2026-11-23)
- [x] Configure image-compatible health checks, frozen dependency installation, committed-corpus indexing, and Neo4j schema initialization before CI tests.
- [x] Validate the frozen Windows environment and collect all 248 tests without changing application logic or dependency versions.
- [x] Verify container startup, dependency installation, and PostgreSQL migrations in GitHub Actions; add bounded model preparation and caching after observing Hugging Face download rate limits.
- [x] Complete a passing Docker-backed CI run: [run 45172813930](https://github.com/Pranav2119/EvidenceFlow/actions/runs/45172813930), commit `aa83738`, passed 249 tests and all configured metric gates, with one warning. No tests or quality thresholds were weakened.

### Interview-Readiness Correctness Review (2026-11-23)
- [x] Repair the equal-defaults/changed-options comparison and verify seven focused regression cases locally.
- [x] Verify the comparison repair in the full GitHub CI suite, keeping all integration tests and metric gates enabled.
- [x] Repair evaluator value matching, scalar/list scoring, uncached baseline controls, and unsupported statistical-success reporting; verify 41 focused local tests.
- [x] Publish the evaluator repair as `cb4c996` and verify [CI run 45192610092](https://github.com/Pranav2119/EvidenceFlow/actions/runs/45192610092): 275 tests and all configured metric gates passed.
- [x] Prepare development-only CI capture with per-repeat evidence records, observed-cache-hit rejection, honest metric labels, and failure-safe temporary settings; verify 47 focused local tests.
- [ ] Publish and verify remaining repair batches in full CI after approval.
- [x] Record the first fresh offline development baseline before RAG behavior changes: [run 45217948535](https://github.com/Pranav2119/EvidenceFlow/actions/runs/45217948535), 40 cases / 120 timed observations, with the original JSON retained and 281 full-suite tests passing.
- [x] Audit all 40 development labels against indexed evidence/scope; revise to v1.1 with 24 supported answers and 16 refusals, preserving queries and the v1.0 baseline. Record exact support quotes, qualified defaults, and the label-only rescore; 50 focused tests pass locally.
- [x] Verify the v1.1 label/provenance patch in [CI run 45648118532](https://github.com/Pranav2119/EvidenceFlow/actions/runs/45648118532): 284 tests and all metric gates passed; the artifact's dataset fingerprint and all 120 observations were checked. Label corrections do not establish application improvement.
- [x] Reproduce and repair deterministic fallback's overlapping-name definition bonus; 14 focused tests pass, and saved-passage replays verify `vacuum_cost_limit=200` without changing the verifier.
- [x] Verify the fallback repair in [CI run 45651481101](https://github.com/Pranav2119/EvidenceFlow/actions/runs/45651481101): 292 tests and all metric gates passed. Same-v1.1 comparison fixes only `d01`/`d02`; all retrieval rankings and other 38 case outputs are unchanged. Decision agreement is 40/40 on this development set, not universal factual accuracy.
- [x] Repair lossy L1/L3/L4 key identity with approved literal-text semantics and v2 namespaces. Verify 67 focused local checks, including meaningful-query collisions, legacy-key isolation, staleness, and exact-hit routing without fresh pipeline work.
- [x] Verify key repair `4facf10` in [CI run 45660978685](https://github.com/Pranav2119/EvidenceFlow/actions/runs/45660978685): 318 tests and all gates passed. All 120 uncached development observations retain their answers/evidence/verdicts; missing-hash and MCP validation remain separate open findings.
- [x] Repair missing/invalid hash acceptance and MCP validation bypasses; check live evidence and existing guard/verifier results before MCP writes, reject/evict bad reads, and log rejection status. 103 focused local tests pass.
- [x] Verify evidence/MCP repair `97fb050` in [CI run 45809498665](https://github.com/Pranav2119/EvidenceFlow/actions/runs/45809498665): 345 tests and all gates passed; all 120 uncached development answers/evidence/verdicts remain unchanged. Caller authentication, deeper verifier gaps, and semantic-cache safety remain separate findings.
- [ ] Resolve remaining evaluation-methodology limitations before resume quality/performance claims; establish a separately controlled Gemini-enabled baseline before prompt or generation changes.
- [ ] Resolve the remaining critical citation, cache, version-state, and evaluation findings before claiming overall interview readiness.
