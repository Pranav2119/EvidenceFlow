# Engineering Learning Log

## Session 1: Architecture Specification & Foundational Design
**Date**: 2026-10-24  
**Focus**: Project conceptualization, USP definition, architectural defenses, repo layout, and core skeletons.

---

### 1. Concepts Mastered
- **Version Bleed in RAG**: When technical documentation updates across major versions (e.g., PostgreSQL 14 vs. 16), semantic search fails because phrasing is almost identical, but defaults or flags have changed. A version-aware system must enforce hard metadata filters during vector traversal.
- **Single-Stage Vector Filtering**: Why Postgres pgvector struggles with combined `GIN + HNSW + WHERE version = '16'` queries (cost-estimator index scan degradation) versus Qdrant's single-stage filtered HNSW traversal.
- **Deterministic-First AI Architecture**: LLMs should only handle semantic reasoning (planning, cross-version synthesis, contradiction analysis). Cache gates, citation verifiers, and guardrails must be deterministic plain code to ensure reliability and 0 token waste.
- **Composite Cache Scope Keys**: How HMAC-SHA256 of `(Intent || VersionScope || ConfigHash || NormalizedQuery)` prevents cross-version cache collisions.

---

### 2. Architectural Trade-offs & Decisions Made
- **Pydantic v2 `BaseModel` vs. `TypedDict` for LangGraph State**:
  - *Decision*: Adopted Pydantic v2 `BaseModel`.
  - *Trade-off*: Slightly more boilerplate than standard dicts, but provides runtime type coercion, automatic validation of citation structures, default factory initialization, and zero-headache JSON serialization for PostgreSQL checkpoints and audit trails.
- **Three-Store Database Strategy**:
  - *PostgreSQL*: ACID ledger, state checkpointer, cache metadata.
  - *Qdrant*: High-performance hybrid vector + lexical search with hard payload filters.
  - *Neo4j*: Relational entity/claim/version conversational memory graph.

---

### 3. Session Progress
- Completed Stages 1–3 design specifications.

---

## Session 2: Implementation Roadmap, Evaluation Framework, & Portfolio Packaging
**Date**: 2026-10-24  
**Focus**: Day-wise 19-day curriculum (Stage 4), 3-level evaluation plan & ablation design (Stage 5), portfolio packaging & ADRs (Stage 6), and resume bullet templates with plain-English analogies (Stage 7).

---

### 1. Concepts Mastered
- **Non-Inferiority Testing in AI Optimization**: Proving mathematically (using 95% confidence intervals and a $\Delta = 2\%$ margin) that a performance optimization like caching or prompt compression does not degrade answer groundedness.
  - *Analogy*: Proving that a generic medication is just as effective as an expensive brand-name drug before prescribing it.
- **Objective Ground Truth from Software Release Notes**: Why synthesizing benchmark datasets with LLMs is flawed, and why using official documentation release notes gives unassailable, deterministic ground truth for changed defaults and parameters.
- **Tiered Evaluation ("Checked Thrice")**: Running fast, deterministic L1 verification on 100% of queries, invoking an independent L2 Judge only on contested claims (5-10%), and enforcing an automated CI regression gate (L3) on pull requests.
- **Zero-LLM Cache Gatekeeper**: Validating cache hits via HMAC-SHA256 scope keys and live database chunk hash lookups in < 3ms with 0 tokens consumed.

---

### 2. Architectural Trade-offs & Decisions Made
- **ADR-001 (Pydantic v2 `BaseModel`)**: Prioritized runtime type coercion and frictionless JSON serialization for state checkpointing over minimal syntax.
- **ADR-002 (3-Store Database Architecture)**: Decoupled PostgreSQL (ACID audit ledger), Qdrant (single-stage filtered vector traversal), and Neo4j (conversational entity memory) to prevent `pgvector` index-scan degradation.
- **ADR-003 (Deterministic Cache Gatekeeper)**: Enforced plain-code validation instead of LLM validation, guaranteeing 0 token consumption on cache hits.
- **ADR-004 (FastMCP Tool Interface)**: Adopted open Model Context Protocol to decouple agent workflows from underlying storage tools.
- **ADR-005 (Selective L2 Judge Activation)**: Limited expensive judge model calls to contested claims, avoiding a 300% latency/cost explosion.

---

### 3. Open Questions / Next Session's Starting Point
- **Next Task**: Begin **Tier 1 (Measure First) - Day 0: Tooling, Repositories, Environment, & Container Infrastructure**.
- **Deliverables for Day 0**:
  - `docker-compose.yml` for PostgreSQL, Qdrant, and Neo4j.
  - `.env.example` and Pydantic `config.py` environment settings validation.
  - Port connectivity verification test script.
- **All Design Specifications Complete**:
  - [01_concept_and_usp.md](file:///C:/Users/Pranav%20Sakhalkar/Desktop/GitHub_AI/EvidenceFlowSetup/EvidenceFlow/docs/design/01_concept_and_usp.md)
  - [02_architecture.md](file:///C:/Users/Pranav%20Sakhalkar/Desktop/GitHub_AI/EvidenceFlowSetup/EvidenceFlow/docs/design/02_architecture.md)
  - [03_repo_structure_and_skeletons.md](file:///C:/Users/Pranav%20Sakhalkar/Desktop/GitHub_AI/EvidenceFlowSetup/EvidenceFlow/docs/design/03_repo_structure_and_skeletons.md)
  - [04_daywise_roadmap.md](file:///C:/Users/Pranav%20Sakhalkar/Desktop/GitHub_AI/EvidenceFlowSetup/EvidenceFlow/docs/design/04_daywise_roadmap.md)
  - [05_evaluation_plan.md](file:///C:/Users/Pranav%20Sakhalkar/Desktop/GitHub_AI/EvidenceFlowSetup/EvidenceFlow/docs/design/05_evaluation_plan.md)
  - [06_portfolio_packaging.md](file:///C:/Users/Pranav%20Sakhalkar/Desktop/GitHub_AI/EvidenceFlowSetup/EvidenceFlow/docs/design/06_portfolio_packaging.md)
  - [07_resume_and_ats.md](file:///C:/Users/Pranav%20Sakhalkar/Desktop/GitHub_AI/EvidenceFlowSetup/EvidenceFlow/docs/design/07_resume_and_ats.md)
