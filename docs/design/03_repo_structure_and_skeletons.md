# Stage 3: Repository Structure & Core Skeletons

This document outlines the file-level repository structure—with a single-sentence purpose for every file—followed by concrete, production-ready skeleton implementations for:
1. **LangGraph State Schema** (`state.py`)
2. **LangGraph Graph Wiring** (`workflow.py`)
3. **Scoped Cache Key Generator & Deterministic Validator** (`key_generator.py` & `validator.py`)
4. **FastMCP Corpus Server** (`corpus_server.py`)

---

## 1. Repository Structure

```text
EvidenceFlow/
├── .env.example                     # Template defining required API keys and database connection strings.
├── .gitignore                       # Prevents committing secrets, virtual environments, and local build artifacts.
├── pyproject.toml                   # Pinned dependency definitions, build system configuration, and project metadata.
├── docker-compose.yml               # Orchestrates local PostgreSQL, Qdrant, and Neo4j container instances.
├── README.md                        # Portfolio landing page displaying project architecture and empirical benchmark results.
├── docs/
│   ├── ROADMAP.md                   # Tier-by-tier sprint checklist and daily progress tracking.
│   ├── experiments.md               # Empirical benchmark logs, latency measurements, and ablation outputs.
│   ├── learning-log.md              # Daily engineering journal capturing conceptual breakthroughs and technical trade-offs.
│   └── design/
│       ├── 01_concept_and_usp.md    # Problem statement, architectural justifications, and interview defenses.
│       ├── 02_architecture.md       # End-to-end conversation flows, Neo4j schema, and cache layer designs.
│       └── 03_repo_structure_and_skeletons.md # Repository file layout and foundational component skeletons.
├── src/
│   └── evidenceflow/
│       ├── __init__.py              # Package initialization exporting public API interfaces.
│       ├── config.py                # Pydantic settings loading environment variables with strict type validation.
│       ├── state.py                 # LangGraph TypedDict state schema carrying conversation context across nodes.
│       ├── graph/
│       │   ├── __init__.py          # Graph module initialization exposing the compiled execution workflow.
│       │   ├── workflow.py          # StateGraph assembly, node binding, and conditional routing logic.
│       │   └── nodes/
│       │       ├── __init__.py      # Node package exports.
│       │       ├── cache_gatekeeper.py # Plain-code node performing zero-LLM cache checks and validation.
│       │       ├── planner.py       # LLM node decomposing queries and resolving multi-turn entity references.
│       │       ├── retriever.py     # FastMCP-connected node executing version-filtered hybrid Qdrant retrieval.
│       │       ├── contradiction.py # LLM node identifying superseded defaults across documentation versions.
│       │       ├── answer_writer.py # LLM node generating strictly cited answers from verified passages.
│       │       ├── verifier.py      # Plain-code node verifying citation quote-containment before release.
│       │       └── abstention.py    # Deterministic node creating structured abstention outputs on failure.
│       ├── cache/
│       │   ├── __init__.py          # Cache package exports.
│       │   ├── key_generator.py     # Deterministic SHA-256 HMAC composite scope-key generator.
│       │   ├── validator.py         # Plain-code validation of passage hashes against live database records.
│       │   └── manager.py           # Central coordinator dispatching across the 5 tiered cache layers.
│       ├── memory/
│       │   ├── __init__.py          # Memory package exports.
│       │   ├── neo4j_client.py      # Cypher query interface for entity, version, and claim graph traversal.
│       │   └── curator.py           # Background worker extracting entities and updating graph memory off the hot path.
│       ├── storage/
│       │   ├── __init__.py          # Storage package exports.
│       │   ├── postgres.py          # SQLAlchemy models for the step ledger, checkpointer, and audit trail.
│       │   └── qdrant_client.py     # Client wrapper configuring dense+sparse collections with payload filters.
│       ├── guardrails/
│       │   ├── __init__.py          # Guardrails package exports.
│       │   ├── input_guard.py       # Plain-code regex and token safety checks rejecting injections and PII.
│       │   ├── retrieval_guard.py   # Sanitizer stripping system-prompt override attempts from retrieved text.
│       │   └── output_guard.py      # Deterministic scanner checking for ungrounded claims or credential leaks.
│       ├── mcp/
│       │   ├── __init__.py          # MCP package exports.
│       │   ├── corpus_server.py     # FastMCP server exposing read-only tools to search and diff documentation.
│       │   ├── memory_cache_server.py # FastMCP server isolating read/write access to Neo4j and cache stores.
│       │   └── external_api_server.py # FastMCP wrapper providing rate-limited access to external technical APIs.
│       └── evaluation/
│           ├── __init__.py          # Evaluation package exports.
│           ├── metrics.py           # Component-level scoring functions for Recall@k, MRR, and groundedness.
│           ├── non_inferiority.py   # Statistical margin testing comparing ablation runs at 95% confidence.
│           └── runner.py            # CLI script running automated evaluation against version-release benchmarks.
├── tests/
│   ├── __init__.py                  # Test suite root.
│   ├── test_cache_collisions.py     # Adversarial test ensuring identical queries for different versions never collide.
│   ├── test_staleness.py            # Verification that document hash changes force immediate cache evictions.
│   ├── test_verifier.py             # Unit tests checking deterministic detection of fabricated or partial citations.
│   └── test_guardrails.py           # Security tests verifying prompt injection payloads are blocked at input.
└── data/
    ├── corpus/                      # Version-segmented Markdown files parsed from official release notes and docs.
    └── benchmarks/                  # Golden QA pairs strictly split into dev, validation, and untouched test sets.
```

---

## 2. Skeleton: LangGraph State Schema with Pydantic (`src/evidenceflow/state.py`)

Using **Pydantic v2 `BaseModel`** over `TypedDict` provides strict runtime type validation, automatic serialization/deserialization for the PostgreSQL checkpointer, and safe default values.

```python
"""
state.py: Pydantic v2 schema defining the LangGraph execution state.
Enforces strict runtime validation on citations, version scopes, and audit traces.
"""

from typing import Annotated, Any, Dict, List, Optional
from pydantic import BaseModel, Field
import operator


class PassageChunk(BaseModel):
    passage_id: str
    doc_path: str
    version: str
    content: str
    chunk_hash: str
    score: float = Field(default=0.0, ge=0.0, le=1.0)


class VerifiedCitation(BaseModel):
    passage_id: str
    exact_quote: str
    contained_verbatim: bool = False


class StepLedgerEntry(BaseModel):
    step_name: str
    started_at: str
    completed_at: str
    duration_ms: float
    llm_tokens_used: int = 0
    cache_status: str = "BYPASS"  # "HIT" | "MISS" | "BYPASS"


class EvidenceFlowState(BaseModel):
    # Session & Query Identifiers
    session_id: str
    raw_query: str
    normalized_query: str = ""
    target_versions: List[str] = Field(default_factory=list)  # e.g., ["14", "16"]
    
    # Context & Planning
    resolved_entity: Optional[str] = None  # e.g., "vacuum_cost_limit"
    decomposed_subqueries: List[str] = Field(default_factory=list)
    is_multi_version_comparison: bool = False
    
    # Evidence & Retrieval
    retrieved_passages: List[PassageChunk] = Field(default_factory=list)
    cross_version_conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Verification & Generation
    draft_answer: Optional[str] = None
    citations: List[VerifiedCitation] = Field(default_factory=list)
    verification_passed: bool = False
    
    # Abstention Protocol
    should_abstain: bool = False
    abstention_reason: Optional[str] = None  # "INSUFFICIENT_EVIDENCE", "UNVERIFIED_CITATION", etc.
    final_output: Optional[Dict[str, Any]] = None
    
    # Audit Trail & Performance (Appended immutably across nodes using operator.add)
    step_ledger: Annotated[List[StepLedgerEntry], operator.add] = Field(default_factory=list)
```

---

## 3. Skeleton: LangGraph Graph Wiring (`src/evidenceflow/graph/workflow.py`)

```python
"""
workflow.py: Graph definition assembling nodes and deterministic routing edges.
"""

from langgraph.graph import StateGraph, END
from evidenceflow.state import EvidenceFlowState

# Import node handlers
from evidenceflow.graph.nodes.cache_gatekeeper import cache_gatekeeper_node
from evidenceflow.graph.nodes.planner import planner_node
from evidenceflow.graph.nodes.retriever import retriever_node
from evidenceflow.graph.nodes.contradiction import contradiction_analyst_node
from evidenceflow.graph.nodes.answer_writer import answer_writer_node
from evidenceflow.graph.nodes.verifier import l1_verifier_node
from evidenceflow.graph.nodes.abstention import abstention_node


def route_after_cache(state: EvidenceFlowState) -> str:
    """If cache hit succeeded, finish immediately with 0 LLM calls."""
    if state.get("final_output") is not None and not state.get("should_abstain"):
        return END
    return "planner"


def route_after_retrieval(state: EvidenceFlowState) -> str:
    """Abstain immediately if no evidence meets similarity thresholds."""
    if not state.get("retrieved_passages"):
        return "abstention"
    return "contradiction_analyst"


def route_after_verification(state: EvidenceFlowState) -> str:
    """If deterministic quote containment fails, divert to abstention."""
    if not state.get("verification_passed"):
        return "abstention"
    return END


def create_evidence_flow_graph() -> StateGraph:
    workflow = StateGraph(EvidenceFlowState)
    
    # Register Nodes
    workflow.add_node("cache_gatekeeper", cache_gatekeeper_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("contradiction_analyst", contradiction_analyst_node)
    workflow.add_node("answer_writer", answer_writer_node)
    workflow.add_node("verifier", l1_verifier_node)
    workflow.add_node("abstention", abstention_node)
    
    # Set Entry Point
    workflow.set_entry_point("cache_gatekeeper")
    
    # Conditional Edges
    workflow.add_conditional_edges(
        "cache_gatekeeper",
        route_after_cache,
        {"planner": "planner", END: END}
    )
    
    workflow.add_edge("planner", "retriever")
    
    workflow.add_conditional_edges(
        "retriever",
        route_after_retrieval,
        {"contradiction_analyst": "contradiction_analyst", "abstention": "abstention"}
    )
    
    workflow.add_edge("contradiction_analyst", "answer_writer")
    workflow.add_edge("answer_writer", "verifier")
    
    workflow.add_conditional_edges(
        "verifier",
        route_after_verification,
        {END: END, "abstention": "abstention"}
    )
    
    workflow.add_edge("abstention", END)
    
    return workflow
```

---

## 4. Skeleton: Scoped Cache Key Generator & Validator (`src/evidenceflow/cache/`)

### `key_generator.py`
```python
"""
key_generator.py: Generates composite HMAC-SHA256 scope keys.
Prevents cross-version cache collisions.
"""

import hashlib
import hmac
import re
from typing import List


def normalize_query(query: str) -> str:
    """Strip punctuation and normalize whitespace for deterministic hashing."""
    cleaned = re.sub(r"[^\w\s]", "", query.lower())
    return " ".join(cleaned.split())


def generate_scope_key(
    query: str,
    target_versions: List[str],
    config_version: str,
    guardrail_version: str,
    secret_salt: str = "evidenceflow-salt"
) -> str:
    """
    Computes a deterministic composite key.
    Any variation in target version or security config creates an entirely distinct key.
    """
    norm_q = normalize_query(query)
    sorted_versions = ",".join(sorted(target_versions))
    
    payload = f"V:{sorted_versions}|CFG:{config_version}|GD:{guardrail_version}|Q:{norm_q}"
    
    return hmac.new(
        secret_salt.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
```

### `validator.py`
```python
"""
validator.py: Zero-LLM deterministic validation for cache hits.
Verifies that cited passage hashes match the live PostgreSQL corpus snapshot.
"""

from typing import List, Dict, Any, Protocol


class CorpusDatabase(Protocol):
    def get_chunk_hash(self, passage_id: str) -> str | None:
        """Fetch current SHA-256 hash of a passage from PostgreSQL."""
        ...


def validate_cache_hit(
    cached_citations: List[Dict[str, str]],
    db: CorpusDatabase
) -> bool:
    """
    Returns True ONLY if all cited passages still exist and have unchanged content hashes.
    If a document was re-indexed or patched, validation fails and forces a cache miss.
    """
    if not cached_citations:
        return False
        
    for item in cached_citations:
        passage_id = item["passage_id"]
        cached_hash = item["chunk_hash"]
        
        live_hash = db.get_chunk_hash(passage_id)
        if live_hash is None or live_hash != cached_hash:
            return False  # Stale cache detected!
            
    return True
```

---

## 5. Skeleton: FastMCP Corpus Server (`src/evidenceflow/mcp/corpus_server.py`)

```python
"""
corpus_server.py: FastMCP server providing read-only tools over the versioned corpus.
Enforces version payload filters and exposes semantic diff tools.
"""

from fastmcp import FastMCP
from typing import List, Dict, Any

mcp = FastMCP("EvidenceFlow-Corpus-MCP")


@mcp.tool()
def list_versions() -> List[str]:
    """List all documentation versions currently indexed in the corpus."""
    return ["14", "15", "16", "17"]


@mcp.tool()
def search_passages(query: str, version: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search documentation chunks with a mandatory version filter.
    Executes hybrid dense + BM25 search via Qdrant.
    """
    # TODO: Connect to src.evidenceflow.storage.qdrant_client
    return [
        {
            "passage_id": f"pg_{version}_sample_01",
            "version": version,
            "section": "Server Configuration / Resource Usage",
            "content": f"Sample excerpt from PostgreSQL {version} documentation for query: {query}",
            "chunk_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        }
    ]


@mcp.tool()
def get_passage(passage_id: str) -> Dict[str, Any]:
    """Retrieve the full text and metadata for a specific passage chunk."""
    # TODO: Fetch from storage
    return {
        "passage_id": passage_id,
        "content": "Full section text...",
        "chunk_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    }


@mcp.tool()
def diff_sections(doc_path: str, version_a: str, version_b: str) -> Dict[str, Any]:
    """
    Compare a documentation section between two versions to detect changed defaults.
    """
    # TODO: Fetch both passages and return a structured diff
    return {
        "doc_path": doc_path,
        "version_a": version_a,
        "version_b": version_b,
        "has_changes": True,
        "diff_summary": "Default value changed from 200 to 1000."
    }


if __name__ == "__main__":
    mcp.run()
```
