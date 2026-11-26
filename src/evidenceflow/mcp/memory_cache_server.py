"""
memory_cache_server.py: FastMCP tools for trusted-local memory and cache operations.
Caller-supplied roles are routing restrictions, not authenticated identities:
  - Cache Lookups: Allowed for cache_gatekeeper, planner, orchestrator, admin.
  - Cache Writes: RESTRICTED to cache_gatekeeper and admin.
  - Memory Graph Queries: Allowed for planner, memory_resolver, orchestrator, analyst, admin.
  - Memory Graph Writes: RESTRICTED to memory_curator and admin.

All server diagnostics log to sys.stderr to protect stdio JSON-RPC protocol framing.
Tool executions log authorization decisions and latency to the audit step ledger.
"""

import functools
import logging
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import Field, ValidationError
from mcp.server.fastmcp import FastMCP

from evidenceflow.cache.manager import ScopedCacheManager
from evidenceflow.cache.models import CachedCitation
from evidenceflow.cache.validator import validate_cache_hit, evict_stale_cache_entry
from evidenceflow.graph.nodes.cache_gatekeeper import get_cache_manager, get_corpus_database
from evidenceflow.graph.nodes.answer_writer import extract_citations
from evidenceflow.graph.nodes.verifier import verifier_node
from evidenceflow.guardrails.input_guardrail import INJECTION_PATTERNS, input_guardrail_node
from evidenceflow.memory.graph_client import (
    get_session_history,
    record_turn as neo4j_record_turn,
)
from evidenceflow.storage.ledger import record_step, get_next_step_number
from evidenceflow.storage.qdrant_client import get_evidence_qdrant_client
from evidenceflow.state import EvidenceFlowState

# Configure logging exclusively to sys.stderr to avoid corrupting stdio transport
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [memory-cache-mcp] %(message)s",
)
logger = logging.getLogger("memory-cache-mcp")

# FastMCP Server Instance
mcp = FastMCP("memory-cache-mcp")

# Role Permission Definitions
ALLOWED_CACHE_READ_ROLES: Set[str] = {"cache_gatekeeper", "planner", "orchestrator", "admin"}
ALLOWED_CACHE_WRITE_ROLES: Set[str] = {"cache_gatekeeper", "admin"}
ALLOWED_MEMORY_READ_ROLES: Set[str] = {"planner", "memory_resolver", "orchestrator", "analyst", "admin"}
ALLOWED_MEMORY_WRITE_ROLES: Set[str] = {"memory_curator", "admin"}


_db_available: Optional[bool] = None
_db_last_checked: float = 0.0


def _is_db_reachable() -> bool:
    """Fast circuit breaker: checks if PostgreSQL port is active with 50ms socket probe."""
    global _db_available, _db_last_checked
    now = time.time()
    if _db_available is not None and (now - _db_last_checked) < 15.0:
        return _db_available

    import socket
    from evidenceflow.config import get_settings
    s = get_settings()
    try:
        with socket.create_connection((s.POSTGRES_HOST, s.POSTGRES_PORT), timeout=0.05):
            _db_available = True
    except Exception:
        _db_available = False
    _db_last_checked = now
    return _db_available


def _log_audit_step(
    caller_role: str,
    operation: str,
    status: str,
    thread_id: str = "mcp_thread",
    latency_ms: float = 0.0,
    details: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """Best-effort recording of MCP tool invocations to PostgreSQL step_ledger."""
    if not _is_db_reachable():
        return
    try:
        step_details = {
            "mcp_server": "memory-cache-mcp",
            "operation": operation,
            "caller_role": caller_role,
        }
        if details:
            step_details.update(details)
        if error:
            step_details["error"] = error

        step_num = 1
        try:
            step_num = get_next_step_number(thread_id)
        except Exception:
            pass

        record_step(
            turn_id=str(uuid.uuid4()),
            thread_id=thread_id,
            step_number=step_num,
            node="memory_cache_mcp",
            action=operation,
            status=status,
            decided_by="rule",
            latency_ms=latency_ms,
            details=step_details,
        )
    except Exception as e:
        logger.debug("Non-critical: Audit ledger step log skipped: %s", e)


def require_role(allowed_roles: Set[str], operation: str):
    """
    Decorator enforcing role-based tool execution boundaries.
    Rejects unauthorized caller roles with a structured PermissionDenied error.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            import inspect
            caller_role = None
            try:
                sig = inspect.signature(func)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                caller_role = bound.arguments.get("caller_role")
            except Exception:
                caller_role = kwargs.get("caller_role")

            role_str = (caller_role or "").strip().lower()
            if role_str not in allowed_roles:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                err_msg = f"PermissionDenied: {caller_role or 'anonymous'} cannot {operation}"
                logger.warning("Access Denied: role '%s' attempted to %s", caller_role, operation)
                _log_audit_step(
                    caller_role=caller_role or "anonymous",
                    operation=operation,
                    status="permission_denied",
                    latency_ms=elapsed_ms,
                    error=err_msg,
                )
                return {
                    "status": "permission_denied",
                    "error": err_msg,
                }

            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            _log_audit_step(
                caller_role=role_str,
                operation=operation,
                status=result["status"],
                latency_ms=elapsed_ms,
                error=result.get("error"),
            )
            return result
        return wrapper
    return decorator


# =============================================================================
# Cache FastMCP Tools (Isolated Read & Write)
# =============================================================================

def _cache_rejection(reason: str) -> Dict[str, Any]:
    logger.warning("Cache operation rejected: %s", reason)
    return {"status": "rejected", "error": reason}


@mcp.tool()
@require_role(ALLOWED_CACHE_READ_ROLES, "read cache")
def lookup_cache(
    query: str,
    version: str = "16",
    caller_role: str = "cache_gatekeeper",
    caller_id: str = "default_caller",
) -> Dict[str, Any]:
    """
    Look up an exact cached response and validate its live evidence hashes.
    Allowed roles: cache_gatekeeper, planner, orchestrator, admin.
    """
    started = time.perf_counter()
    cm = get_cache_manager()
    res = cm.lookup(query=query, target_versions=version)
    if res.hit:
        if not res.answer or not res.answer.strip():
            valid, reason, stale = False, "INVALID_ANSWER: cached answer is empty", []
        else:
            valid, reason, stale = validate_cache_hit(
                res.citations,
                get_corpus_database(),
                current_config_version=cm.default_config_version,
                current_guardrail_version=cm.default_guardrail_version,
            )
        if not valid:
            assert reason is not None
            evict_stale_cache_entry(cm, stale, cache_key=res.metadata.get("key"))
            return {
                **_cache_rejection(reason),
                "hit": False,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            }
        return {
            "status": "hit",
            "hit": True,
            "layer": res.layer.value if res.layer else "L1_EXACT",
            "answer": res.answer,
            "citations": [c.model_dump() for c in res.citations],
            "similarity_score": res.similarity_score,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }
    return {
        "status": "miss",
        "hit": False,
        "latency_ms": (time.perf_counter() - started) * 1000.0,
    }


@mcp.tool()
@require_role(ALLOWED_CACHE_WRITE_ROLES, "write to cache")
def write_cache(
    query: str,
    answer: Any = "",
    citations: Optional[List[Dict[str, Any]]] = None,
    version: str = "16",
    caller_role: str = "cache_gatekeeper",
    caller_id: str = "default_caller",
) -> Dict[str, Any]:
    """
    Validate input, live versioned evidence, and deterministic answer checks before L1 storage.
    RESTRICTED ROLE: Only cache_gatekeeper and admin may commit to cache.
    Attempts from retriever, answer_writer, or planner are strictly rejected.
    """
    cm = get_cache_manager()

    # Handle dictionary passed as answer or payload (convenience pattern)
    ans_text = ""
    cits_list = citations or []
    if isinstance(answer, dict):
        ans_text = answer.get("answer", "")
        cits_list = cits_list or answer.get("citations", [])
    elif isinstance(answer, str):
        ans_text = answer

    if not isinstance(ans_text, str) or not ans_text.strip():
        return _cache_rejection("INVALID_ANSWER: answer must be a nonblank string")
    if not isinstance(cits_list, list):
        return _cache_rejection("INVALID_CITATION: citations must be a list")
    if not isinstance(query, str) or not query.strip():
        return _cache_rejection("INVALID_QUERY: query must be a nonblank string")

    guard = input_guardrail_node(EvidenceFlowState(query=query, target_version=version))
    if guard["is_abstained"]:
        return _cache_rejection(f"{guard['abstention_code']}: {guard['abstention_reason']}")
    if any(pattern.search(ans_text) for pattern in INJECTION_PATTERNS):
        return _cache_rejection("INJECTION: answer contains a detected instruction override")

    parsed_citations: List[CachedCitation] = []
    for c in cits_list:
        if isinstance(c, CachedCitation):
            parsed_citations.append(c)
        elif isinstance(c, dict):
            pid = c.get("passage_id") or c.get("chunk_id", "")
            chash = c.get("chunk_hash") or c.get("sha256", "")
            try:
                parsed_citations.append(
                    CachedCitation(
                        passage_id=pid,
                        chunk_hash=chash,
                        source_url=c.get("source_url", ""),
                        title=c.get("title", ""),
                    )
                )
            except ValidationError:
                return _cache_rejection("INVALID_CITATION: citation fields must be strings")
        else:
            return _cache_rejection("INVALID_CITATION: every citation must be an object")

    valid, reason, _ = validate_cache_hit(parsed_citations, get_corpus_database())
    if not valid:
        assert reason is not None
        return _cache_rejection(reason)

    client = get_evidence_qdrant_client()
    passages: List[Dict[str, Any]] = []
    for citation in parsed_citations:
        passage = client.get_passage(citation.passage_id)
        if not passage or passage.get("status") != "success" or passage.get("chunk_id") != citation.passage_id:
            return _cache_rejection(f"PASSAGE_UNAVAILABLE: cannot inspect '{citation.passage_id}'")
        if passage.get("version") != version:
            return _cache_rejection(f"PASSAGE_VERSION_MISMATCH: '{citation.passage_id}' is outside version {version}")
        content = passage.get("content")
        if not isinstance(content, str) or not content.strip():
            return _cache_rejection(f"INVALID_PASSAGE: '{citation.passage_id}' lacks content")
        # Recheck the fetched snapshot in case it changed after the initial hash lookup.
        if passage.get("sha256") != citation.chunk_hash:
            return _cache_rejection(f"HASH_MISMATCH: '{citation.passage_id}' changed before write verification")
        if any(pattern.search(content) for pattern in INJECTION_PATTERNS):
            return _cache_rejection(f"INJECTION: detected instruction override in '{citation.passage_id}'")
        passages.append(passage)

    answer_citations = extract_citations(ans_text, passages)
    if {c.chunk_id for c in answer_citations} != {c.passage_id for c in parsed_citations}:
        return _cache_rejection("CITATION_MISMATCH: answer markers must match the supplied evidence IDs")
    verified = verifier_node(EvidenceFlowState(
        query=query,
        target_version=version,
        draft_answer=ans_text,
        citations=answer_citations,
        retrieved_passages=passages,
    ))
    if not verified["l1_passed"] or verified["is_abstained"]:
        return _cache_rejection(f"{verified['abstention_code']}: {verified['abstention_reason']}")

    entry = cm.set_exact(
        query=query,
        answer=ans_text,
        citations=parsed_citations,
        target_versions=[version],
    )

    return {
        "status": "success",
        "key": entry.key,
        "layer": "L1_EXACT",
        "cached_citations_count": len(parsed_citations),
    }


# =============================================================================
# Conversational Memory Graph FastMCP Tools (Isolated Read & Write)
# =============================================================================

@mcp.tool()
@require_role(ALLOWED_MEMORY_READ_ROLES, "query entity graph")
def query_entity_graph(
    thread_id: str,
    entity_name: Optional[str] = None,
    caller_role: str = "planner",
    caller_id: str = "default_caller",
) -> Dict[str, Any]:
    """
    Query the Neo4j conversational memory graph for conversation history and resolved entities.
    Allowed roles: planner, memory_resolver, orchestrator, analyst, admin.
    """
    try:
        history = get_session_history(thread_id=thread_id)
        return {
            "status": "success",
            "thread_id": thread_id,
            "turns_count": len(history),
            "history": history,
            "filtered_entity": entity_name,
        }
    except Exception as e:
        logger.warning("query_entity_graph error: %s", e)
        return {
            "status": "error",
            "thread_id": thread_id,
            "error": str(e),
            "history": [],
        }


@mcp.tool()
@require_role(ALLOWED_MEMORY_WRITE_ROLES, "write to memory")
def commit_turn_memory(
    turn_data: Dict[str, Any],
    caller_role: str = "memory_curator",
    caller_id: str = "default_caller",
) -> Dict[str, Any]:
    """
    Commit a conversation turn, resolved entities, and verified claims to Neo4j graph memory.
    RESTRICTED ROLE: Only memory_curator and admin may write to conversational memory.
    Attempts from retriever, answer_writer, or planner are strictly rejected.
    """
    turn_id = turn_data.get("turn_id") or str(uuid.uuid4())
    thread_id = turn_data.get("thread_id", "default_thread")
    turn_number = turn_data.get("turn_number", 1)
    question = turn_data.get("question", "")
    version_scope = turn_data.get("version_scope")
    entities = turn_data.get("entities", [])
    claims = turn_data.get("claims", [])
    is_abstained = turn_data.get("is_abstained", False)
    abstention_code = turn_data.get("abstention_code")
    abstention_reason = turn_data.get("abstention_reason")

    try:
        recorded_id = neo4j_record_turn(
            turn_id=turn_id,
            thread_id=thread_id,
            turn_number=turn_number,
            question=question,
            version_scope=version_scope,
            entities=entities,
            claims=claims,
            is_abstained=is_abstained,
            abstention_code=abstention_code,
            abstention_reason=abstention_reason,
        )
        return {
            "status": "success",
            "turn_id": recorded_id,
            "thread_id": thread_id,
            "committed_entities_count": len(entities),
            "committed_claims_count": len(claims),
        }
    except Exception as e:
        logger.warning("commit_turn_memory Neo4j error: %s", e)
        return {
            "status": "error",
            "turn_id": turn_id,
            "thread_id": thread_id,
            "error": str(e),
        }


if __name__ == "__main__":
    # Start FastMCP server on stdio transport
    logger.info("Starting memory-cache-mcp server on stdio transport...")
    mcp.run(transport="stdio")
