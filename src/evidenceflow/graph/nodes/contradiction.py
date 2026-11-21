"""
contradiction.py: Contradiction Analyst node for cross-version reasoning in EvidenceFlow AI.
Analyzes retrieved documentation passages across versions to identify:
  1. Evolutionary parameter changes (SUPERSEDES): Newer version changed default, added options, or deprecated setting.
  2. Direct factual conflicts (CONTRADICTS): Opposing assertions for the same release or un-scoped claims.
  3. Consistent evidence (NO_CONTRADICTION): Passages affirm identical values and behavior.

Records structural graph relationships (SUPERSEDES, CONTRADICTS) in Neo4j.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from evidenceflow.config import get_settings
from evidenceflow.memory.graph_client import record_passage_relationship
from evidenceflow.state import EvidenceFlowState

logger = logging.getLogger(__name__)

# Singleton Gemini client for contradiction analysis
_gemini_contradiction_client = None


def _get_contradiction_client(api_key: str):
    global _gemini_contradiction_client
    if _gemini_contradiction_client is None:
        from google import genai
        _gemini_contradiction_client = genai.Client(api_key=api_key)
    return _gemini_contradiction_client


def extract_parameter_values_from_passage(text: str) -> Dict[str, Any]:
    """
    Deterministically extracts parameter name, default value, and enum options from passage text.
    """
    clean = text.lower()
    # Check for parameter header: Parameter: <name> ( <type> )
    param_match = re.search(r"parameter:\s*([a-z0-9_]+)", clean)
    param_name = param_match.group(1) if param_match else None

    # Check for default value: default value is <val> | default is <val> | defaults to <val>
    default_match = re.search(r"\b(?:default\s+value\s+is|default\s+is|defaults\s+to)\s+([^\s,.;]+)", clean)
    default_val = default_match.group(1).rstrip(".") if default_match else None

    # Check for options/algorithms: e.g. "on, off, pg_lz, lz4, zstd"
    options = []
    opt_match = re.search(r"\b(?:accepts|values\s+are|can\s+be)\s+([^.\n]+)", clean)
    if opt_match:
        raw_opts = re.split(r"[,;]|\band\b|\bor\b", opt_match.group(1))
        options = [o.strip(" '\"`") for o in raw_opts if o.strip(" '\"`") and len(o.strip()) < 30]

    return {
        "parameter": param_name,
        "default": default_val,
        "options": options,
    }


def analyze_cross_version_differences_deterministic(
    passages_by_version: Dict[str, List[Dict[str, Any]]],
    query: str,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Deterministic rule-based cross-version comparison.
    Compares parameter defaults and options between versions.
    """
    sorted_versions = sorted(passages_by_version.keys(), key=lambda v: int(v) if v.isdigit() else 0)
    if len(sorted_versions) < 2:
        # Check intra-version contradictions
        for v, p_list in passages_by_version.items():
            if len(p_list) >= 2:
                v1_info = extract_parameter_values_from_passage(p_list[0].get("content", ""))
                v2_info = extract_parameter_values_from_passage(p_list[1].get("content", ""))
                if (
                    v1_info["default"]
                    and v2_info["default"]
                    and v1_info["default"] != v2_info["default"]
                ):
                    reason = f"Intra-version conflict in PostgreSQL {v}: passage '{p_list[0]['chunk_id']}' asserts default is '{v1_info['default']}' but '{p_list[1]['chunk_id']}' asserts '{v2_info['default']}'."
                    return True, "CONTRADICTS", {
                        "relationship_type": "CONTRADICTS",
                        "subject_version": v,
                        "target_version": v,
                        "subject_chunk_id": p_list[0]["chunk_id"],
                        "target_chunk_id": p_list[1]["chunk_id"],
                        "reason": reason,
                        "values_by_version": {p_list[0]["chunk_id"]: v1_info["default"], p_list[1]["chunk_id"]: v2_info["default"]},
                    }
        return False, "NO_CONTRADICTION", None

    v_older = sorted_versions[0]
    v_newer = sorted_versions[-1]
    p_older = passages_by_version[v_older][0]
    p_newer = passages_by_version[v_newer][0]

    info_older = extract_parameter_values_from_passage(p_older.get("content", ""))
    info_newer = extract_parameter_values_from_passage(p_newer.get("content", ""))

    # Compare defaults
    def_older = info_older.get("default")
    def_newer = info_newer.get("default")

    if def_older and def_newer:
        if def_older != def_newer:
            reason = f"PostgreSQL {v_newer} supersedes PostgreSQL {v_older}: default value changed from '{def_older}' to '{def_newer}'."
            return True, "SUPERSEDES", {
                "relationship_type": "SUPERSEDES",
                "evolution_type": "changed_default",
                "subject_version": v_newer,
                "target_version": v_older,
                "subject_chunk_id": p_newer["chunk_id"],
                "target_chunk_id": p_older["chunk_id"],
                "reason": reason,
                "values_by_version": {v_older: def_older, v_newer: def_newer},
            }
        else:
            # Values are identical! NO CONTRADICTION
            reason = f"PostgreSQL {v_older} and PostgreSQL {v_newer} both specify default value of '{def_older}'."
            return False, "NO_CONTRADICTION", {
                "relationship_type": "NO_CONTRADICTION",
                "subject_version": v_newer,
                "target_version": v_older,
                "subject_chunk_id": p_newer["chunk_id"],
                "target_chunk_id": p_older["chunk_id"],
                "reason": reason,
                "values_by_version": {v_older: def_older, v_newer: def_newer},
            }

    # Compare options/algorithms
    opts_older = set(info_older.get("options") or [])
    opts_newer = set(info_newer.get("options") or [])
    if opts_older and opts_newer and opts_older != opts_newer:
        added = opts_newer - opts_older
        removed = opts_older - opts_newer
        diff_desc = []
        if added:
            diff_desc.append(f"added {list(added)}")
        if removed:
            diff_desc.append(f"removed {list(removed)}")
        reason = f"PostgreSQL {v_newer} supersedes PostgreSQL {v_older}: {', '.join(diff_desc)}."
        return True, "SUPERSEDES", {
            "relationship_type": "SUPERSEDES",
            "evolution_type": "options_update",
            "subject_version": v_newer,
            "target_version": v_older,
            "subject_chunk_id": p_newer["chunk_id"],
            "target_chunk_id": p_older["chunk_id"],
            "reason": reason,
            "values_by_version": {v_older: list(opts_older), v_newer: list(opts_newer)},
        }

    # Identical or no explicit differences detected
    return False, "NO_CONTRADICTION", None


def analyze_with_llm(
    query: str,
    passages: List[Dict[str, Any]],
    api_key: str,
) -> Optional[Dict[str, Any]]:
    """
    Invokes Gemini with strict structured prompt to perform deep semantic contradiction/supersedes analysis.
    """
    client = _get_contradiction_client(api_key)
    passages_text = []
    for p in passages:
        passages_text.append(
            f"Passage ID: {p['chunk_id']} | Version: {p.get('version', 'unknown')}\n"
            f"Content: {p.get('content', '')}\n"
        )

    prompt = (
        "You are an expert technical documentation contradiction and evolution analyst for PostgreSQL.\n"
        "Your task is to analyze the provided documentation passages and the user's question to determine whether:\n"
        "1. A newer version SUPERSEDES an older version (e.g. default parameter changed, algorithms added, parameter removed/deprecated).\n"
        "2. An outright CONTRADICTION exists (two passages give conflicting facts for the same version).\n"
        "3. NO CONTRADICTION exists (the passages describe consistent or identical behavior across versions).\n\n"
        f"User Query: {query}\n\n"
        "Passages:\n" + "\n---\n".join(passages_text) + "\n\n"
        "Respond strictly with a JSON object in this format:\n"
        "{\n"
        '  "contradiction_detected": true/false (set to true if SUPERSEDES or CONTRADICTS, false only if NO_CONTRADICTION),\n'
        '  "relationship_type": "SUPERSEDES" | "CONTRADICTS" | "NO_CONTRADICTION",\n'
        '  "subject_chunk_id": "<newer or conflicting chunk id>",\n'
        '  "target_chunk_id": "<older or contradicted chunk id>",\n'
        '  "subject_version": "<newer or subject version>",\n'
        '  "target_version": "<older or target version>",\n'
        '  "reason": "<clear explanation of change or conflict>",\n'
        '  "parameter": "<parameter name if applicable>",\n'
        '  "values_by_version": {"<version1>": "<value1>", "<version2>": "<value2>"}\n'
        "}\n"
    )

    try:
        settings = get_settings()
        model_name = getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash")
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        txt = resp.text.strip()
        txt = re.sub(r"^```(?:json)?\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt)
        data = json.loads(txt)
        # Normalize contradiction_detected flag based on relationship_type
        if data.get("relationship_type") in {"SUPERSEDES", "CONTRADICTS"}:
            data["contradiction_detected"] = True
        elif data.get("relationship_type") == "NO_CONTRADICTION":
            data["contradiction_detected"] = False
        return data
    except Exception as e:
        logger.warning("LLM contradiction analysis failed (%s), using deterministic fallback.", e)
        return None


def contradiction_analyst_node(state: EvidenceFlowState) -> Dict[str, Any]:
    """
    LangGraph node: Analyzes retrieved passages for cross-version evolution or factual contradiction.
    Executes between Retriever and AnswerWriter.
    """
    t0 = time.perf_counter()
    passages = state.retrieved_passages or []
    query = state.query or ""

    if not passages or len(passages) < 2:
        return {
            "contradiction_detected": False,
            "contradiction_details": None,
        }

    # Group passages by documentation version
    passages_by_version: Dict[str, List[Dict[str, Any]]] = {}
    for p in passages:
        v = str(p.get("version") or state.target_version or "16")
        passages_by_version.setdefault(v, []).append(p)

    has_multi_versions = len(passages_by_version) >= 2
    is_comparative_query = bool(
        re.search(r"\b(compare|change|changed|differ|difference|supersede|deprecated|same\s+in|did\s+it\s+change|versus|vs\.?)\b", query, re.IGNORECASE)
    )

    details = None
    contradiction_detected = False
    rel_type = "NO_CONTRADICTION"

    settings = get_settings()
    api_key = getattr(settings, "GEMINI_API_KEY", None)

    # If comparative query or multi-version retrieved evidence, run analysis
    if has_multi_versions or is_comparative_query:
        if api_key and len(api_key) > 15 and "placeholder" not in api_key.lower():
            llm_result = analyze_with_llm(query, passages, api_key)
            if llm_result:
                rel_type = llm_result.get("relationship_type", "NO_CONTRADICTION")
                contradiction_detected = rel_type in {"SUPERSEDES", "CONTRADICTS"} or bool(llm_result.get("contradiction_detected", False))
                details = llm_result

        # If LLM didn't produce result, use deterministic analysis
        if details is None:
            contradiction_detected, rel_type, details = analyze_cross_version_differences_deterministic(
                passages_by_version, query
            )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Record structural graph edge in Neo4j if a relationship was detected
    if contradiction_detected and details:
        subj_id = details.get("subject_chunk_id")
        tgt_id = details.get("target_chunk_id")
        reason = details.get("reason", "")
        if subj_id and tgt_id and rel_type in {"SUPERSEDES", "CONTRADICTS"}:
            try:
                record_passage_relationship(
                    subject_passage_id=subj_id,
                    target_passage_id=tgt_id,
                    rel_type=rel_type,
                    reason=reason,
                    metadata={
                        "query": query[:200],
                        "parameter": details.get("parameter", ""),
                    },
                )
            except Exception as e:
                logger.warning("Failed to record passage relationship in Neo4j: %s", e)

    audit_entry = {
        "node": "contradiction_analyst",
        "action": "contradiction_analyzed",
        "contradiction_detected": contradiction_detected,
        "relationship_type": rel_type,
        "latency_ms": elapsed_ms,
        "details": details,
    }

    return {
        "contradiction_detected": contradiction_detected,
        "contradiction_details": details,
        "audit_trail": list(state.audit_trail) + [audit_entry],
    }
