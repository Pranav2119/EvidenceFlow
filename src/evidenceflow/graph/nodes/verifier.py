"""
verifier.py: Deterministic L1 Citation Existence and L2 Value-Presence & Proximity Verifier.
Executes pure Python verification (ZERO LLM calls) to guarantee:
  1. Citation Existence: Every cited chunk ID exists in retrieved context.
  2. Citation Completeness: Ungrounded statements with zero citations are caught.
  3. L1.5/L2 Value & Proximity Presence: Numeric parameters, units, and enum values cited
     in a claim actually exist in close proximity to the setting in the cited passage text.
  4. Unit Equivalence: Treats equivalent units identically (e.g. 1min == 60s, 1024kB == 1MB).
  5. Setting Association Check: Prevents multi-value chunk spoofing where a chunk mentions
     multiple numbers (e.g. 200 and 400) and the LLM cites the wrong one for a setting.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from evidenceflow.state import EvidenceFlowState, Citation

RECOGNIZED_ENUM_VALUES = {
    "replica",
    "minimal",
    "logical",
    "scram-sha-256",
    "serializable",
    "repeatable read",
    "read committed",
    "read uncommitted",
}


def get_equivalent_quantities(val: str) -> Set[str]:
    """
    Expands a quantity string into its unit-normalized equivalents.
    Supports time conversions (s, min, ms) and memory conversions (kB, MB, GB).
    """
    norm = val.strip().lower()
    variants = {
        norm,
        re.sub(r"([0-9])([a-z])", r"\1 \2", norm),
        re.sub(r"([0-9])\s+([a-z])", r"\1\2", norm),
    }

    # Match numeric quantity and unit
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([a-z]+)$", norm)
    if m:
        num = float(m.group(1))
        unit = m.group(2)
        int_num = int(num) if num.is_integer() else None

        # Time equivalences
        if unit == "min" and int_num is not None:
            secs = int_num * 60
            variants.update({f"{secs}s", f"{secs} s", f"{secs * 1000}ms", f"{secs * 1000} ms"})
        elif unit == "s" and int_num is not None:
            ms = int_num * 1000
            variants.update({f"{ms}ms", f"{ms} ms"})
            if int_num % 60 == 0:
                mins = int_num // 60
                variants.update({f"{mins}min", f"{mins} min"})
        elif unit == "ms" and int_num is not None:
            if int_num % 1000 == 0:
                secs = int_num // 1000
                variants.update({f"{secs}s", f"{secs} s"})
                if secs % 60 == 0:
                    mins = secs // 60
                    variants.update({f"{mins}min", f"{mins} min"})

        # Memory equivalences (Postgres base 1024)
        elif unit == "mb" and int_num is not None:
            kb = int_num * 1024
            variants.update({f"{kb}kb", f"{kb} kb"})
            if int_num % 1024 == 0:
                gb = int_num // 1024
                variants.update({f"{gb}gb", f"{gb} gb"})
        elif unit == "kb" and int_num is not None:
            if int_num % 1024 == 0:
                mb = int_num // 1024
                variants.update({f"{mb}mb", f"{mb} mb"})
        elif unit == "gb" and int_num is not None:
            mb = int_num * 1024
            variants.update({f"{mb}mb", f"{mb} mb"})

    return variants


def extract_claimed_values(statement: str) -> List[str]:
    """
    Extracts numeric quantities and configuration enum values from a claim statement.
    Excludes bare single-digit numbers (to avoid false triggers on versions like '9.6' or '14.4.7').
    """
    # 1. Numeric quantities with units (e.g. '128MB', '1min', '60s', '8kB') OR multi-digit integers ('100', '200', '400')
    raw_quantities = re.findall(
        r"\b\d+(?:\.\d+)?\s*(?:ms|min|kB|MB|GB|TB|pages|bytes|rows|s|h|d)\b|\b\d{2,}\b",
        statement,
        re.IGNORECASE,
    )
    # Filter out common PostgreSQL versions
    filtered = [
        q.strip().lower()
        for q in raw_quantities
        if q.strip() not in {"14", "15", "16"}
    ]

    # 2. Domain enum values (e.g., 'replica', 'logical', 'minimal')
    stmt_lower = statement.lower()
    for enum_val in RECOGNIZED_ENUM_VALUES:
        if re.search(r"\b" + re.escape(enum_val) + r"\b", stmt_lower):
            filtered.append(enum_val)

    # 3. Explicit boolean / state assertions (e.g. 'is on', 'defaults to off', '= on')
    bool_states = re.findall(
        r"\b(?:is|to|set\s+to|defaults?\s+to|=)\s+(on|off|true|false)\b",
        stmt_lower,
    )
    for bs in bool_states:
        filtered.append(bs.lower())

    return list(dict.fromkeys(filtered))


def split_answer_into_citation_statements(draft_answer: str) -> List[Tuple[str, List[str]]]:
    """
    Splits a draft response into individual factual statements, each bound to its cited chunk IDs.
    Handles:
      - Multiple sentences, each with its own [^citation]
      - Bullet points with citations
      - Multi-citation blocks like [^chunk1, ^chunk2]
      - Consecutive citation blocks like [^chunk1] [^chunk2]
    Returns:
      List of (statement_text, [cited_chunk_ids])
    """
    citation_pattern = re.compile(r"\[\^([^\]]+)\]")
    matches = list(citation_pattern.finditer(draft_answer))
    if not matches:
        return []

    # Group adjacent citations together if separated only by whitespace or punctuation
    grouped = []
    i = 0
    while i < len(matches):
        m = matches[i]
        cids = [c.strip().lstrip("^") for c in re.split(r"[,;\s]+", m.group(1)) if c.strip().lstrip("^")]
        start_idx = m.start()
        end_idx = m.end()
        j = i + 1
        while j < len(matches):
            between = draft_answer[end_idx:matches[j].start()]
            if not between.strip() or re.fullmatch(r"[,;\s]*", between):
                cids.extend([c.strip().lstrip("^") for c in re.split(r"[,;\s]+", matches[j].group(1)) if c.strip().lstrip("^")])
                end_idx = matches[j].end()
                j += 1
            else:
                break
        grouped.append((start_idx, end_idx, list(dict.fromkeys(cids))))
        i = j

    results = []
    last_end = 0
    for start_idx, end_idx, cids in grouped:
        raw_stmt = draft_answer[last_end:start_idx].strip()
        # Clean leading bullet markers, newlines, or sentence-separating punctuation
        raw_stmt = re.sub(r"^[.\n\r\t\-*#;,\s]+", "", raw_stmt).strip()
        results.append((raw_stmt, cids))
        last_end = end_idx

    return results



def verify_statement_value_presence(
    statement: str,
    passage_content: str,
) -> Tuple[List[str], Optional[str]]:
    """
    Verifies that specific quantities, units, and enum states mentioned in the statement
    exist in the cited passage text, and respect setting proximity boundaries.

    Returns:
        (missing_values, detail_error_message)
    """
    claimed_values = extract_claimed_values(statement)
    if not claimed_values:
        return [], None

    content_lower = passage_content.lower()

    # Extract any snake_case parameter names present in the statement
    stmt_params = re.findall(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b", statement.lower())

    # Build proximity text windows if parameter is present in passage
    proximity_windows = []
    if stmt_params:
        for p in stmt_params:
            if p in content_lower:
                for match in re.finditer(re.escape(p), content_lower):
                    start = max(0, match.start() - 60)
                    end = min(len(content_lower), match.end() + 280)
                    proximity_windows.append(content_lower[start:end])

    missing_values = []
    for val in claimed_values:
        equiv_candidates = get_equivalent_quantities(val)

        # If parameter proximity windows exist, check that the value is in the proximity window
        if proximity_windows:
            in_proximity = any(
                cand in window
                for cand in equiv_candidates
                for window in proximity_windows
            )
            if not in_proximity:
                # Value is either completely missing or attached to a different setting elsewhere in chunk
                in_passage_elsewhere = any(cand in content_lower for cand in equiv_candidates)
                if in_passage_elsewhere:
                    return [val], f"Value '{val}' appears elsewhere in chunk, but does not belong to setting '{stmt_params[0]}'."
                else:
                    missing_values.append(val)
        else:
            # Fall back to whole passage check
            in_passage = any(cand in content_lower for cand in equiv_candidates)
            if not in_passage:
                missing_values.append(val)

    if missing_values:
        return missing_values, f"Values {missing_values} do not appear in cited passage."
    return [], None


def verifier_node(state: EvidenceFlowState) -> Dict[str, Any]:
    """
    LangGraph node: Deterministic citation existence and value-presence check.
    Fails L1/L2 verification if:
      - Any cited chunk is hallucinated (not retrieved) -> FAKE_CITATION
      - The answer makes claims with zero citations -> FAKE_CITATION
      - A numeric or configuration value in a claim does not appear in the source passage -> VALUE_MISMATCH
      - A numeric value belongs to a different setting in the same chunk -> VALUE_MISMATCH
    """
    draft_answer = state.draft_answer
    citations = list(state.citations)
    passages = list(state.retrieved_passages)
    is_abstained = state.is_abstained
    abstention_reason = state.abstention_reason
    existing_audit = list(state.audit_trail)

    # If already abstained by upstream guardrails, pass through
    if is_abstained:
        audit_entry = {
            "node": "verifier",
            "action": "passed_through_abstention",
            "l1_passed": True,
            "reason": abstention_reason,
        }
        return {
            "l1_passed": True,
            "is_abstained": True,
            "abstention_code": state.abstention_code,
            "abstention_reason": abstention_reason,
            "audit_trail": existing_audit + [audit_entry],
        }

    # Extract all citation markers [^chunk_id] in the draft answer, normalizing multi-citations
    raw_citation_blocks = re.findall(r"\[\^([^\]]+)\]", draft_answer)
    cited_chunk_ids = []
    for block in raw_citation_blocks:
        parts = re.split(r"[,;\s]+", block)
        for part in parts:
            clean_cid = part.lstrip("^").strip()
            if clean_cid:
                cited_chunk_ids.append(clean_cid)

    passage_map = {p["chunk_id"]: p.get("content", "") for p in passages}
    retrieved_chunk_ids = set(passage_map.keys())

    # Failure Mode 1: Answer has no citations at all
    if not cited_chunk_ids:
        fail_reason = "L1 Failed: Answer contains ungrounded statements with zero citations."
        audit_entry = {
            "node": "verifier",
            "action": "l1_failed_no_citations",
            "code": "FAKE_CITATION",
            "l1_passed": False,
        }
        return {
            "l1_passed": False,
            "is_abstained": True,
            "abstention_code": "FAKE_CITATION",
            "abstention_reason": fail_reason,
            "audit_trail": existing_audit + [audit_entry],
        }

    # Failure Mode 2: Hallucinated citations (cited chunk was NOT in retrieved_passages)
    hallucinated_ids = [cid for cid in cited_chunk_ids if cid not in retrieved_chunk_ids]
    if hallucinated_ids:
        fail_reason = f"L1 Failed: Hallucinated citation(s) not in retrieved evidence: {hallucinated_ids}"
        audit_entry = {
            "node": "verifier",
            "action": "l1_failed_hallucinated_citations",
            "code": "FAKE_CITATION",
            "hallucinated": hallucinated_ids,
            "l1_passed": False,
        }
        return {
            "l1_passed": False,
            "is_abstained": True,
            "abstention_code": "FAKE_CITATION",
            "abstention_reason": fail_reason,
            "audit_trail": existing_audit + [audit_entry],
        }

    # Failure Mode 3 (L1.5 & L2): Per-Citation Value & Proximity Presence Verification
    statements = split_answer_into_citation_statements(draft_answer)
    for stmt_text, clause_cids in statements:
        if not stmt_text:
            continue

        if len(clause_cids) == 1:
            cid = clause_cids[0]
            passage_content = passage_map.get(cid, "")
            missing, err_detail = verify_statement_value_presence(stmt_text, passage_content)
            if missing or err_detail:
                fail_reason = (
                    f"L1.5/L2 Value Verification Failed: {err_detail or missing} in source passage '{cid}'."
                )
                audit_entry = {
                    "node": "verifier",
                    "action": "l2_failed_value_mismatch",
                    "code": "VALUE_MISMATCH",
                    "chunk_id": cid,
                    "statement": stmt_text,
                    "missing_values": missing,
                    "detail": err_detail,
                    "l1_passed": False,
                }
                return {
                    "l1_passed": False,
                    "is_abstained": True,
                    "abstention_code": "VALUE_MISMATCH",
                    "abstention_reason": fail_reason,
                    "audit_trail": existing_audit + [audit_entry],
                }
        else:
            # Multi-chunk citation: e.g. [^chunk1, ^chunk2]
            stmt_params = re.findall(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b", stmt_text.lower())
            for cid in clause_cids:
                passage_content = passage_map.get(cid, "")
                p_lower = passage_content.lower()

                # If statement names multiple settings, check the part relevant to this chunk
                matching_params = [p for p in stmt_params if p in p_lower]
                if matching_params and len(matching_params) < len(stmt_params):
                    for p in matching_params:
                        m = re.search(re.escape(p) + r"[^,;\.]*", stmt_text, re.IGNORECASE)
                        sub_stmt = m.group(0) if m else stmt_text
                        missing, err_detail = verify_statement_value_presence(sub_stmt, passage_content)
                        if missing or err_detail:
                            fail_reason = (
                                f"L1.5/L2 Value Verification Failed: {err_detail or missing} in source passage '{cid}'."
                            )
                            audit_entry = {
                                "node": "verifier",
                                "action": "l2_failed_value_mismatch",
                                "code": "VALUE_MISMATCH",
                                "chunk_id": cid,
                                "statement": sub_stmt,
                                "missing_values": missing,
                                "detail": err_detail,
                                "l1_passed": False,
                            }
                            return {
                                "l1_passed": False,
                                "is_abstained": True,
                                "abstention_code": "VALUE_MISMATCH",
                                "abstention_reason": fail_reason,
                                "audit_trail": existing_audit + [audit_entry],
                            }
                else:
                    # Shared claim across all cited chunks (e.g. 'Both settings default to 500')
                    missing, err_detail = verify_statement_value_presence(stmt_text, passage_content)
                    if missing or err_detail:
                        fail_reason = (
                            f"L1.5/L2 Value Verification Failed: {err_detail or missing} in source passage '{cid}'."
                        )
                        audit_entry = {
                            "node": "verifier",
                            "action": "l2_failed_value_mismatch",
                            "code": "VALUE_MISMATCH",
                            "chunk_id": cid,
                            "statement": stmt_text,
                            "missing_values": missing,
                            "detail": err_detail,
                            "l1_passed": False,
                        }
                        return {
                            "l1_passed": False,
                            "is_abstained": True,
                            "abstention_code": "VALUE_MISMATCH",
                            "abstention_reason": fail_reason,
                            "audit_trail": existing_audit + [audit_entry],
                        }

    # Mark Citations as verified
    updated_citations: List[Union[Citation, Dict[str, Any]]] = []
    for c in citations:
        if isinstance(c, Citation):
            is_valid = c.chunk_id in retrieved_chunk_ids
            updated_citations.append(c.model_copy(update={"is_verified": is_valid}))
        else:
            c_copy = dict(c)
            if c_copy.get("chunk_id") in retrieved_chunk_ids:
                c_copy["is_verified"] = True
            updated_citations.append(c_copy)

    audit_entry = {
        "node": "verifier",
        "action": "l1_passed",
        "l1_passed": True,
        "verified_citations_count": len(updated_citations),
    }

    return {
        "l1_passed": True,
        "is_abstained": False,
        "abstention_code": None,
        "abstention_reason": None,
        "citations": updated_citations,
        "audit_trail": existing_audit + [audit_entry],
    }
