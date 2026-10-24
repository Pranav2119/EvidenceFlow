"""
answer_writer.py: Grounded response synthesis node with inline citation markers.
Uses Gemini LLM when available with strict factual grounding, or falls back to
deterministic passage extraction for offline/zero-cost environments.
"""

import re
import logging
from typing import Any, Dict, List, Optional
from evidenceflow.config import get_settings
from evidenceflow.state import EvidenceFlowState, Citation

logger = logging.getLogger(__name__)


def build_grounded_prompt(query: str, version: str, passages: List[Dict[str, Any]]) -> str:
    """
    Constructs a strict grounding prompt with passage chunk IDs.
    """
    passages_text = []
    for i, p in enumerate(passages, 1):
        passages_text.append(
            f"--- PASSAGE {i} [ID: {p['chunk_id']}] (Doc: {p.get('doc_name')}, Section: {p.get('section')}) ---\n"
            f"{p.get('content', '')}\n"
        )

    context = "\n".join(passages_text)

    prompt = (
        f"You are EvidenceFlow, a factual research assistant for PostgreSQL technical documentation.\n"
        f"Answer the user query based ONLY on the documentation passages provided below.\n\n"
        f"STRICT RULES:\n"
        f"1. Every factual statement must cite its source chunk ID using Markdown footnote format: [^chunk_id].\n"
        f"   Example: 'The default value of vacuum_cost_limit is 200 [^pg16-runtime-config-resource-vacuum-cost-limit].'\n"
        f"2. Do NOT invent, assume, or extrapolate any values, units, or parameter settings.\n"
        f"3. If the provided passages do not contain the answer, respond: 'Insufficient evidence in the documentation.'\n\n"
        f"USER QUESTION: {query}\n"
        f"POSTGRESQL VERSION SCOPE: {version}\n\n"
        f"DOCUMENTATION PASSAGES:\n{context}\n\n"
        f"ANSWER:"
    )
    return prompt


def deterministic_fallback_answer(query: str, version: str, passages: List[Dict[str, Any]]) -> str:
    """
    Zero-LLM deterministic synthesis template.
    D3-12 fix: Inspects all passages to find the chunk whose title/chunk_id/content
    best matches the query keywords, rather than blindly citing passages[0].
    """
    if not passages:
        return f"Insufficient evidence in the PostgreSQL {version} documentation."

    # Identify significant query keywords (snake_case or terms > 3 chars)
    query_terms = [t.lower() for t in re.findall(r"\b[a-z0-9_]{3,}\b", query.lower())]

    best_passage = passages[0]
    best_match_count = -1

    for p in passages:
        searchable_text = f"{p.get('chunk_id', '')} {p.get('title', '')} {p.get('content', '')}".lower()
        matches = sum(1 for term in query_terms if term in searchable_text)
        if matches > best_match_count:
            best_match_count = matches
            best_passage = p

    cid = best_passage["chunk_id"]
    section = best_passage.get("section", "Documentation")
    content = best_passage.get("content", "").strip()

    # Extract first 2 sentences or 250 characters
    sentences = re.split(r"(?<=[.!?])\s+", content)
    summary = " ".join(sentences[:2]) if sentences else content[:200]

    return f"According to PostgreSQL {version} documentation on {section}: {summary} [^{cid}]"


def extract_citations(answer: str, passages: List[Dict[str, Any]]) -> List[Citation]:
    """
    Extracts all [^chunk_id] citation markers from the answer and maps them to Citation models.
    D3-4 fix: Constructs strongly-typed Citation Pydantic instances.
    """
    cited_ids = re.findall(r"\[\^([^\]]+)\]", answer)
    passage_map = {p["chunk_id"]: p for p in passages}

    citations: List[Citation] = []
    seen = set()
    for cid in cited_ids:
        if cid not in seen:
            seen.add(cid)
            passage = passage_map.get(cid, {})
            citations.append(
                Citation(
                    chunk_id=cid,
                    doc_name=passage.get("doc_name", "unknown"),
                    section=passage.get("section", "unknown"),
                    quote=passage.get("title", ""),
                    is_verified=False,
                )
            )

    return citations


def is_valid_gemini_key(api_key: str) -> bool:
    """
    D3-1 fix: Validates API key is non-empty and not a known placeholder string.
    """
    if not api_key or not isinstance(api_key, str):
        return False
    placeholders = {"your_key_here", "change_me", "placeholder", "your_gemini_api_key"}
    cleaned = api_key.strip().lower()
    return len(cleaned) > 15 and cleaned not in placeholders


def answer_writer_node(state: EvidenceFlowState) -> Dict[str, Any]:
    """
    LangGraph node: Synthesizes an evidence-grounded response citing retrieved passages.
    D3-1: Validates Gemini key format before invoking LLM.
    D3-2: Direct Pydantic attribute access.
    D3-4: Typed Citation instances returned.
    D3-5: Narrow exception handling catching only API/network errors.
    D3-13: Usage metadata / token counts captured in audit trail.
    """
    query = state.query
    target_version = state.target_version or "16"
    passages = list(state.retrieved_passages)
    existing_audit = list(state.audit_trail)

    if not passages:
        reason = f"No relevant passages found for PostgreSQL {target_version}."
        audit_entry = {
            "node": "answer_writer",
            "action": "abstained",
            "reason": reason,
        }
        return {
            "draft_answer": f"I cannot answer this question because no relevant documentation was found for PostgreSQL {target_version}.",
            "citations": [],
            "is_abstained": True,
            "abstention_reason": reason,
            "audit_trail": existing_audit + [audit_entry],
        }

    settings = get_settings()
    draft_answer = ""
    token_usage: Dict[str, Any] = {}

    # D3-1: Check for real API key
    if is_valid_gemini_key(settings.GEMINI_API_KEY):
        try:
            from google import genai
            from google.genai.errors import APIError, ClientError, ServerError

            client = genai.Client(api_key=settings.GEMINI_API_KEY)
            prompt = build_grounded_prompt(query, target_version, passages)
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
            )
            draft_answer = response.text.strip()

            # D3-13: Capture token metrics for ablation & benchmarking
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                token_usage = {
                    "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "candidate_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                    "total_tokens": getattr(response.usage_metadata, "total_token_count", 0),
                }

        # D3-5: Catch specific GenAI and network errors without swallowing code bugs
        except (APIError, ClientError, ServerError, ConnectionError, TimeoutError) as e:
            logger.warning("Gemini generation API call failed (%s), falling back to deterministic synthesis.", e)
            draft_answer = deterministic_fallback_answer(query, target_version, passages)
    else:
        logger.info("No active Gemini API key configured. Using deterministic synthesis.")
        draft_answer = deterministic_fallback_answer(query, target_version, passages)

    # Check for LLM abstention statement
    is_abstained = False
    abstention_reason = None
    if "insufficient evidence" in draft_answer.lower():
        is_abstained = True
        abstention_reason = "Model declared insufficient evidence in retrieved passages."

    # D3-4: strongly-typed Citation objects
    citations = extract_citations(draft_answer, passages)

    audit_entry = {
        "node": "answer_writer",
        "action": "answer_drafted",
        "citations_count": len(citations),
        "is_abstained": is_abstained,
        **token_usage,
    }

    return {
        "draft_answer": draft_answer,
        "citations": citations,
        "is_abstained": is_abstained,
        "abstention_reason": abstention_reason,
        "audit_trail": existing_audit + [audit_entry],
    }
