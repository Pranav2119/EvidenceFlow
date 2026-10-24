"""
answer_writer.py: Grounded response synthesis node with inline citation markers.
Uses Gemini LLM when available with strict factual grounding, or falls back to
deterministic passage extraction for offline/zero-cost environments.
"""

import re
import logging
from typing import Any, Dict, List
from evidenceflow.config import get_settings
from evidenceflow.state import EvidenceFlowState

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
    Zero-LLM deterministic synthesis template extracting relevant sentences from top passages.
    """
    if not passages:
        return f"Insufficient evidence in the PostgreSQL {version} documentation."

    top = passages[0]
    cid = top["chunk_id"]
    section = top.get("section", "Documentation")
    content = top.get("content", "").strip()

    # Extract first 2 sentences or 250 characters
    sentences = re.split(r"(?<=[.!?])\s+", content)
    summary = " ".join(sentences[:2]) if sentences else content[:200]

    return f"According to PostgreSQL {version} documentation on {section}: {summary} [^{cid}]"


def extract_citations(answer: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extracts all [^chunk_id] citation markers from the answer and maps them to passage metadata.
    """
    cited_ids = re.findall(r"\[\^([^\]]+)\]", answer)
    passage_map = {p["chunk_id"]: p for p in passages}

    citations = []
    seen = set()
    for cid in cited_ids:
        if cid not in seen:
            seen.add(cid)
            passage = passage_map.get(cid, {})
            citations.append({
                "chunk_id": cid,
                "doc_name": passage.get("doc_name", "unknown"),
                "section": passage.get("section", "unknown"),
                "quote": passage.get("title", ""),
                "is_verified": False,
            })

    return citations


def answer_writer_node(state: EvidenceFlowState) -> Dict[str, Any]:
    """
    LangGraph node: Synthesizes an evidence-grounded response citing retrieved passages.
    """
    query = state.query if hasattr(state, "query") else state["query"]
    target_version = state.target_version if hasattr(state, "target_version") else state.get("target_version", "16")
    passages = list(state.retrieved_passages if hasattr(state, "retrieved_passages") else state.get("retrieved_passages", []))
    existing_audit = list(state.audit_trail if hasattr(state, "audit_trail") else state.get("audit_trail", []))

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

    # Attempt LLM generation via Gemini if configured
    if settings.GEMINI_API_KEY:
        try:
            from google import genai
            client = genai.Client(api_key=settings.GEMINI_API_KEY)
            prompt = build_grounded_prompt(query, target_version, passages)
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
            )
            draft_answer = response.text.strip()
        except Exception as e:
            logger.warning("Gemini generation failed (%s), falling back to deterministic synthesis.", e)
            draft_answer = deterministic_fallback_answer(query, target_version, passages)
    else:
        draft_answer = deterministic_fallback_answer(query, target_version, passages)

    # Check for LLM abstention statement
    is_abstained = False
    abstention_reason = None
    if "insufficient evidence" in draft_answer.lower():
        is_abstained = True
        abstention_reason = "Model declared insufficient evidence in retrieved passages."

    citations = extract_citations(draft_answer, passages)

    audit_entry = {
        "node": "answer_writer",
        "action": "answer_drafted",
        "citations_count": len(citations),
        "is_abstained": is_abstained,
    }

    return {
        "draft_answer": draft_answer,
        "citations": citations,
        "is_abstained": is_abstained,
        "abstention_reason": abstention_reason,
        "audit_trail": existing_audit + [audit_entry],
    }
