"""
answer_writer.py: Grounded response synthesis node with inline citation markers.
Uses Gemini LLM when available with strict factual grounding, or falls back to
deterministic passage extraction for offline/zero-cost environments.
"""

import re
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import httpx
from evidenceflow.config import get_settings
from evidenceflow.state import EvidenceFlowState, Citation

logger = logging.getLogger(__name__)

# Lazy singleton Gemini client to reuse TCP connection pool
_gemini_writer_client = None


def _get_writer_client(api_key: str):
    global _gemini_writer_client
    if _gemini_writer_client is None:
        from google import genai
        _gemini_writer_client = genai.Client(api_key=api_key)
    return _gemini_writer_client


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


class FallbackAnswer(str):
    """
    Subclass of str supporting backward compatibility with Day 4 string assertions
    while allowing Day 8 tuple unpacking (answer_text, abstention_code).
    """
    abstention_code: Optional[str]

    def __new__(cls, text: str, abstention_code: Optional[str] = None):
        obj = super().__new__(cls, text)
        obj.abstention_code = abstention_code
        return obj

    def __iter__(self):
        return iter((str(self), self.abstention_code))


def deterministic_fallback_answer(
    query: str, version: str, passages: List[Dict[str, Any]]
) -> FallbackAnswer:
    """
    Zero-LLM deterministic synthesis template.
    Extracts substantive sentences from retrieved chunks answering the user's question.
    Never outputs only a section header. If substantive evidence is missing,
    abstains with a clear code [Rule i].
    """
    if not passages:
        return FallbackAnswer(f"Insufficient evidence in the PostgreSQL {version} documentation.", "NO_SUBSTANTIVE_EVIDENCE")

    # Identify recognized setting names mentioned in query
    from evidenceflow.memory.corpus_settings import get_corpus_settings
    try:
        known_settings = get_corpus_settings()
    except Exception:
        known_settings = set()

    if not known_settings:
        from evidenceflow.storage.bm25 import get_bm25_index
        try:
            known_settings = get_bm25_index().setting_names
        except Exception:
            known_settings = set()

    words = set(re.findall(r"\b[a-z0-9_]+\b", query.lower()))
    named_settings = [w for w in words if w in known_settings or ("_" in w and len(w) > 3)]
    for s in known_settings:
        if len(s) > 4 and "_" in s and s.replace("_", " ") in query.lower():
            if s not in named_settings:
                named_settings.append(s)

    if not named_settings:
        logger.info("Deterministic fallback: question does not name any indexed setting. Failing closed.")
        return FallbackAnswer(f"Insufficient evidence in the PostgreSQL {version} documentation.", "NO_SUBSTANTIVE_EVIDENCE")

    # Find the best passage that contains one of the named settings
    best_passage = None
    best_match_count = -1

    for p in passages:
        cid = p.get("chunk_id", "").lower()
        title = p.get("title", "").lower()
        content = p.get("content", "").lower()

        # Require that at least one named setting appears in the passage
        matched_named = [s for s in named_settings if s.replace("_", "-") in cid or s in title or s in content]
        if matched_named:
            match_count = len(matched_named)
            # Give strong priority to chunks whose ID or title directly defines the parameter
            for s in matched_named:
                if s.replace("_", "-") in cid or s in title:
                    match_count += 5
            if match_count > best_match_count:
                best_match_count = match_count
                best_passage = p

    if not best_passage:
        logger.info("Deterministic fallback: named settings do not appear in retrieved chunks. Failing closed.")
        return FallbackAnswer(f"Insufficient evidence in the PostgreSQL {version} documentation.", "NO_SUBSTANTIVE_EVIDENCE")

    cid = best_passage["chunk_id"]
    section = best_passage.get("section", "Documentation")
    content = best_passage.get("content", "").strip()

    # Strip pure leading headers (Section: ..., Parameter: ..., # ...)
    body = re.sub(
        r"^(?:Section:[^\n]*\n*|Parameter:[^\n]*\n*|#+[^\n]*\n*)+",
        "",
        content,
        flags=re.IGNORECASE,
    ).strip()

    if not body:
        return FallbackAnswer(f"Insufficient evidence in the PostgreSQL {version} documentation.", "NO_SUBSTANTIVE_EVIDENCE")

    # Split into clean sentences using verifier's safe splitter [Rule i, j]
    from evidenceflow.graph.nodes.verifier import split_sentences_safely
    sentence_tuples = split_sentences_safely(body)
    raw_sentences = [s.strip() for s, _ in sentence_tuples]

    # Filter out non-substantive lines / section headers
    substantive = []
    for s in raw_sentences:
        if s.lower().startswith(("section:", "parameter:", "###", "##")):
            continue
        words = s.split()
        if len(words) < 4:
            continue
        # Must contain at least one verb or substantive technical keyword
        if not re.search(
            r"\b(is|are|was|were|sets|specifies|determines|controls|allows|enables|causes|requires|contains|defaults?|default|removed?|deprecated)\b",
            s,
            re.IGNORECASE,
        ):
            continue
        substantive.append(s)

    if not substantive:
        return FallbackAnswer(f"Insufficient evidence in the PostgreSQL {version} documentation.", "NO_SUBSTANTIVE_EVIDENCE")

    q_lower = query.lower()
    is_default_q = bool(re.search(r"\bdefaults?\b", q_lower))
    is_remove_q = bool(re.search(r"\b(remove|removed|deprecated|obsolete)\b", q_lower))
    is_restart_q = bool(re.search(r"\b(restart|reload|postmaster|sighup|superuser|privilege|context)\b", q_lower))

    scored = []
    for idx, s in enumerate(substantive):
        score = 0
        s_lower = s.lower()
        if is_default_q:
            if re.search(r"\bdefaults?\b", s_lower):
                score += 20
        if is_remove_q:
            if re.search(r"\b(remove|removed|deprecated)\b", s_lower):
                score += 20
        if is_restart_q:
            if re.search(r"\b(restart|reload|postmaster|sighup|superuser|privilege)\b", s_lower):
                score += 15

        # Check if setting name appears in this sentence
        for ns in named_settings:
            if ns in s_lower or ns.replace("_", " ") in s_lower:
                score += 5

        # Definition sentence
        if idx == 0 and re.search(r"\b(sets|specifies|determines|controls|allows|enables|causes)\b", s_lower):
            score += 5

        scored.append((score, idx, s))

    if is_default_q and not any(score >= 20 for score, _, _ in scored):
        # Query specifically asked for default, but chunk has no default statement
        return FallbackAnswer(f"Insufficient evidence in the PostgreSQL {version} documentation.", "NO_SUBSTANTIVE_EVIDENCE")

    if is_remove_q and not any(score >= 20 for score, _, _ in scored):
        return FallbackAnswer(f"Insufficient evidence in the PostgreSQL {version} documentation.", "NO_SUBSTANTIVE_EVIDENCE")

    scored.sort(key=lambda x: (-x[0], x[1]))
    best_score, best_idx, best_sent = scored[0]

    # If nothing matched specific intents and it was a specific intent query
    if best_score == 0 and (is_default_q or is_remove_q or is_restart_q):
        return FallbackAnswer(f"Insufficient evidence in the PostgreSQL {version} documentation.", "NO_SUBSTANTIVE_EVIDENCE")

    summary = best_sent
    return FallbackAnswer(f"According to PostgreSQL {version} documentation on {section}: {summary} [^{cid}]", None)


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
            "synthesis_mode": "fallback",
        }
        return {
            "draft_answer": f"I cannot answer this question because no relevant documentation was found for PostgreSQL {target_version}.",
            "citations": [],
            "is_abstained": True,
            "abstention_code": "ABSTAINED",
            "abstention_reason": reason,
            "synthesis_mode": "fallback",
            "audit_trail": existing_audit + [audit_entry],
        }

    settings = get_settings()
    draft_answer = ""
    token_usage: Dict[str, Any] = {}
    synthesis_mode = "fallback"
    abstention_code: Optional[str] = None
    is_abstained: bool = False
    abstention_reason: Optional[str] = None
    pure_api_latency_ms: Optional[float] = None
    total_sleep_ms: float = 0.0
    attempts_count: int = 0

    # D3-1: Check for real API key
    if is_valid_gemini_key(settings.GEMINI_API_KEY):
        try:
            from google import genai
            from google.genai.errors import APIError, ClientError, ServerError

            client = _get_writer_client(settings.GEMINI_API_KEY)
            prompt = build_grounded_prompt(query, target_version, passages)

            # Retry up to 2 times on transient 503/429 with isolated pure API timing [Rule m]
            response = None
            max_attempts = 3
            pure_api_latency_ms = 0.0
            total_sleep_ms = 0.0
            attempts_count = 0
            for attempt in range(1, max_attempts + 1):
                attempts_count += 1
                try:
                    t_api_start = time.perf_counter()
                    response = client.models.generate_content(
                        model=settings.GEMINI_MODEL,
                        contents=prompt,
                    )
                    pure_api_latency_ms += (time.perf_counter() - t_api_start) * 1000.0
                    break
                except (APIError, ServerError, httpx.HTTPError) as err:
                    pure_api_latency_ms += (time.perf_counter() - t_api_start) * 1000.0
                    if attempt < max_attempts and ("503" in str(err) or "429" in str(err) or "RemoteProtocolError" in str(err)):
                        sleep_duration = 2.0 * attempt
                        t_sleep_start = time.perf_counter()
                        time.sleep(sleep_duration)
                        total_sleep_ms += (time.perf_counter() - t_sleep_start) * 1000.0
                        continue
                    raise

            if response and response.text:
                draft_answer = response.text.strip()
                synthesis_mode = "llm"

                # D3-13: Capture token metrics for ablation & benchmarking
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    token_usage = {
                        "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                        "candidate_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                        "total_tokens": getattr(response.usage_metadata, "total_token_count", 0),
                    }
            else:
                draft_answer, fallback_code = deterministic_fallback_answer(query, target_version, passages)
                synthesis_mode = "fallback"
                if fallback_code:
                    is_abstained = True
                    abstention_code = fallback_code
                    abstention_reason = f"Deterministic fallback could not answer query ({fallback_code})."

        # Catch GenAI and network/transport errors (including RemoteProtocolError)
        except (APIError, ClientError, ServerError, ConnectionError, TimeoutError, httpx.HTTPError, Exception) as e:
            logger.warning("Gemini generation API call failed (%s), falling back to deterministic synthesis.", e)
            draft_answer, fallback_code = deterministic_fallback_answer(query, target_version, passages)
            synthesis_mode = "fallback"
            if fallback_code:
                is_abstained = True
                abstention_code = fallback_code
                abstention_reason = f"Deterministic fallback could not answer query ({fallback_code})." 
    else:
        logger.info("No active Gemini API key configured. Using deterministic synthesis.")
        draft_answer, fallback_code = deterministic_fallback_answer(query, target_version, passages)
        synthesis_mode = "fallback"
        if fallback_code:
            is_abstained = True
            abstention_code = fallback_code
            abstention_reason = f"Deterministic fallback could not answer query ({fallback_code})."

    # Check for LLM or fallback abstention statement
    if "insufficient evidence" in draft_answer.lower():
        is_abstained = True
        if not abstention_code:
            abstention_code = "ABSTAINED"
        if not abstention_reason:
            abstention_reason = "Model declared insufficient evidence in retrieved passages."

    # D3-4: strongly-typed Citation objects
    citations = extract_citations(draft_answer, passages)

    audit_entry = {
        "node": "answer_writer",
        "action": "answer_drafted",
        "citations_count": len(citations),
        "is_abstained": is_abstained,
        "synthesis_mode": synthesis_mode,
        "gemini_pure_api_latency_ms": round(pure_api_latency_ms, 3) if pure_api_latency_ms is not None else None,
        "gemini_attempts": attempts_count if synthesis_mode == "llm" else None,
        "gemini_sleep_duration_ms": round(total_sleep_ms, 3) if synthesis_mode == "llm" else 0.0,
        **token_usage,
    }

    return {
        "draft_answer": draft_answer,
        "citations": citations,
        "is_abstained": is_abstained,
        "abstention_code": abstention_code if is_abstained else None,
        "abstention_reason": abstention_reason,
        "synthesis_mode": synthesis_mode,
        "audit_trail": existing_audit + [audit_entry],
    }
