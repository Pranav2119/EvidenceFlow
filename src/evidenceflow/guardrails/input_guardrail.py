"""
input_guardrail.py: Pre-execution security filter and domain relevance guardrail.
Evaluates queries before any retrieval or LLM execution to prevent prompt injection
and reject out-of-domain requests with zero token cost.
"""

import re
from typing import Any, Dict, Optional, Tuple
from evidenceflow.state import EvidenceFlowState

# 1. Prompt Injection & Adversarial Jailbreak Patterns
INJECTION_PATTERNS = [
    re.compile(r"\b(?:ignore|forget|disregard)\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|rules|directives)\b", re.IGNORECASE),
    re.compile(r"\b(?:system\s+prompt|developer\s+mode|unrestricted\s+mode|dan\s+mode|jailbreak)\b", re.IGNORECASE),
    re.compile(r"\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+(?:unfiltered|unrestricted|an\s+ai\s+without\s+rules|evil|dan)\b", re.IGNORECASE),
    re.compile(r"\b(?:reveal|print|display|dump|output)\s+(?:your\s+)?(?:credentials|api[_\s]?key|system\s+prompt|hidden\s+instructions)\b", re.IGNORECASE),
    re.compile(r"(?:###\s*instruction|system:\s*you\s+are|<\|im_start\|>|<\|im_end\|>)", re.IGNORECASE),
]

# 2. Explicit Out-of-Domain Indicators
OFF_DOMAIN_PATTERNS = [
    re.compile(r"\b(?:capital\s+of|weather\s+in|recipe\s+for|tell\s+me\s+a\s+joke|write\s+a\s+(?:poem|story|song))\b", re.IGNORECASE),
    re.compile(r"\b(?:who\s+won\s+the|horoscope|celebrity|football|basketball|cricket\s+score)\b", re.IGNORECASE),
]

# 3. PostgreSQL & Technical Domain Anchors
TECHNICAL_DOMAIN_KEYWORDS = {
    "postgres", "postgresql", "pg", "sql", "database", "table", "index",
    "query", "transaction", "wal", "vacuum", "autovacuum", "checkpoint",
    "buffer", "lock", "replication", "cluster", "server", "connection",
    "memory", "worker", "storage", "schema", "view", "role", "user",
    "privilege", "partition", "log", "archive", "standby", "primary",
}


def evaluate_input_guardrails(query: str) -> Tuple[bool, Optional[str]]:
    """
    Evaluates input query against security and domain guardrails.
    Returns:
        (is_safe, failure_reason)
        is_safe: True if query passes all guardrails, False if it should abstain.
    """
    cleaned = query.strip()
    if not cleaned:
        return False, "Input Guardrail: Query is empty."

    # Check 1: Prompt Injection
    for pattern in INJECTION_PATTERNS:
        if pattern.search(cleaned):
            return False, "Security Guardrail: Potential prompt injection or adversarial directive detected."

    # Check 2: Explicit Off-Domain Query
    for pattern in OFF_DOMAIN_PATTERNS:
        if pattern.search(cleaned):
            return False, "Scope Guardrail: Question is outside PostgreSQL technical documentation scope."

    # Check 3: Domain Relevance Check
    # If question contains an underscore (e.g. max_wal_size, vacuum_cost_limit), it's a technical parameter
    has_parameter = bool(re.search(r"\b[a-z][a-z0-9_]{2,}[a-z0-9]\b", cleaned.lower()) and "_" in cleaned)
    words = set(re.findall(r"\b[a-z]{2,}\b", cleaned.lower()))
    has_domain_anchor = bool(words & TECHNICAL_DOMAIN_KEYWORDS)

    # General questions without any technical or PostgreSQL terms get rejected
    if not has_parameter and not has_domain_anchor and len(words) > 3:
        # Check if query seems like general casual conversation
        casual_patterns = [
            re.compile(r"^(?:hello|hi|hey|good\s+morning|good\s+evening)\b", re.IGNORECASE),
            re.compile(r"\b(?:what\s+is\s+the\s+meaning\s+of\s+life|who\s+are\s+you)\b", re.IGNORECASE),
        ]
        for cp in casual_patterns:
            if cp.search(cleaned):
                return False, "Scope Guardrail: Conversational greetings or general questions must specify a PostgreSQL documentation topic."

    return True, None


def input_guardrail_node(state: EvidenceFlowState) -> Dict[str, Any]:
    """
    LangGraph node: Executes pre-flight security and domain boundary checks.
    Fail-fast: halts pipeline execution before vector search or LLM generation.
    """
    query = state.query
    existing_audit = list(state.audit_trail)

    is_safe, fail_reason = evaluate_input_guardrails(query)

    if not is_safe:
        audit_entry = {
            "node": "input_guardrail",
            "action": "abstained",
            "reason": fail_reason,
            "query": query,
        }
        return {
            "is_abstained": True,
            "abstention_reason": fail_reason,
            "audit_trail": existing_audit + [audit_entry],
        }

    audit_entry = {
        "node": "input_guardrail",
        "action": "passed",
    }
    return {
        "is_abstained": False,
        "abstention_reason": None,
        "audit_trail": existing_audit + [audit_entry],
    }
