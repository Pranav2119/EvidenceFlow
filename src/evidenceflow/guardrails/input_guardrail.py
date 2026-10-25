"""
input_guardrail.py: Pre-execution security filter and domain relevance guardrail.
Evaluates queries before any retrieval or LLM execution to prevent prompt injection,
reject out-of-domain requests, and validate supported PostgreSQL versions.
"""

import re
from typing import Any, Dict, Optional, Tuple
from evidenceflow.state import EvidenceFlowState

# Supported PostgreSQL documentation versions in corpus
SUPPORTED_VERSIONS = {"14", "15", "16"}

# 1. Prompt Injection & Adversarial Jailbreak Patterns
INJECTION_PATTERNS = [
    re.compile(r"\b(?:ignore|forget|disregard)\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|rules|directives)\b", re.IGNORECASE),
    re.compile(r"\b(?:system\s+prompt|developer\s+mode|unrestricted\s+mode|dan\s+mode|jailbreak)\b", re.IGNORECASE),
    re.compile(r"\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+(?:unfiltered|unrestricted|an\s+ai\s+without\s+rules|evil|dan)\b", re.IGNORECASE),
    re.compile(r"\b(?:reveal|print|display|dump|output|show)\s+(?:[a-z_\s]{0,25})?(?:credentials|api[_\s]?keys?|system\s+prompt|prompt\s+instructions|hidden\s+instructions|internal\s+instructions)\b", re.IGNORECASE),
    re.compile(r"(?:###\s*instruction|system:\s*you\s+are|<\|im_start\|>|<\|im_end\|>)", re.IGNORECASE),
]

# 2. Explicit Out-of-Domain Indicators
OFF_DOMAIN_PATTERNS = [
    re.compile(r"\b(?:capital\s+of|weather\s+in|recipe\s+for|tell\s+me\s+a\s+joke|write\s+a\s+(?:poem|story|song))\b", re.IGNORECASE),
    re.compile(r"\b(?:who\s+won\s+the|horoscope|celebrity|football|basketball|cricket\s+score|movie\s+recommendation)\b", re.IGNORECASE),
]

# 3. PostgreSQL & Technical Domain Anchors
TECHNICAL_DOMAIN_KEYWORDS = {
    "postgres", "postgresql", "pg", "sql", "database", "table", "index",
    "query", "transaction", "wal", "vacuum", "autovacuum", "checkpoint",
    "buffer", "lock", "replication", "cluster", "server", "connection",
    "memory", "worker", "storage", "schema", "view", "role", "user",
    "privilege", "partition", "log", "archive", "standby", "primary",
    "pg_restore", "pg_dump", "pg_stat", "pg_basebackup",
}


def evaluate_input_guardrails(query: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Evaluates input query against security, version support, and domain guardrails.
    Returns:
        (is_safe, error_code, failure_reason)
        error_code: 'INJECTION', 'OUT_OF_SCOPE', 'UNSUPPORTED_VERSION', or None
    """
    cleaned = query.strip()
    if not cleaned:
        return False, "OUT_OF_SCOPE", "Input Guardrail: Query is empty."

    # Check 1: Prompt Injection (Strict: ensures phrases like 'ignore errors' in pg_restore are NOT blocked)
    for pattern in INJECTION_PATTERNS:
        if pattern.search(cleaned):
            return False, "INJECTION", "Security Guardrail: Potential prompt injection or adversarial directive detected."

    # Check 2: Version Compatibility Check (Special to EvidenceFlow)
    version_match = re.search(r"\b(?:postgres(?:ql)?|pg|v|version|release)\s*(\d{1,2})\b", cleaned, re.IGNORECASE)
    if version_match:
        requested_v = version_match.group(1)
        if requested_v not in SUPPORTED_VERSIONS:
            return (
                False,
                "UNSUPPORTED_VERSION",
                f"Unsupported Version: PostgreSQL version {requested_v} is not indexed. "
                f"EvidenceFlow currently supports versions {', '.join(sorted(SUPPORTED_VERSIONS))}.",
            )

    # Check 3: Explicit Off-Domain Query
    for pattern in OFF_DOMAIN_PATTERNS:
        if pattern.search(cleaned):
            return False, "OUT_OF_SCOPE", "Scope Guardrail: Question is outside PostgreSQL technical documentation scope."

    # Check 4: Domain Relevance Check
    has_parameter = bool(re.search(r"\b[a-z][a-z0-9_]{2,}[a-z0-9]\b", cleaned.lower()) and "_" in cleaned)
    words = set(re.findall(r"\b[a-z0-9_]{2,}\b", cleaned.lower()))
    has_domain_anchor = bool(words & TECHNICAL_DOMAIN_KEYWORDS)

    # Casual queries with no database keywords
    if not has_parameter and not has_domain_anchor and len(words) > 3:
        casual_patterns = [
            re.compile(r"^(?:hello|hi|hey|good\s+morning|good\s+evening)\b", re.IGNORECASE),
            re.compile(r"\b(?:what\s+is\s+the\s+meaning\s+of\s+life|who\s+are\s+you)\b", re.IGNORECASE),
        ]
        for cp in casual_patterns:
            if cp.search(cleaned):
                return False, "OUT_OF_SCOPE", "Scope Guardrail: Conversational greetings must specify a PostgreSQL documentation topic."

    return True, None, None


def input_guardrail_node(state: EvidenceFlowState) -> Dict[str, Any]:
    """
    LangGraph node: Executes pre-flight security and domain boundary checks.
    Fail-fast: halts pipeline execution before vector search or LLM generation.
    """
    query = state.query
    existing_audit = list(state.audit_trail)

    is_safe, error_code, fail_reason = evaluate_input_guardrails(query)

    if not is_safe:
        audit_entry = {
            "node": "input_guardrail",
            "action": "abstained",
            "code": error_code,
            "reason": fail_reason,
            "query": query,
        }
        return {
            "is_abstained": True,
            "abstention_code": error_code,
            "abstention_reason": fail_reason,
            "audit_trail": existing_audit + [audit_entry],
        }

    audit_entry = {
        "node": "input_guardrail",
        "action": "passed",
    }
    return {
        "is_abstained": False,
        "abstention_code": None,
        "abstention_reason": None,
        "audit_trail": existing_audit + [audit_entry],
    }
