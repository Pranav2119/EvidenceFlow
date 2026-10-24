"""
abstention.py: Standardized compliance-safe response generator for abstained states.
Provides clear, inspectable explanations when queries are out of domain, adversarial,
unanswerable, or ungrounded.
"""

from typing import Any, Dict
from evidenceflow.state import EvidenceFlowState


def format_abstention_response(version: str, reason: str) -> str:
    """
    Constructs a uniform, production-standard enterprise abstention message.
    """
    return (
        f"I cannot provide an answer for this question.\n\n"
        f"- **Scope**: PostgreSQL {version}\n"
        f"- **Status**: ABSTAINED (High Integrity Guardrail)\n"
        f"- **Reason**: {reason}\n\n"
        f"*EvidenceFlow AI adheres to strict groundedness: answers are only generated "
        f"when verifiable evidence exists in official documentation.*"
    )


def abstention_node(state: EvidenceFlowState) -> Dict[str, Any]:
    """
    LangGraph node: Formats standardized abstention text and finalizes state.
    """
    version = state.target_version or "16"
    reason = state.abstention_reason or "Insufficient verified documentation evidence."
    existing_audit = list(state.audit_trail)

    formatted_answer = format_abstention_response(version, reason)

    audit_entry = {
        "node": "abstention",
        "action": "abstention_standardized",
        "version": version,
        "reason": reason,
    }

    return {
        "draft_answer": formatted_answer,
        "citations": [],
        "is_abstained": True,
        "l1_passed": True,  # Abstention is safe, compliant termination
        "abstention_reason": reason,
        "audit_trail": existing_audit + [audit_entry],
    }
