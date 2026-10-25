"""
abstention.py: Standardized compliance-safe response generator for abstained states.
Provides clear, inspectable explanations, standardized failure codes, and helpful
guidance when queries are out of domain, adversarial, unanswerable, or ungrounded.
"""

from typing import Any, Dict, Optional
from evidenceflow.state import EvidenceFlowState


def format_abstention_response(
    version: str,
    reason: str,
    code: str = "ABSTAINED",
    hint: Optional[str] = None,
) -> str:
    """
    Constructs a uniform, production-standard enterprise abstention message with
    structured error code and helpful resolution guidance.
    """
    hint_line = f"- **Guidance**: {hint}\n" if hint else ""
    return (
        f"I cannot provide an answer for this question.\n\n"
        f"- **Scope**: PostgreSQL {version}\n"
        f"- **Status**: ABSTAINED (High Integrity Guardrail)\n"
        f"- **Code**: {code}\n"
        f"- **Reason**: {reason}\n"
        f"{hint_line}\n"
        f"*EvidenceFlow AI adheres to strict groundedness: answers are only generated "
        f"when verifiable evidence exists in official documentation.*"
    )


def generate_abstention_hint(code: str, query: str, version: str) -> Optional[str]:
    """
    Generates user-helpful diagnostic hints based on the failure code.
    """
    if code == "UNSUPPORTED_VERSION":
        return "Please specify a supported PostgreSQL version (14, 15, or 16)."
    elif code == "OUT_OF_SCOPE":
        return "Ask a technical question regarding PostgreSQL configuration, internals, or SQL administration."
    elif code == "INJECTION":
        return "Query contains adversarial prompt instructions. Please rephrase as a direct technical question."
    elif code == "MISSING_KEYWORD":
        return "Verify that the parameter or feature name is spelled correctly in official PostgreSQL documentation."
    elif code == "VERSION_MISSING":
        return f"This feature is not present in PostgreSQL {version}. Try checking PostgreSQL 15 or 16."
    elif code == "LOW_SCORE":
        return "Try including specific parameter names (e.g., 'max_wal_size', 'shared_buffers') or relevant sections."
    elif code == "FAKE_CITATION":
        return "The generated response referenced an invalid or ungrounded documentation chunk."
    elif code == "VALUE_MISMATCH":
        return "The cited configuration value or setting parameter does not match the official documentation passage."
    return None


def abstention_node(state: EvidenceFlowState) -> Dict[str, Any]:
    """
    LangGraph node: Formats standardized abstention text and finalizes state.
    """
    version = state.target_version or "16"
    reason = state.abstention_reason or "Insufficient verified documentation evidence."
    code = state.abstention_code or "ABSTAINED"
    hint = generate_abstention_hint(code, state.query, version)
    existing_audit = list(state.audit_trail)

    formatted_answer = format_abstention_response(
        version=version,
        reason=reason,
        code=code,
        hint=hint,
    )

    audit_entry = {
        "node": "abstention",
        "action": "abstention_standardized",
        "code": code,
        "version": version,
        "reason": reason,
        "hint": hint,
    }

    return {
        "draft_answer": formatted_answer,
        "citations": [],
        "is_abstained": True,
        "l1_passed": True,  # Abstention is safe, compliant termination
        "abstention_code": code,
        "abstention_reason": reason,
        "audit_trail": existing_audit + [audit_entry],
    }
