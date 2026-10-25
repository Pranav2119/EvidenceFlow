"""
ledger.py: Relational ACID Step Ledger & Claim-Evidence Audit Trail for EvidenceFlow AI.
Uses psycopg 3 and connection pooling to record immutable turns, node transitions,
and deterministic verification verdicts into PostgreSQL.
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from evidenceflow.config import get_settings

logger = logging.getLogger(__name__)

# Module-level connection pool singleton
_pool_instance: Optional[ConnectionPool] = None


def get_postgres_connection_string(use_app_role: bool = True) -> str:
    """
    Builds standard psycopg connection string from Pydantic settings.
    When use_app_role is True, uses the limited runtime app role (evidenceflow_app)
    which has SELECT and INSERT privileges, but strictly NO UPDATE/DELETE/TRUNCATE on audit tables.
    When use_app_role is False, uses the admin/owner role (evidenceflow) for migrations and DDL setup.
    """
    s = get_settings()
    user = s.POSTGRES_APP_USER if use_app_role else s.POSTGRES_USER
    pwd = s.POSTGRES_APP_PASSWORD if use_app_role else s.POSTGRES_PASSWORD
    return f"host={s.POSTGRES_HOST} port={s.POSTGRES_PORT} user={user} password={pwd} dbname={s.POSTGRES_DB}"


def get_connection_pool(use_app_role: bool = True) -> ConnectionPool:
    """Returns the process-level shared ConnectionPool singleton."""
    global _pool_instance
    if _pool_instance is None or _pool_instance.closed:
        conn_str = get_postgres_connection_string(use_app_role=use_app_role)
        try:
            _pool_instance = ConnectionPool(
                conninfo=conn_str,
                min_size=1,
                max_size=10,
                open=True,
                kwargs={"autocommit": True, "row_factory": dict_row},
            )
        except Exception as e:
            if use_app_role:
                logger.warning("Failed to connect using limited app role (%s), falling back to owner role.", e)
                conn_str = get_postgres_connection_string(use_app_role=False)
                _pool_instance = ConnectionPool(
                    conninfo=conn_str,
                    min_size=1,
                    max_size=10,
                    open=True,
                    kwargs={"autocommit": True, "row_factory": dict_row},
                )
            else:
                raise e
    return _pool_instance


def close_connection_pool():
    """Closes the connection pool (used for test isolation across restarts)."""
    global _pool_instance
    if _pool_instance is not None and not _pool_instance.closed:
        _pool_instance.close()
        _pool_instance = None


def sanitize_payload(obj: Any) -> Any:
    """
    Recursively strips sensitive keys (API keys, passwords, credentials)
    before writing payloads into the audit ledger.
    Guarantees that token counts (tokens_in, tokens_out, prompt_tokens, etc.) are never redacted.
    """
    token_count_keys = {
        "tokens_in", "tokens_out", "prompt_tokens", "candidate_tokens",
        "total_tokens", "token_count", "tokens", "tokens_used"
    }
    sensitive_patterns = (
        "api_key", "password", "secret", "authorization",
        "credential", "access_token", "private_key", "auth_token",
        "bearer_token", "token"
    )
    if isinstance(obj, dict):
        clean = {}
        for k, v in obj.items():
            k_str = str(k).lower()
            # 1. Explicitly protect numeric token counts
            if k_str in token_count_keys or (("token" in k_str or "count" in k_str) and isinstance(v, (int, float))):
                clean[k] = v
            # 2. Redact sensitive patterns
            elif any(p in k_str for p in sensitive_patterns):
                clean[k] = "[REDACTED]"
            else:
                clean[k] = sanitize_payload(v)
        return clean
    elif isinstance(obj, (list, tuple)):
        return [sanitize_payload(item) for item in obj]
    return obj


def record_turn(
    turn_id: str,
    thread_id: str,
    turn_number: int,
    target_version: str,
    query: str,
    draft_answer: str,
    is_abstained: bool = False,
    abstention_code: Optional[str] = None,
    abstention_reason: Optional[str] = None,
) -> str:
    """
    Inserts a completed conversation turn into the immutable turns table.
    """
    pool = get_connection_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO turns (
                    turn_id, thread_id, turn_number, target_version,
                    query, draft_answer, is_abstained, abstention_code, abstention_reason
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (turn_id) DO NOTHING
                RETURNING turn_id;
                """,
                (
                    turn_id,
                    thread_id,
                    turn_number,
                    target_version,
                    query,
                    draft_answer,
                    is_abstained,
                    abstention_code,
                    abstention_reason,
                ),
            )
    return turn_id


def record_step(
    turn_id: str,
    thread_id: str,
    step_number: int,
    node: str,
    action: str,
    status: str,
    decided_by: str,
    latency_ms: float,
    abstention_code: Optional[str] = None,
    tokens: Optional[Dict[str, Any]] = None,
    details: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Inserts an immutable record into step_ledger.
    """
    pool = get_connection_pool()
    clean_tokens = sanitize_payload(tokens or {})
    clean_details = sanitize_payload(details or {})

    step_id = str(uuid.uuid4())
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO step_ledger (
                    step_id, turn_id, thread_id, step_number, node,
                    action, status, abstention_code, decided_by,
                    latency_ms, tokens, details
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING step_id;
                """,
                (
                    step_id,
                    turn_id,
                    thread_id,
                    step_number,
                    node,
                    action,
                    status,
                    abstention_code,
                    decided_by,
                    latency_ms,
                    json.dumps(clean_tokens),
                    json.dumps(clean_details),
                ),
            )
    return step_id


def record_claim(
    turn_id: str,
    thread_id: str,
    statement: str,
    chunk_id: str,
    chunk_sha256: str,
    chunk_version: str,
    verifier_verdict: str,
    verifier_code: str,
    claimed_values: Optional[List[str]] = None,
) -> str:
    """
    Inserts an immutable claim verification record into claim_audit.
    """
    pool = get_connection_pool()
    claim_id = str(uuid.uuid4())
    clean_values = sanitize_payload(claimed_values or [])

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO claim_audit (
                    claim_id, turn_id, thread_id, statement,
                    chunk_id, chunk_sha256, chunk_version,
                    claimed_values, verifier_verdict, verifier_code
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING claim_id;
                """,
                (
                    claim_id,
                    turn_id,
                    thread_id,
                    statement,
                    chunk_id,
                    chunk_sha256,
                    chunk_version,
                    json.dumps(clean_values),
                    verifier_verdict,
                    verifier_code,
                ),
            )
    return claim_id


def fetch_turns(thread_id: str) -> List[Dict[str, Any]]:
    """Retrieves all turns for a given thread ordered chronologically."""
    pool = get_connection_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM turns WHERE thread_id = %s ORDER BY turn_number ASC, created_at ASC;",
                (thread_id,),
            )
            return cur.fetchall()


def fetch_steps(thread_id: str, turn_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves step_ledger records for a thread or specific turn."""
    pool = get_connection_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            if turn_id:
                cur.execute(
                    "SELECT * FROM step_ledger WHERE thread_id = %s AND turn_id = %s ORDER BY step_number ASC;",
                    (thread_id, turn_id),
                )
            else:
                cur.execute(
                    "SELECT * FROM step_ledger WHERE thread_id = %s ORDER BY created_at ASC, step_number ASC;",
                    (thread_id,),
                )
            return cur.fetchall()


def fetch_claims(thread_id: str, turn_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves claim_audit records for a thread or specific turn."""
    pool = get_connection_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            if turn_id:
                cur.execute(
                    "SELECT * FROM claim_audit WHERE thread_id = %s AND turn_id = %s ORDER BY created_at ASC;",
                    (thread_id, turn_id),
                )
            else:
                cur.execute(
                    "SELECT * FROM claim_audit WHERE thread_id = %s ORDER BY created_at ASC;",
                    (thread_id,),
                )
            return cur.fetchall()
