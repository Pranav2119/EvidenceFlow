"""
graph_client.py: Cypher query interface for conversational memory graph in Neo4j.
Manages multi-turn dialogue state, entity tracking, and pronoun follow-up resolution.

Enforces:
  [Req 2] Uniqueness constraints for Turn.id, Question.id, Claim.id, Entity.name, Version.tag.
  [Req 3] Discrete parameterized queries inside a single ACID transaction.
          Empty entities or missing Version never skips Claim write.
  [Req 4] Explicit setting/version in new question wins over memory.
          Only answered turns store active entities.
          Refusals or multi-entity turns return 'ambiguous' with clarifying question.
  [Req 5] Memory ONLY rewrites the query. Answers strictly cite Qdrant corpus chunks.
  [Req 6] thread_id, turn_id, turn_number match PostgreSQL storage.ledger schema.
"""

import re
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from evidenceflow.memory.connection import get_neo4j_session

logger = logging.getLogger(__name__)

# Known PostgreSQL GUC configuration parameters indexed in our corpus
# Used to determine if the user's new question explicitly names a real setting
KNOWN_POSTGRES_SETTINGS = {
    "wal_compression", "max_connections", "shared_buffers", "work_mem",
    "maintenance_work_mem", "checkpoint_timeout", "max_wal_size", "min_wal_size",
    "checkpoint_completion_target", "autovacuum_naptime", "autovacuum_max_workers",
    "autovacuum_vacuum_cost_limit", "autovacuum_vacuum_cost_delay", "log_min_duration_statement",
    "log_checkpoints", "log_connections", "log_disconnections", "archive_mode",
    "archive_command", "archive_timeout", "vacuum_buffer_usage_limit",
    "vacuum_defer_cleanup_age", "ignore_system_indexes", "zero_damaged_pages",
    "fsync", "synchronous_commit", "wal_level", "max_worker_processes",
    "max_parallel_workers", "max_parallel_maintenance_workers", "deadlock_timeout",
    "statement_timeout", "lock_timeout", "idle_in_transaction_session_timeout",
    "track_io_timing", "track_activities", "track_counts", "listen_addresses", "port",
}

# Pronoun patterns that trigger follow-up resolution when NO explicit parameter is mentioned
FOLLOW_UP_PRONOUN_PATTERNS = [
    r"\b(it|that|this|these|those)\b",
    r"\b(that setting|the setting|that parameter|the parameter|the option|that value)\b",
    r"\b(what about|how about|did that change|is that still|was that)\b",
    r"^\s*(what about\s+|how about\s+)?in v?(14|15|16)\??\s*$",
]

# Patterns for explicit questions that happen to use "it" or "that" as grammar words
EXPLICIT_NON_FOLLOWUP_PATTERNS = [
    r"\b(is it possible to|can it be|does it mean that|how is it that)\b",
    r"\b(setting that controls|parameter that defines|option that sets)\b",
]


def extract_explicit_setting(query: str) -> Optional[str]:
    """
    Checks if the user query explicitly names a known technical setting.
    If found, the explicit setting strictly wins over conversational memory.
    """
    q = query.lower()
    for setting in KNOWN_POSTGRES_SETTINGS:
        pattern = r"\b" + re.escape(setting) + r"\b"
        if re.search(pattern, q):
            return setting
    return None


def extract_explicit_version(query: str) -> Optional[str]:
    """
    Extracts an explicit PostgreSQL version target from the query ('14', '15', '16').
    """
    q = query.lower()
    match = re.search(r"\b(?:postgresql\s+|postgres\s+|v)?(14|15|16)\b", q)
    if match:
        return match.group(1)
    return None


def is_follow_up_query(query: str) -> bool:
    """
    Determines whether a query is an anaphoric follow-up referencing prior context.
    Returns False if an explicit setting is named, or if 'it'/'that' is part of an
    explicit non-followup syntactic construction.
    """
    q = query.lower().strip()

    # If the user explicitly named a known parameter, it is not an ambiguous follow-up
    if extract_explicit_setting(q) is not None:
        return False

    # Check for false-trigger grammatical phrases like "setting that controls"
    if any(re.search(pat, q) for pat in EXPLICIT_NON_FOLLOWUP_PATTERNS):
        return False

    # Check for genuine pronoun / version-switch follow-ups
    return any(re.search(pat, q) for pat in FOLLOW_UP_PRONOUN_PATTERNS)


def record_turn(
    turn_id: str,
    thread_id: str,
    turn_number: int,
    question: str,
    version_scope: Optional[str] = None,
    entities: Optional[List[Dict[str, str]]] = None,
    claims: Optional[List[Dict[str, Any]]] = None,
    is_abstained: bool = False,
    abstention_code: Optional[str] = None,
) -> str:
    """
    Persists a conversation turn into Neo4j in a single ACID transaction
    using discrete, parameterized Cypher statements.

    Guarantees:
      - Uses identical turn_id, thread_id, turn_number as PostgreSQL storage.ledger.
      - An empty entities list or missing/invalid Version NEVER skips Claim write.
      - If is_abstained is True, entities are NOT linked as active conversation memory.
      - Zero string-formatted Cypher; all inputs passed as parameters.
    """
    entities = entities or []
    claims = claims or []
    question_id = f"{turn_id}_q"
    version_tag = str(version_scope).strip() if version_scope else None

    with get_neo4j_session() as session:
        with session.begin_transaction() as tx:
            # 1. Turn Node
            tx.run(
                """
                MERGE (t:Turn {id: $turn_id})
                SET t.thread_id = $thread_id,
                    t.turn_number = $turn_number,
                    t.is_abstained = $is_abstained,
                    t.abstention_code = $abstention_code,
                    t.created_at = datetime()
                """,
                {
                    "turn_id": turn_id,
                    "thread_id": thread_id,
                    "turn_number": turn_number,
                    "is_abstained": is_abstained,
                    "abstention_code": abstention_code,
                },
            )

            # 2. Chronological FOLLOWS_UP edge (if turn_number > 1)
            if turn_number > 1:
                tx.run(
                    """
                    MATCH (t:Turn {id: $turn_id})
                    MATCH (prev:Turn {thread_id: $thread_id, turn_number: $prev_turn_number})
                    MERGE (t)-[:FOLLOWS_UP]->(prev)
                    """,
                    {
                        "turn_id": turn_id,
                        "thread_id": thread_id,
                        "prev_turn_number": turn_number - 1,
                    },
                )

            # 3. Question Node
            tx.run(
                """
                MATCH (t:Turn {id: $turn_id})
                MERGE (q:Question {id: $question_id})
                SET q.text = $question,
                    q.thread_id = $thread_id,
                    q.turn_number = $turn_number,
                    q.created_at = datetime()
                MERGE (t)-[:HAS_QUESTION]->(q)
                """,
                {
                    "turn_id": turn_id,
                    "question_id": question_id,
                    "question": question,
                    "thread_id": thread_id,
                    "turn_number": turn_number,
                },
            )

            # 4. Version Edge (if version exists; failure to match version never breaks tx)
            if version_tag:
                tx.run(
                    """
                    MATCH (q:Question {id: $question_id})
                    OPTIONAL MATCH (v:Version {tag: $version_tag})
                    FOREACH (_ IN CASE WHEN v IS NOT NULL THEN [1] ELSE [] END |
                        MERGE (q)-[:SCOPED_TO]->(v)
                    )
                    """,
                    {"question_id": question_id, "version_tag": version_tag},
                )

            # 5. Entity Nodes & Edges (Only stored as active memory if turn was NOT an abstention)
            if not is_abstained:
                for ent in entities:
                    ent_name = ent.get("name", "").strip().lower()
                    ent_type = ent.get("type", "ConfigParam")
                    if ent_name:
                        tx.run(
                            """
                            MATCH (t:Turn {id: $turn_id})
                            MATCH (q:Question {id: $question_id})
                            MERGE (e:Entity {name: $name})
                            ON CREATE SET e.type = $type
                            MERGE (t)-[:CONCERNS]->(e)
                            MERGE (q)-[:CONCERNS]->(e)
                            """,
                            {
                                "turn_id": turn_id,
                                "question_id": question_id,
                                "name": ent_name,
                                "type": ent_type,
                            },
                        )

            # 6. Claim Nodes & Edges (Must run even if entities was empty or version was missing)
            for idx, claim in enumerate(claims):
                claim_id = claim.get("id") or f"{turn_id}_c{idx + 1}"
                claim_text = claim.get("text", "")
                verdict = claim.get("verdict", "VERIFIED")
                passage_id = claim.get("passage_id")
                chunk_hash = claim.get("chunk_hash")

                tx.run(
                    """
                    MATCH (t:Turn {id: $turn_id})
                    MERGE (c:Claim {id: $claim_id})
                    SET c.text = $claim_text,
                        c.verdict = $verdict,
                        c.created_at = datetime()
                    MERGE (t)-[:PRODUCED]->(c)
                    """,
                    {
                        "turn_id": turn_id,
                        "claim_id": claim_id,
                        "claim_text": claim_text,
                        "verdict": verdict,
                    },
                )

                if passage_id:
                    tx.run(
                        """
                        MATCH (c:Claim {id: $claim_id})
                        MERGE (p:Passage {passage_id: $passage_id})
                        ON CREATE SET p.hash = $chunk_hash
                        MERGE (c)-[:SUPPORTED_BY]->(p)
                        """,
                        {
                            "claim_id": claim_id,
                            "passage_id": passage_id,
                            "chunk_hash": chunk_hash or "",
                        },
                    )
                    if version_tag:
                        tx.run(
                            """
                            MATCH (p:Passage {passage_id: $passage_id})
                            OPTIONAL MATCH (v:Version {tag: $version_tag})
                            FOREACH (_ IN CASE WHEN v IS NOT NULL THEN [1] ELSE [] END |
                                MERGE (p)-[:BELONGS_TO]->(v)
                            )
                            """,
                            {"passage_id": passage_id, "version_tag": version_tag},
                        )

            tx.commit()

    return turn_id


def get_session_history(thread_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves the chronological thread of turns for a thread_id,
    including questions, version scopes, entities, and claims.
    """
    with get_neo4j_session() as session:
        result = session.run(
            """
            MATCH (t:Turn {thread_id: $thread_id})
            OPTIONAL MATCH (t)-[:HAS_QUESTION]->(q:Question)
            OPTIONAL MATCH (q)-[:SCOPED_TO]->(v:Version)
            OPTIONAL MATCH (t)-[:CONCERNS]->(e:Entity)
            OPTIONAL MATCH (t)-[:PRODUCED]->(c:Claim)
            OPTIONAL MATCH (t)-[:FOLLOWS_UP]->(prev:Turn)
            RETURN t.id AS turn_id,
                   t.turn_number AS turn_number,
                   t.is_abstained AS is_abstained,
                   t.abstention_code AS abstention_code,
                   q.text AS question,
                   v.tag AS version,
                   prev.id AS follows_up_turn_id,
                   collect(DISTINCT e.name) AS entities,
                   collect(DISTINCT {id: c.id, text: c.text, verdict: c.verdict}) AS claims
            ORDER BY t.turn_number ASC
            """,
            {"thread_id": thread_id},
        )
        history = []
        for r in result:
            history.append({
                "turn_id": r["turn_id"],
                "turn_number": r["turn_number"],
                "is_abstained": r["is_abstained"],
                "abstention_code": r["abstention_code"],
                "question": r["question"],
                "version": r["version"],
                "follows_up_turn_id": r["follows_up_turn_id"],
                "entities": [e for e in r["entities"] if e],
                "claims": [c for c in r["claims"] if c.get("id")],
            })
        return history


def resolve_entity_for_turn(
    thread_id: str,
    follow_up_query: str,
    force_resolution: bool = False,
) -> Dict[str, Any]:
    """
    Resolves multi-turn follow-up queries using graph conversational memory.

    Rules [Req 4]:
      1. Explicit setting or version in the new question wins over memory.
      2. If not a follow-up query, returns status='passthrough' (0 memory modification).
      3. If the prior turn was a refusal, returns status='ambiguous' with clarifying question.
      4. If the prior turn discussed multiple entities, returns status='ambiguous' with clarifying question.
      5. If single entity resolved, returns status='resolved' with rewritten query.

    Returns Dict with:
      - status: 'resolved' | 'ambiguous' | 'passthrough'
      - rewritten_query: self-contained rewritten query (if resolved)
      - entity_name: resolved entity (if resolved)
      - target_version: target version
      - clarification: clarifying question string (if ambiguous)
      - latency_ms: measured execution duration in milliseconds
    """
    t_start = time.perf_counter()
    explicit_setting = extract_explicit_setting(follow_up_query)
    explicit_version = extract_explicit_version(follow_up_query)

    # 1. Rule 1: Explicit setting in new question strictly wins over memory
    if explicit_setting is not None:
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "status": "passthrough",
            "reason": "explicit_setting_in_query",
            "entity_name": explicit_setting,
            "target_version": explicit_version,
            "rewritten_query": follow_up_query,
            "latency_ms": round(latency_ms, 3),
        }

    # 2. Check if this is a genuine pronoun follow-up
    if not force_resolution and not is_follow_up_query(follow_up_query):
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "status": "passthrough",
            "reason": "not_a_follow_up",
            "entity_name": None,
            "target_version": explicit_version,
            "rewritten_query": follow_up_query,
            "latency_ms": round(latency_ms, 3),
        }

    # 3. Query Neo4j for the most recent turn in this thread
    with get_neo4j_session() as session:
        result = session.run(
            """
            MATCH (t:Turn {thread_id: $thread_id})
            WITH t ORDER BY t.turn_number DESC LIMIT 1
            OPTIONAL MATCH (t)-[:CONCERNS]->(e:Entity)
            OPTIONAL MATCH (t)-[:HAS_QUESTION]->(q:Question)-[:SCOPED_TO]->(v:Version)
            RETURN t.id AS turn_id,
                   t.turn_number AS turn_number,
                   coalesce(t.is_abstained, false) AS is_abstained,
                   t.abstention_code AS abstention_code,
                   v.tag AS prior_version,
                   collect(DISTINCT e.name) AS entities
            """,
            {"thread_id": thread_id},
        )
        record = result.single()

    latency_ms = (time.perf_counter() - t_start) * 1000.0

    # No prior turns found in memory
    if not record:
        return {
            "status": "ambiguous",
            "reason": "no_prior_history",
            "clarification": "I couldn't identify the parameter you are referring to. Which setting would you like to investigate?",
            "latency_ms": round(latency_ms, 3),
        }

    # 4. Rule 3: Prior turn was an abstention / refusal
    if record["is_abstained"]:
        return {
            "status": "ambiguous",
            "reason": "prior_turn_refusal",
            "clarification": "The previous question could not be answered from the documentation. Which parameter would you like to investigate?",
            "latency_ms": round(latency_ms, 3),
        }

    prior_entities = [e for e in record["entities"] if e]

    # 5. Rule 4: Multiple entities in prior turn
    if len(prior_entities) > 1:
        options = " or ".join([f"'{e}'" for e in prior_entities])
        return {
            "status": "ambiguous",
            "reason": "multiple_entities",
            "clarification": f"The previous question discussed multiple parameters ({options}). Which one did you mean?",
            "candidate_entities": prior_entities,
            "latency_ms": round(latency_ms, 3),
        }

    # 6. No entities recorded on prior turn
    if len(prior_entities) == 0:
        return {
            "status": "ambiguous",
            "reason": "no_entity_in_prior_turn",
            "clarification": "I couldn't determine which parameter you are referring to. Which setting would you like to check?",
            "latency_ms": round(latency_ms, 3),
        }

    # 7. Rule 5: Exactly one entity resolved!
    resolved_entity = prior_entities[0]
    target_version = explicit_version or record["prior_version"] or "16"

    # Construct the canonical self-contained query for downstream pipeline
    rewritten_query = f"What is the configuration and behavior of {resolved_entity} in PostgreSQL {target_version}?"

    return {
        "status": "resolved",
        "reason": "single_entity_resolved",
        "entity_name": resolved_entity,
        "target_version": target_version,
        "prior_version": record["prior_version"],
        "prior_turn_number": record["turn_number"],
        "rewritten_query": rewritten_query,
        "latency_ms": round(latency_ms, 3),
    }
