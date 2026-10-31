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
from evidenceflow.memory.corpus_settings import get_corpus_settings

logger = logging.getLogger(__name__)


def extract_all_explicit_settings(query: str) -> List[str]:
    """
    Extracts all known PostgreSQL configuration settings explicitly named in the query.
    Matches against setting names discovered dynamically from the indexed corpus.
    Sorts candidate settings by length descending to ensure longer compound names take precedence.
    """
    q = query.lower()
    corpus_settings = get_corpus_settings()
    found: List[str] = []
    # Match longer names first
    for setting in sorted(corpus_settings, key=len, reverse=True):
        pattern = r"\b" + re.escape(setting) + r"\b"
        if re.search(pattern, q):
            # Avoid adding sub-tokens if already covered by longer parameter
            if not any(setting in existing for existing in found):
                found.append(setting)
    return found


def extract_explicit_setting(query: str) -> Optional[str]:
    """
    Checks if the user query explicitly names a known technical setting.
    If found, the explicit setting strictly wins over conversational memory.
    """
    settings = extract_all_explicit_settings(query)
    return settings[0] if settings else None

# Pronoun patterns that trigger follow-up resolution when NO explicit parameter is mentioned
FOLLOW_UP_PRONOUN_PATTERNS = [
    r"\b(it|its|that|this|these|those|their|them|both)\b",
    r"\b(that setting|the setting|that parameter|the parameter|the option|that value)\b",
    r"\b(which of (?:those|these|them)|both (?:settings|parameters|options))\b",
    r"\b(what about|how about|did that change|is that still|was that)\b",
    # Short setting-less questions containing a version or starting with and/in/also/how about/what about/same [Item 2]
    r"^\s*(?:and\s+|also\s+|same\s+|what\s+about\s+|how\s+about\s+)?(?:in\s+)?(?:postgresql\s+|postgres\s+|v)?(1[0-9]|2[0-9])\??\s*$",
]

# Patterns for explicit questions that happen to use "it" or "that" as grammar words
EXPLICIT_NON_FOLLOWUP_PATTERNS = [
    r"\b(is it possible to|can it be|does it mean that|how is it that)\b",
    r"\b(setting that controls|parameter that defines|option that sets)\b",
    r"\b(in its documentation|on its website|its source code)\b",
]



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
    abstention_reason: Optional[str] = None,
) -> str:
    """
    Persists a conversation turn into Neo4j in a single ACID transaction
    using discrete, parameterized Cypher statements.

    Guarantees:
      - Uses identical turn_id, thread_id, turn_number as PostgreSQL storage.ledger.
      - An empty entities list or missing/invalid Version NEVER skips Claim write.
      - Stores a turn's setting entities when the setting exists in the corpus,
        even if the answer step refused for a verifier/writer reason [Rule g].
      - Skips entities for MISSING_KEYWORD, INJECTION, OUT_OF_SCOPE, and UNSUPPORTED_VERSION.
      - Preserves refusal reasons for visibility on subsequent turns.
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
                    t.abstention_reason = $abstention_reason,
                    t.created_at = datetime()
                """,
                {
                    "turn_id": turn_id,
                    "thread_id": thread_id,
                    "turn_number": turn_number,
                    "is_abstained": is_abstained,
                    "abstention_code": abstention_code,
                    "abstention_reason": abstention_reason,
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

            # 5. Entity Nodes & Edges [Rule g]:
            # Store a turn's setting entities when the setting exists in the corpus,
            # even if the answer step refused for a verifier/writer reason.
            # Keep skipping entities for MISSING_KEYWORD, INJECTION and OUT_OF_SCOPE.
            skip_entity_storage = is_abstained and abstention_code in {
                "INJECTION", "MISSING_KEYWORD", "OUT_OF_SCOPE", "UNSUPPORTED_VERSION"
            }
            if not skip_entity_storage:
                corpus_settings = get_corpus_settings()
                for ent in entities:
                    ent_name = ent.get("name", "").strip().lower()
                    ent_type = ent.get("type", "ConfigParam")
                    if ent_name and (not corpus_settings or ent_name in corpus_settings):
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


def safe_substitute_pronoun(query: str, entity_name: str, target_version: Optional[str] = None) -> Optional[str]:
    """
    Substitutes anaphoric pronouns with the resolved entity name while preserving
    the grammatical intent and phrasing of the user's original query.
    If safe substitution is not possible, returns None to signal a clarifying question.
    """
    q = query.strip()

    # 1. Standalone version switch queries: "in v14?", "what about 15?", "and in 16?", "same in 15?" [Item 2]
    m_same_ver = re.match(
        r"^\s*same\s+(?:in\s+)?(?:postgresql\s+|postgres\s+|v)?(1[0-9]|2[0-9])\??\s*$",
        q,
        re.IGNORECASE,
    )
    if m_same_ver:
        ver = m_same_ver.group(1)
        return f"Is {entity_name} the same in PostgreSQL {ver}?"

    m_ver_switch = re.match(
        r"^\s*(?:and\s+|also\s+|what\s+about\s+|how\s+about\s+)?(?:in\s+)?(?:postgresql\s+|postgres\s+|v)?(1[0-9]|2[0-9])\??\s*$",
        q,
        re.IGNORECASE,
    )
    if m_ver_switch:
        ver = m_ver_switch.group(1)
        return f"What is the configuration and behavior of {entity_name} in PostgreSQL {ver}?"

    substitutions = [
        # Multi-word parameter references
        (r"\b(that\s+setting|the\s+setting|this\s+setting)\b", entity_name),
        (r"\b(that\s+parameter|the\s+parameter|this\s+parameter)\b", entity_name),
        (r"\b(that\s+option|the\s+option|this\s+option)\b", entity_name),
        (r"\b(that\s+value|the\s+value)\b", f"the value of {entity_name}"),
        # Possessive pronoun
        (r"\b(its)\b", f"{entity_name}'s"),
        # Singular pronouns
        (r"\b(it|that|this)\b", entity_name),
    ]

    new_q = q
    replaced = False

    for pattern, replacement in substitutions:
        if re.search(pattern, new_q, flags=re.IGNORECASE):
            new_q = re.sub(pattern, replacement, new_q, count=1, flags=re.IGNORECASE)
            replaced = True
            break

    if not replaced:
        return None

    # If rewritten query lacks PostgreSQL version mention and target_version is known, append cleanly
    # Normalize short version tokens like v16 -> PostgreSQL 16
    new_q = re.sub(r"\bv(14|15|16)\b", r"PostgreSQL \1", new_q, flags=re.IGNORECASE)
    if target_version and not re.search(r"\b(?:v|version\s+|postgresql\s+)?(?:14|15|16)\b", new_q, re.IGNORECASE):
        if new_q.endswith("?"):
            new_q = f"{new_q[:-1]} in PostgreSQL {target_version}?"
        else:
            new_q = f"{new_q} in PostgreSQL {target_version}"

    return new_q


def resolve_entity_for_turn(
    thread_id: str,
    follow_up_query: str,
    force_resolution: bool = False,
) -> Dict[str, Any]:
    """
    Resolves multi-turn follow-up queries using graph conversational memory.

    Rules [Req 4, b, d]:
      1. Explicit setting or version in the new question wins over memory.
      2. If turn 2 has no version flag and mentions none, carries version from turn 1.
      3. Rewrites by safely substituting pronoun with setting name, preserving question text.
      4. If the prior turn was a refusal, returns status='ambiguous' with clarifying question.
      5. If the prior turn discussed multiple entities, returns status='ambiguous' with clarifying question.
      6. If single entity resolved, returns status='resolved' with rewritten query.

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

    # Query Neo4j for the most recent turn in this thread to get prior version / entities
    record = None
    try:
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
                       t.abstention_reason AS abstention_reason,
                       v.tag AS prior_version,
                       collect(DISTINCT e.name) AS entities
                """,
                {"thread_id": thread_id},
            )
            record = result.single()
    except Exception as e:
        logger.warning("Failed to query Neo4j for prior turn: %s", e)

    prior_version = record["prior_version"] if record and record["prior_version"] else "16"
    target_version = explicit_version or prior_version

    # 1. Rule 1: Explicit setting in new question strictly wins over memory
    if explicit_setting is not None:
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "status": "passthrough",
            "reason": "explicit_setting_in_query",
            "entity_name": explicit_setting,
            "target_version": target_version,
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
            "target_version": target_version,
            "rewritten_query": follow_up_query,
            "latency_ms": round(latency_ms, 3),
        }

    latency_ms = (time.perf_counter() - t_start) * 1000.0

    # No prior turns found in memory
    if not record:
        return {
            "status": "ambiguous",
            "reason": "no_prior_history",
            "clarification": "I couldn't identify the parameter you are referring to. Which setting would you like to investigate?",
            "latency_ms": round(latency_ms, 3),
        }

    # 3. Rule 3: Prior turn was an abstention / refusal [Rule g]
    if record["is_abstained"] and record.get("abstention_code") in {
        "INJECTION", "MISSING_KEYWORD", "OUT_OF_SCOPE", "UNSUPPORTED_VERSION"
    }:
        return {
            "status": "ambiguous",
            "reason": "prior_turn_refusal",
            "clarification": f"The previous question could not be answered from the documentation (refused: {record.get('abstention_code')}). Which parameter would you like to investigate?",
            "latency_ms": round(latency_ms, 3),
        }

    prior_entities = [e for e in record["entities"] if e]

    # 4. Rule 4 & Rule h: Multiple entities in prior turn
    if len(prior_entities) > 1:
        options = " and ".join([f"'{e}'" for e in prior_entities])
        is_plural = any(re.search(pat, follow_up_query.lower()) for pat in [
            r"\b(those|these|both|them|their)\b",
            r"\b(which of (?:those|these|them)|both of (?:those|these|them))\b",
        ])
        if is_plural:
            return {
                "status": "ambiguous",
                "reason": "multiple_entities_plural",
                "clarification": f"The previous question discussed multiple parameters ({options}). Would you like me to check both or focus on a specific one?",
                "candidate_entities": prior_entities,
                "latency_ms": round(latency_ms, 3),
            }
        else:
            return {
                "status": "ambiguous",
                "reason": "multiple_entities",
                "clarification": f"The previous question discussed multiple parameters ({options}). Which one did you mean?",
                "candidate_entities": prior_entities,
                "latency_ms": round(latency_ms, 3),
            }

    # 5. No entities recorded on prior turn
    if len(prior_entities) == 0:
        return {
            "status": "ambiguous",
            "reason": "no_entity_in_prior_turn",
            "clarification": "I couldn't determine which parameter you are referring to. Which setting would you like to check?",
            "latency_ms": round(latency_ms, 3),
        }

    # 6. Rule 5 & Rule b: Exactly one entity resolved, safe grammatical substitution
    resolved_entity = prior_entities[0]
    rewritten_query = safe_substitute_pronoun(follow_up_query, resolved_entity, target_version)

    if not rewritten_query:
        return {
            "status": "ambiguous",
            "reason": "unsafe_substitution",
            "clarification": f"I couldn't safely resolve which aspect of '{resolved_entity}' you are inquiring about. Could you rephrase your question?",
            "candidate_entities": [resolved_entity],
            "latency_ms": round(latency_ms, 3),
        }

    return {
        "status": "resolved",
        "reason": "single_entity_resolved",
        "entity_name": resolved_entity,
        "target_version": target_version,
        "prior_version": record["prior_version"],
        "prior_turn_number": record["turn_number"],
        "prior_is_abstained": record["is_abstained"],
        "prior_abstention_code": record.get("abstention_code"),
        "prior_abstention_reason": record.get("abstention_reason"),
        "rewritten_query": rewritten_query,
        "latency_ms": round(latency_ms, 3),
    }
