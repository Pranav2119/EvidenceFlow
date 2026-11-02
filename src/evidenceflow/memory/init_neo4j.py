"""
init_neo4j.py: Neo4j schema constraint and index initialization script.
Creates uniqueness constraints and performance indexes for conversational memory graph.
"""

import logging
from typing import List, Optional
from evidenceflow.memory.connection import get_neo4j_session, close_neo4j_driver

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evidenceflow.memory.init")

SCHEMA_STATEMENTS: List[str] = [
    # 1. Uniqueness constraint: Entity names must be unique and canonical (e.g., 'wal_compression')
    "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",

    # 2. Uniqueness constraint: Documentation version tags ('14', '15', '16')
    "CREATE CONSTRAINT version_tag_unique IF NOT EXISTS FOR (v:Version) REQUIRE v.tag IS UNIQUE",

    # 3. Uniqueness constraint: Passage IDs (e.g., 'pg16-runtime-config-wal-wal-compression')
    "CREATE CONSTRAINT passage_id_unique IF NOT EXISTS FOR (p:Passage) REQUIRE p.passage_id IS UNIQUE",

    # 4. Uniqueness constraint: Turn IDs (matches Postgres turns.turn_id UUID)
    "CREATE CONSTRAINT turn_id_unique IF NOT EXISTS FOR (t:Turn) REQUIRE t.id IS UNIQUE",

    # 5. Uniqueness constraint: Question IDs (Requirement 2)
    "CREATE CONSTRAINT question_id_unique IF NOT EXISTS FOR (q:Question) REQUIRE q.id IS UNIQUE",

    # 6. Uniqueness constraint: Claim IDs (Requirement 2)
    "CREATE CONSTRAINT claim_id_unique IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE",

    # 7. Composite index for fast thread history lookups by thread_id and turn_number (Requirement 6)
    "CREATE INDEX turn_thread_idx IF NOT EXISTS FOR (t:Turn) ON (t.thread_id, t.turn_number)",

    # 8. Index for fast question thread lookups
    "CREATE INDEX question_thread_idx IF NOT EXISTS FOR (q:Question) ON (q.thread_id)",
]

SEED_VERSIONS: List[str] = ["14", "15", "16"]


def init_neo4j_schema() -> None:
    """
    Executes schema constraints and indexes, then seeds canonical Version nodes.
    """
    logger.info("Applying Neo4j schema constraints and indexes...")
    with get_neo4j_session() as session:
        for stmt in SCHEMA_STATEMENTS:
            logger.info("Executing Cypher: %s", stmt)
            session.run(stmt)

        logger.info("Seeding canonical documentation Version nodes (%s)...", SEED_VERSIONS)
        for version in SEED_VERSIONS:
            session.run(
                """
                MERGE (v:Version {tag: $tag})
                ON CREATE SET v.name = 'PostgreSQL ' + $tag, v.created_at = datetime()
                """,
                {"tag": version},
            )

    logger.info("Neo4j schema and initial seeds applied successfully.")


def verify_schema() -> List[str]:
    """
    Queries Neo4j system metadata and returns active constraint and index names.
    """
    constraints_and_indexes = []
    with get_neo4j_session() as session:
        result = session.run("SHOW CONSTRAINTS")
        for record in result:
            constraints_and_indexes.append(f"CONSTRAINT: {record.get('name')} ON {record.get('labelsOrTypes')}")

        result = session.run("SHOW INDEXES")
        for record in result:
            constraints_and_indexes.append(f"INDEX: {record.get('name')} ON {record.get('labelsOrTypes')}")
    return constraints_and_indexes


def drop_graph(thread_id: Optional[str] = None) -> None:
    """
    Helper function to clear graph nodes.
    If thread_id is provided, only deletes turns, questions, and claims for that thread.
    If thread_id is None, clears all nodes (used in teardown/testing).
    """
    with get_neo4j_session() as session:
        if thread_id:
            logger.warning("Clearing memory nodes for thread %s", thread_id)
            session.run(
                """
                MATCH (t:Turn {thread_id: $thread_id})
                OPTIONAL MATCH (t)-[:HAS_QUESTION]->(q:Question)
                OPTIONAL MATCH (t)-[:PRODUCED]->(c:Claim)
                DETACH DELETE t, q, c
                """,
                {"thread_id": thread_id},
            )
        else:
            logger.warning("DROPPING ALL NODES IN NEO4J GRAPH!")
            session.run("MATCH (n) DETACH DELETE n")


if __name__ == "__main__":
    try:
        init_neo4j_schema()
        active = verify_schema()
        print("\n--- Active Neo4j Schema Items ---")
        for item in active:
            print(f"  * {item}")
    finally:
        close_neo4j_driver()
