"""
migrate_postgres.py: Database migration runner for EvidenceFlow AI.
Initializes LangGraph PostgresSaver checkpointer tables, audit ledgers,
and enforces append-only triggers.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional
import psycopg
from langgraph.checkpoint.postgres import PostgresSaver

from evidenceflow.config import get_settings

logger = logging.getLogger(__name__)


def get_postgres_connection_string() -> str:
    """Builds standard psycopg connection string from Pydantic settings."""
    s = get_settings()
    return f"host={s.POSTGRES_HOST} port={s.POSTGRES_PORT} user={s.POSTGRES_USER} password={s.POSTGRES_PASSWORD} dbname={s.POSTGRES_DB}"


def run_migrations(migrations_dir: Optional[Path] = None) -> List[str]:
    """
    Executes LangGraph checkpointer setup and all sequential SQL migrations.
    Returns list of executed migration filenames.
    """
    conn_str = get_postgres_connection_string()
    applied = []

    # 1. Setup LangGraph PostgresSaver checkpointer tables (requires autocommit=True)
    logger.info("Initializing LangGraph checkpointer schema...")
    with psycopg.connect(conn_str, autocommit=True) as conn:
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()
        applied.append("langgraph_checkpointer_setup")

    # 2. Run custom SQL migrations
    repo_root = Path(__file__).resolve().parents[3]
    m_dir = migrations_dir or (repo_root / "migrations")

    if not m_dir.exists():
        logger.warning("Migrations directory not found at %s", m_dir)
        return applied

    sql_files = sorted(m_dir.glob("*.sql"))
    with psycopg.connect(conn_str, autocommit=True) as conn:
        with conn.cursor() as cur:
            for sql_file in sql_files:
                logger.info("Applying migration: %s", sql_file.name)
                sql_content = sql_file.read_text(encoding="utf-8")
                cur.execute(sql_content)
                applied.append(sql_file.name)

    logger.info("All migrations applied successfully! Total: %d", len(applied))
    return applied


def drop_all_tables():
    """Drops custom audit tables and checkpointer tables (for clean test isolation)."""
    conn_str = get_postgres_connection_string()
    with psycopg.connect(conn_str, autocommit=True) as conn:
        with conn.cursor() as cur:
            # Drop custom tables
            cur.execute("""
                DROP TABLE IF EXISTS claim_audit CASCADE;
                DROP TABLE IF EXISTS step_ledger CASCADE;
                DROP TABLE IF EXISTS turns CASCADE;
                DROP FUNCTION IF EXISTS enforce_append_only() CASCADE;
                DROP FUNCTION IF EXISTS enforce_no_truncate() CASCADE;
            """)
            # Drop checkpointer tables
            cur.execute("""
                DROP TABLE IF EXISTS checkpoint_writes CASCADE;
                DROP TABLE IF EXISTS checkpoint_blobs CASCADE;
                DROP TABLE IF EXISTS checkpoints CASCADE;
                DROP TABLE IF EXISTS checkpoint_migrations CASCADE;
            """)
    logger.info("All EvidenceFlow PostgreSQL tables dropped.")


def main():
    parser = argparse.ArgumentParser(description="Run EvidenceFlow PostgreSQL migrations.")
    parser.add_argument("--drop", action="store_true", help="Drop all tables before running migrations")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.drop:
        logger.warning("Dropping all existing tables...")
        drop_all_tables()

    run_migrations()


if __name__ == "__main__":
    main()
