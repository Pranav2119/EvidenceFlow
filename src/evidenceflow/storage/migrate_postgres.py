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


def get_postgres_connection_string(dbname: Optional[str] = None) -> str:
    """Builds standard psycopg connection string from Pydantic settings."""
    s = get_settings()
    db = dbname or s.POSTGRES_DB
    return f"host={s.POSTGRES_HOST} port={s.POSTGRES_PORT} user={s.POSTGRES_USER} password={s.POSTGRES_PASSWORD} dbname={db}"


def ensure_database_exists(dbname: str):
    """Creates the target PostgreSQL database if it does not already exist."""
    s = get_settings()
    admin_conn_str = f"host={s.POSTGRES_HOST} port={s.POSTGRES_PORT} user={s.POSTGRES_USER} password={s.POSTGRES_PASSWORD} dbname=postgres"
    with psycopg.connect(admin_conn_str, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (dbname,))
            if not cur.fetchone():
                logger.info("Creating target database '%s'...", dbname)
                cur.execute(f'CREATE DATABASE "{dbname}";')


def run_migrations(migrations_dir: Optional[Path] = None, dbname: Optional[str] = None) -> List[str]:
    """
    Executes LangGraph checkpointer setup and all sequential SQL migrations.
    Returns list of executed migration filenames.
    """
    s = get_settings()
    target_db = dbname or s.POSTGRES_DB
    conn_str = get_postgres_connection_string(dbname=target_db)
    applied = []

    # 1. Setup LangGraph PostgresSaver checkpointer tables (requires autocommit=True)
    logger.info("Initializing LangGraph checkpointer schema on database '%s'...", target_db)
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
                logger.info("Applying migration: %s on database '%s'", sql_file.name, target_db)
                sql_content = sql_file.read_text(encoding="utf-8")
                cur.execute(sql_content)
                applied.append(sql_file.name)

    logger.info("All migrations applied successfully on '%s'! Total: %d", target_db, len(applied))
    return applied


def drop_all_tables(dbname: Optional[str] = None):
    """
    Drops custom audit tables and checkpointer tables (for clean test isolation).
    Strictly refuses to run unless the target database name AND the actual connected
    database (verified via SELECT current_database()) end with '_test'.
    """
    s = get_settings()
    target_db = dbname or s.POSTGRES_DB
    if not target_db.endswith("_test"):
        raise RuntimeError(
            f"Safety Violation: drop_all_tables() target dbname '{target_db}' does not end in '_test'."
        )

    conn_str = get_postgres_connection_string(dbname=target_db)
    with psycopg.connect(conn_str, autocommit=True) as conn:
        with conn.cursor() as cur:
            # Engine-level safety verification: query current_database() directly from PostgreSQL session
            cur.execute("SELECT current_database();")
            row = cur.fetchone()
            actual_connected_db = str(row[0]) if row else ""
            if not actual_connected_db.endswith("_test"):
                raise RuntimeError(
                    f"Safety Violation: Connected database '{actual_connected_db}' does not end in '_test'. "
                    f"drop_all_tables() aborted to protect non-test data."
                )

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
    logger.info("All EvidenceFlow PostgreSQL tables dropped on test database '%s'.", actual_connected_db)


def main():
    parser = argparse.ArgumentParser(description="Run EvidenceFlow PostgreSQL migrations.")
    parser.add_argument("--dbname", type=str, default=None, help="Target PostgreSQL database name")
    parser.add_argument("--drop", action="store_true", help="Drop all tables before running migrations (requires database ending in '_test')")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.drop:
        logger.warning("Dropping tables on target database...")
        drop_all_tables(dbname=args.dbname)

    run_migrations(dbname=args.dbname)


if __name__ == "__main__":
    main()
