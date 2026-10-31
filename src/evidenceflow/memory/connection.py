"""
connection.py: Neo4j database driver lifecycle and connection management.
Provides thread-safe connection pooling, health checks, and session management.
"""

import logging
from contextlib import contextmanager
from typing import Generator, Optional
from neo4j import GraphDatabase, Driver, Session
from evidenceflow.config import get_settings

logger = logging.getLogger(__name__)

# Module-level singleton driver instance
_neo4j_driver: Optional[Driver] = None


def get_neo4j_driver() -> Driver:
    """
    Returns the singleton Neo4j driver instance.
    Initializes the driver on first call with connection pool settings.
    """
    global _neo4j_driver
    if _neo4j_driver is None:
        settings = get_settings()
        logger.info("Initializing Neo4j Bolt driver connection to %s", settings.NEO4J_URI)
        _neo4j_driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            max_connection_lifetime=30 * 60,  # 30 minutes
            max_connection_pool_size=50,
            connection_acquisition_timeout=10.0,  # 10s timeout
        )
    return _neo4j_driver


def close_neo4j_driver() -> None:
    """
    Gracefully closes all connections in the Neo4j driver pool.
    """
    global _neo4j_driver
    if _neo4j_driver is not None:
        logger.info("Closing Neo4j Bolt driver connection pool.")
        _neo4j_driver.close()
        _neo4j_driver = None


def verify_neo4j_connectivity() -> bool:
    """
    Performs an active connectivity ping against the Neo4j cluster.
    Returns True if reachable and authenticated, False otherwise.
    """
    try:
        driver = get_neo4j_driver()
        driver.verify_connectivity()
        return True
    except Exception as e:
        logger.error("Neo4j connectivity check failed: %s", e)
        return False


@contextmanager
def get_neo4j_session(database: Optional[str] = None) -> Generator[Session, None, None]:
    """
    Context manager yielding an auto-closing Neo4j session.
    Usage:
        with get_neo4j_session() as session:
            result = session.run("MATCH (n) RETURN count(n)")
    """
    driver = get_neo4j_driver()
    session = driver.session(database=database) if database else driver.session()
    try:
        yield session
    finally:
        session.close()
