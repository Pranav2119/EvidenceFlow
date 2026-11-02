"""
evidenceflow.memory: Conversational graph memory and follow-up resolution backed by Neo4j.
"""

from evidenceflow.memory.connection import get_neo4j_driver, close_neo4j_driver

__all__ = ["get_neo4j_driver", "close_neo4j_driver"]
