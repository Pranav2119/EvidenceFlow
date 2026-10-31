"""
corpus_settings.py: Dynamically extracts and caches indexed PostgreSQL GUC parameter names
from the Qdrant vector corpus (titles and chunk payloads).
Replaces hardcoded parameter dictionaries with live corpus-backed discovery.
"""

import logging
import re
from typing import Optional, Set
from qdrant_client import QdrantClient

from evidenceflow.config import get_settings

logger = logging.getLogger(__name__)

# Module-level cache for indexed settings
_CORPUS_SETTINGS_CACHE: Optional[Set[str]] = None
_SETTINGS_OVERRIDE: Optional[Set[str]] = None


def set_corpus_settings_override(override: Optional[Set[str]]) -> None:
    """Sets a temporary override for synthetic testing without Qdrant dependencies."""
    global _SETTINGS_OVERRIDE
    _SETTINGS_OVERRIDE = override


def load_corpus_settings(force_refresh: bool = False) -> Set[str]:
    """
    Scrolls the Qdrant corpus collection to extract all known GUC parameter names
    from chunk titles and payload parameter headers.
    """
    global _CORPUS_SETTINGS_CACHE
    if _SETTINGS_OVERRIDE is not None:
        return _SETTINGS_OVERRIDE

    if _CORPUS_SETTINGS_CACHE is not None and not force_refresh:
        return _CORPUS_SETTINGS_CACHE

    s = get_settings()
    discovered: Set[str] = set()

    try:
        client = QdrantClient(host=s.QDRANT_HOST, port=s.QDRANT_PORT, timeout=10.0)
        offset = None

        while True:
            records, next_offset = client.scroll(
                collection_name="evidenceflow_passages_fastembed",
                limit=250,
                offset=offset,
                with_payload=["title", "content"],
            )
            for r in records:
                # 1. Setting from Title: e.g. "max_connections ( integer )"
                title = r.payload.get("title") or ""
                m_title = re.match(
                    r"^([a-z0-9_]+)\s*\(\s*(?:integer|string|boolean|enum|floating\s+point|real)\s*\)",
                    title,
                    re.IGNORECASE,
                )
                if m_title:
                    discovered.add(m_title.group(1).lower())

                # 2. Setting from Content Header: e.g. "Parameter: max_connections ( integer )"
                content = r.payload.get("content") or ""
                m_content = re.search(r"Parameter:\s*([a-z0-9_]+)", content, re.IGNORECASE)
                if m_content:
                    discovered.add(m_content.group(1).lower())

            if not next_offset:
                break
            offset = next_offset

        logger.info("Successfully discovered %d configuration parameters from Qdrant corpus.", len(discovered))
        _CORPUS_SETTINGS_CACHE = discovered
        return _CORPUS_SETTINGS_CACHE

    except Exception as e:
        logger.warning("Could not load setting names from Qdrant corpus (%s).", e)
        if _CORPUS_SETTINGS_CACHE is not None:
            return _CORPUS_SETTINGS_CACHE
        return set()


def get_corpus_settings() -> Set[str]:
    """Returns the cached set of indexed configuration settings."""
    if _SETTINGS_OVERRIDE is not None:
        return _SETTINGS_OVERRIDE
    if _CORPUS_SETTINGS_CACHE is None:
        return load_corpus_settings()
    return _CORPUS_SETTINGS_CACHE
