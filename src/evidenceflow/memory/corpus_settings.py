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


# Module-level cache for corpus vocabulary [Item 4]
_CORPUS_VOCABULARY_CACHE: Optional[Set[str]] = None
_VOCABULARY_OVERRIDE: Optional[Set[str]] = None

STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "also", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself",
    "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would",
    "you", "your", "yours", "yourself", "yourselves", "tell", "show", "give", "please",
}

COMMON_QUESTION_WORDS: Set[str] = {
    "behave", "behavior", "mean", "meaning", "work", "working", "look", "see", "find", "think",
    "want", "like", "try", "happen", "happens", "occur", "occurs", "affect", "affects",
    "support", "supports", "supported", "allow", "allows", "allowed", "enable", "enables", "enabled",
    "disable", "disables", "disabled", "apply", "applies", "applied", "turn", "turns", "turned",
    "set", "sets", "setting", "settings", "change", "changes", "changed", "changing",
    "configure", "configures", "configured", "configuring", "show", "shows", "showing",
    "tell", "explain", "describe", "difference", "compare", "same", "similar",
    "safe", "true", "false", "possible", "valid", "good", "bad", "high", "low",
    "run", "runs", "running", "start", "starts", "restart", "restarts", "stop", "stops",
    "reload", "reloads", "reset", "resets", "kill", "cancel",
    "runtime", "online", "offline", "parameter", "parameters", "value", "values",
    "config", "configuration", "database", "cluster", "session", "transaction",
    "backend", "server", "servers", "client", "clients", "connection", "connections",
}

EXCLUDED_WORDS: Set[str] = STOPWORDS | COMMON_QUESTION_WORDS


def set_corpus_vocabulary_override(override: Optional[Set[str]]) -> None:
    """Sets a temporary override for synthetic testing without file or Qdrant dependencies."""
    global _VOCABULARY_OVERRIDE
    _VOCABULARY_OVERRIDE = override


def load_corpus_vocabulary(force_refresh: bool = False) -> Set[str]:
    """
    Extracts the unique vocabulary from indexed corpus chunks (data/processed/chunks.jsonl).
    """
    global _CORPUS_VOCABULARY_CACHE
    if _VOCABULARY_OVERRIDE is not None:
        return _VOCABULARY_OVERRIDE
    if _CORPUS_VOCABULARY_CACHE is not None and not force_refresh:
        return _CORPUS_VOCABULARY_CACHE

    vocab: Set[str] = set()
    from pathlib import Path
    chunks_path = Path(__file__).resolve().parents[3] / "data" / "processed" / "chunks.jsonl"
    if chunks_path.exists():
        try:
            import json
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    text = (chunk.get("title", "") + " " + chunk.get("content", "")).lower()
                    vocab.update(re.findall(r"[a-z0-9_-]{2,}", text))
            logger.info("Loaded %d vocabulary words from %s", len(vocab), chunks_path)
        except Exception as e:
            logger.warning("Failed to parse chunks.jsonl for vocabulary: %s", e)

    _CORPUS_VOCABULARY_CACHE = vocab
    return _CORPUS_VOCABULARY_CACHE


def get_corpus_vocabulary() -> Set[str]:
    """Returns the cached set of corpus vocabulary words."""
    if _VOCABULARY_OVERRIDE is not None:
        return _VOCABULARY_OVERRIDE
    if _CORPUS_VOCABULARY_CACHE is None:
        return load_corpus_vocabulary()
    return _CORPUS_VOCABULARY_CACHE


def has_off_corpus_content_words(query: str) -> bool:
    """
    Checks if a pronoun query contains content words not present in the documentation corpus vocabulary.
    Identifies non-technical / off-domain queries like 'Tell me a joke about servers. Is it funny?'
    or 'What is the capital of France? Is it Paris?' [Item 4].
    """
    vocab = get_corpus_vocabulary()
    if not vocab:
        return False

    tokens = re.findall(r"\b[a-z]{3,}\b", query.lower())
    for token in tokens:
        if token in EXCLUDED_WORDS:
            continue
        # Direct match or basic English inflection reduction
        if token in vocab:
            continue
        if token.endswith("s") and token[:-1] in vocab:
            continue
        if token.endswith("es") and token[:-2] in vocab:
            continue
        if token.endswith("ed") and (token[:-2] in vocab or token[:-1] in vocab):
            continue
        if token.endswith("ing") and token[:-3] in vocab:
            continue
        # Content word outside the PostgreSQL documentation corpus
        return True

    return False
