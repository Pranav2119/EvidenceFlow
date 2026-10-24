"""
bm25.py: High-performance BM25 lexical search index over versioned PostgreSQL documentation chunks.
Provides tokenization with snake_case and hyphen decomposition, cached per-version BM25Okapi models,
and typo suggestion utilities for setting names using difflib.
"""

import difflib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

COMMON_STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "does", "have", "from", "with", "postgres", "postgresql", "version", "release",
    "configure", "setting", "parameter", "options", "value", "default", "explain",
    "show", "tell", "about", "describe", "find", "list", "between", "this", "that",
    "the", "and", "for", "are", "can", "should", "would", "could", "into", "also",
    "is", "in", "it", "to", "of", "an", "as", "at", "by", "or",
}


def tokenize_text(text: str) -> List[str]:
    """
    Extracts alphanumeric tokens, splitting snake_case and hyphenated compounds
    so technical terms match both combined and decomposed representations.
    """
    words = re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())
    sub_tokens = []
    for w in words:
        if "_" in w:
            sub_tokens.extend(w.split("_"))
    return [t for t in words + sub_tokens if t not in COMMON_STOPWORDS and len(t) > 1]


class BM25Index:
    """
    In-memory BM25 lexical index partitioned by documentation version.
    Loads from data/processed/chunks.jsonl or falls back to Qdrant collection.
    """

    def __init__(self, chunks_path: Optional[str] = None):
        self.chunks_by_version: Dict[str, List[Dict[str, Any]]] = {"14": [], "15": [], "16": []}
        self.models: Dict[str, BM25Okapi] = {}
        self.setting_names: Set[str] = set()
        self._load_corpus(chunks_path)
        self._build_indices()

    def _load_corpus(self, chunks_path: Optional[str] = None):
        """Loads chunks from JSONL file or fallback paths."""
        default_paths = [
            chunks_path,
            "data/processed/chunks.jsonl",
            "../data/processed/chunks.jsonl",
            os.path.join(os.path.dirname(__file__), "../../../data/processed/chunks.jsonl"),
        ]
        resolved_path = None
        for p in default_paths:
            if p and os.path.exists(p):
                resolved_path = p
                break

        if not resolved_path:
            logger.warning("BM25: chunks.jsonl not found. Initializing empty BM25 index.")
            return

        with open(resolved_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                v = str(item.get("version"))
                cid = item.get("id", item.get("chunk_id", ""))
                if v in self.chunks_by_version:
                    chunk_dict = {
                        "chunk_id": cid,
                        "version": v,
                        "doc_name": item.get("doc_name", ""),
                        "section": item.get("section", ""),
                        "title": item.get("title", ""),
                        "content": item.get("content", ""),
                        "sha256": item.get("sha256", ""),
                    }
                    self.chunks_by_version[v].append(chunk_dict)

                # Extract recognized setting names for did-you-mean suggestions
                if "runtime-config-" in cid:
                    suffix = re.sub(r"^pg\d+-runtime-config-[a-z]+-", "", cid).replace("-", "_")
                    if suffix and not re.match(r"^[0-9a-f]{8,}$", suffix):
                        self.setting_names.add(suffix)

    def _build_indices(self):
        """Constructs BM25Okapi index for each version partition."""
        for v, chunks in self.chunks_by_version.items():
            if not chunks:
                continue
            corpus = [
                tokenize_text(f"{c['title']} {c['chunk_id']} {c['content']}")
                for c in chunks
            ]
            self.models[v] = BM25Okapi(corpus)

    def search(
        self,
        query: str,
        version: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Executes BM25 lexical search scoped to the requested version.
        Returns top chunks ranked by BM25 score.
        """
        v = str(version) if version else "16"
        if v not in self.models or not self.chunks_by_version[v]:
            return []

        tokens = tokenize_text(query)
        if not tokens:
            return []

        scores = self.models[v].get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in ranked_indices[:limit]:
            if scores[idx] <= 0.0:
                break
            chunk = dict(self.chunks_by_version[v][idx])
            chunk["bm25_score"] = float(scores[idx])
            results.append(chunk)

        return results

    def suggest_setting_name(self, term: str, cutoff: float = 0.75) -> Optional[str]:
        """Returns closest known setting name if a typo is detected."""
        norm_term = term.replace("-", "_").lower()
        matches = difflib.get_close_matches(norm_term, list(self.setting_names), n=1, cutoff=cutoff)
        return matches[0] if matches else None


_bm25_instance: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    """Returns the process-level cached singleton BM25Index."""
    global _bm25_instance
    if _bm25_instance is None:
        _bm25_instance = BM25Index()
    return _bm25_instance
