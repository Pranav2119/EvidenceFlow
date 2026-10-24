"""
qdrant_client.py: Search client interface for Qdrant with single-stage version payload filtering.
"""

import argparse
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from evidenceflow.config import get_settings
from evidenceflow.storage.embeddings import get_embedding

logger = logging.getLogger(__name__)


class EvidenceQdrantClient:
    """
    Production search interface wrapping Qdrant with single-stage version and payload filtering.
    """

    def __init__(self):
        self.settings = get_settings()
        self.client = QdrantClient(
            host=self.settings.QDRANT_HOST,
            port=self.settings.QDRANT_PORT,
            check_compatibility=False,
        )
        self.collection_name = self.settings.QDRANT_COLLECTION

    def search(
        self,
        query: str,
        version: Optional[str] = None,
        limit: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes a dense vector similarity search over documentation chunks.
        Applies single-stage payload filtering for documentation version during HNSW traversal.
        """
        query_vector = get_embedding(query)

        query_filter = None
        if version:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="version",
                        match=models.MatchValue(value=str(version)),
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
        )

        results = []
        for point in response.points:
            results.append({
                "chunk_id": point.payload.get("chunk_id", str(point.id)),
                "version": point.payload.get("version", ""),
                "doc_name": point.payload.get("doc_name", ""),
                "section": point.payload.get("section", ""),
                "title": point.payload.get("title", ""),
                "content": point.payload.get("content", ""),
                "score": point.score,
                "sha256": point.payload.get("sha256", ""),
            })

        return results


def main():
    parser = argparse.ArgumentParser(description="Query Qdrant vector store with version filtering.")
    parser.add_argument("--query", type=str, required=True, help="Search query string")
    parser.add_argument("--version", type=str, default=None, help="Optional version filter (e.g. 14, 15, 16)")
    parser.add_argument("--limit", type=int, default=3, help="Max results to return (default: 3)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    client = EvidenceQdrantClient()

    print(f"\nSearching for: '{args.query}' (version filter: {args.version or 'None'})...\n")
    results = client.search(query=args.query, version=args.version, limit=args.limit)

    for i, r in enumerate(results, 1):
        print(f"[{i}] Score: {r['score']:.4f} | v{r['version']} | {r['chunk_id']}")
        print(f"    Section: {r['section']}")
        print(f"    Title:   {r['title']}")
        print(f"    Content: {r['content'][:140]}...")
        print("-" * 70)


if __name__ == "__main__":
    main()
