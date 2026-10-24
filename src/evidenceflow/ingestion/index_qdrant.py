"""
index_qdrant.py: Batch embedding and indexing of documentation chunks into Qdrant.
"""

import argparse
import json
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any
from qdrant_client.http import models
from evidenceflow.storage import get_qdrant_client
from evidenceflow.storage.embeddings import get_embeddings
from evidenceflow.storage.init_qdrant import init_collection

logger = logging.getLogger(__name__)


def generate_qdrant_uuid(chunk_id: str) -> str:
    """
    Qdrant string IDs must be RFC 4122 UUIDs.
    Deterministically converts human-readable chunk_id into a stable UUIDv5.
    D2-6: Uses a project-specific namespace (not NAMESPACE_DNS) to avoid semantic
    mismatch and prevent collision with unrelated systems using the same namespace.
    """
    # Fixed project namespace UUID — never change this or all existing point IDs will shift
    EVIDENCEFLOW_NS = uuid.UUID("6ba7b814-9dad-11d1-80b4-00c04fd430c8")
    return str(uuid.uuid5(EVIDENCEFLOW_NS, chunk_id))


def index_chunks(
    input_file: Path = Path("data/processed/chunks.jsonl"),
    batch_size: int = 64,
    recreate: bool = False,
) -> int:
    """
    Reads chunks from JSONL, generates embeddings in batches, and upserts into Qdrant.
    """
    collection_name = init_collection(recreate=recreate)

    # D2-5: use shared factory
    client = get_qdrant_client()

    logger.info("Reading chunks from %s ...", input_file)
    chunks: List[Dict[str, Any]] = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    total = len(chunks)
    logger.info("Loaded %d chunks. Generating embeddings and uploading to '%s' (batch_size=%d)...", total, collection_name, batch_size)

    uploaded = 0
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        batch_texts = [c["content"] for c in batch]

        # Generate dense embeddings for the batch
        embeddings = get_embeddings(batch_texts)

        # Build Qdrant PointStruct list
        points = []
        for chunk, vector in zip(batch, embeddings):
            point_id = generate_qdrant_uuid(chunk["id"])
            payload = {
                "chunk_id": chunk["id"],
                "version": chunk["version"],
                "doc_name": chunk["doc_name"],
                "section": chunk.get("section", ""),
                "title": chunk.get("title", ""),
                "content": chunk["content"],
                "sha256": chunk["sha256"],
                "char_count": chunk.get("char_count", len(chunk["content"])),
            }
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        client.upsert(collection_name=collection_name, points=points)
        uploaded += len(points)
        logger.info("  Uploaded %d / %d chunks (%.1f%%)", uploaded, total, (uploaded / total) * 100)

    # Verify final count in collection
    info = client.get_collection(collection_name)
    logger.info("\nIndexing complete! Qdrant reports %d points in '%s'", info.points_count, collection_name)
    return info.points_count


def main():
    parser = argparse.ArgumentParser(description="Index documentation chunks into Qdrant.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/chunks.jsonl"), help="Path to chunks.jsonl")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size (default: 64)")
    parser.add_argument("--recreate", action="store_true", help="Recreate collection before indexing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    index_chunks(input_file=args.input, batch_size=args.batch_size, recreate=args.recreate)


if __name__ == "__main__":
    main()
