"""
init_qdrant.py: Initializes Qdrant collection and version-filtered payload indexes.
"""

import argparse
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from evidenceflow.config import get_settings
from evidenceflow.storage.embeddings import get_vector_dimension

logger = logging.getLogger(__name__)


def init_collection(recreate: bool = False) -> str:
    """
    Creates the Qdrant collection with cosine distance and payload schema indexes.
    """
    settings = get_settings()
    collection_name = settings.QDRANT_COLLECTION
    dim = get_vector_dimension()

    client = QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        check_compatibility=False,
    )

    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        if recreate:
            logger.info("Collection '%s' exists. Recreating...", collection_name)
            client.delete_collection(collection_name)
        else:
            logger.info("Collection '%s' already exists. Use --recreate to rebuild.", collection_name)
            return collection_name

    logger.info(
        "Creating collection '%s' with dimension=%d (provider: %s, distance: COSINE)...",
        collection_name, dim, settings.EMBEDDING_PROVIDER
    )
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )

    # Create payload indexes for single-stage filtered HNSW traversal
    indexed_keyword_fields = ["version", "doc_name", "section", "title"]
    for field in indexed_keyword_fields:
        logger.info("Creating keyword payload index on '%s'...", field)
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    # Full-text index on content field for lexical keyword matching
    logger.info("Creating full-text payload index on 'content'...")
    client.create_payload_index(
        collection_name=collection_name,
        field_name="content",
        field_schema=models.TextIndexParams(
            type="text",
            tokenizer=models.TokenizerType.WORD,
            lowercase=True,
        ),
    )

    logger.info("Collection '%s' initialized successfully!", collection_name)
    return collection_name


def main():
    parser = argparse.ArgumentParser(description="Initialize Qdrant collection and payload indexes.")
    parser.add_argument("--recreate", action="store_true", help="Recreate collection if it exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    init_collection(recreate=args.recreate)


if __name__ == "__main__":
    main()
