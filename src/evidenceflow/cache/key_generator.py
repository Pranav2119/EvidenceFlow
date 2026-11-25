"""
key_generator.py: Generates versioned keys for scoped cache entries.
Preserves literal query/text identity; evidence staleness is checked separately.
"""

import hashlib
import hmac
from typing import List, Sequence, Union


def normalize_query(query: str) -> str:
    """
    Preserve literal input text; the legacy function name is retained for callers.
    Punctuation, case, and whitespace can all carry technical meaning.
    """
    if not isinstance(query, str):
        raise TypeError("Cache key input must be a string")
    return query


def generate_scope_key(
    query: str,
    target_versions: Union[Sequence[str], str, None] = "16",
    config_version: str = "v1.0",
    guardrail_version: str = "v1.0",
    secret_salt: str = "evidenceflow-salt",
    intent: str = "query",
) -> str:
    """
    Computes a v2 HMAC-SHA256 key for literal query text and existing scope fields.
    The namespace separates entries from the old lossy query-normalization format.

    Payload format:
        SCOPE:v2|I:{intent}|V:{sorted_versions}|CFG:{config_version}|GD:{guardrail_version}|Q:{norm_q}
    """
    norm_q = normalize_query(query)

    # Normalize target_versions to a sorted comma-delimited string
    if target_versions is None:
        sorted_versions = "any"
    elif isinstance(target_versions, str):
        sorted_versions = target_versions.strip()
    elif isinstance(target_versions, (list, tuple, set)):
        sorted_versions = ",".join(sorted(str(v).strip() for v in target_versions if str(v).strip()))
        if not sorted_versions:
            sorted_versions = "any"
    else:
        sorted_versions = str(target_versions).strip()

    payload = f"SCOPE:v2|I:{intent}|V:{sorted_versions}|CFG:{config_version}|GD:{guardrail_version}|Q:{norm_q}"

    return hmac.new(
        secret_salt.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def generate_retrieval_key(
    query: str,
    target_versions: Union[Sequence[str], str, None] = "16",
    top_k: int = 5,
    secret_salt: str = "evidenceflow-salt",
) -> str:
    """
    Computes a v2 HMAC-SHA256 key for Layer 3 Retrieval cache.
    Binds literal query text, version scope filter, and top_k limit.
    """
    norm_q = normalize_query(query)
    if target_versions is None:
        sorted_versions = "any"
    elif isinstance(target_versions, str):
        sorted_versions = target_versions.strip()
    else:
        sorted_versions = ",".join(sorted(str(v).strip() for v in target_versions if str(v).strip())) or "any"

    payload = f"RETRIEVAL:v2|V:{sorted_versions}|K:{top_k}|Q:{norm_q}"
    return hmac.new(
        secret_salt.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def generate_embedding_key(
    text: str,
    model_name: str = "text-embedding-004",
) -> str:
    """
    Computes a v2 SHA-256 key for Layer 4 Embedding cache.
    Binds literal input text and embedding model name.

    NOTE: Uses plain SHA-256 (not HMAC) intentionally. Embedding vectors are
    not secret-sensitive — they are pure mathematical representations of text.
    No cryptographic salt is required here; the model name alone provides
    sufficient namespace separation between embedding model versions.
    """
    norm_text = normalize_query(text)
    payload = f"EMBEDDING:v2|MODEL:{model_name}|TEXT:{norm_text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_claim_key(
    entity: str,
    attribute: str,
    version: str,
) -> str:
    """
    Returns a canonical readable composite key for Layer 5 Verified-Claim cache.
    Maps an atomic parameter-attribute fact to its key.

    Format: ``CLAIM:{version}:{entity}:{attribute}`` (e.g. "CLAIM:16:shared_buffers:default").

    NOTE: This is a plain human-readable string key, NOT a cryptographic hash.
    It is intentionally reversible for debuggability. If L5 is migrated to an
    external store (Redis, Postgres), treat this key as a plaintext identifier,
    not as opaque bytes.
    """
    clean_entity = entity.strip().lower()
    clean_attr = attribute.strip().lower()
    clean_v = str(version).strip()
    return f"CLAIM:{clean_v}:{clean_entity}:{clean_attr}"
