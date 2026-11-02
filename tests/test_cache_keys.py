"""
test_cache_keys.py: Unit tests for composite HMAC-SHA256 scope key generator.
Verifies version isolation, query normalization, config isolation, and salt security.
"""

import pytest
from evidenceflow.cache.key_generator import (
    generate_claim_key,
    generate_embedding_key,
    generate_retrieval_key,
    generate_scope_key,
    normalize_query,
)


def test_version_isolation():
    """Identical questions with different target versions must produce completely distinct hash keys."""
    q = "What is the default value of wal_level?"
    key_v14 = generate_scope_key(q, target_versions="14")
    key_v16 = generate_scope_key(q, target_versions="16")
    key_multi = generate_scope_key(q, target_versions=["14", "16"])

    assert key_v14 != key_v16, "Collision between v14 and v16 keys!"
    assert key_v14 != key_multi, "Collision between single and multi-version keys!"
    assert key_v16 != key_multi, "Collision between single and multi-version keys!"


def test_query_normalization():
    """Punctuation, casing, and whitespace variations must yield the exact same scope key."""
    q1 = "What is the default value of shared_buffers in PostgreSQL 16?"
    q2 = "  what is the default value of shared_buffers in postgresql 16?  "
    q3 = "WHAT IS THE DEFAULT VALUE OF SHARED_BUFFERS IN POSTGRESQL 16???"
    q4 = "What   is  the   default   value   of   shared_buffers   in   PostgreSQL 16\n\t"

    key1 = generate_scope_key(q1, "16")
    key2 = generate_scope_key(q2, "16")
    key3 = generate_scope_key(q3, "16")
    key4 = generate_scope_key(q4, "16")

    assert key1 == key2 == key3 == key4, "Query normalization failed to produce identical keys!"


def test_config_version_isolation():
    """Updating configuration version must invalidate prior cached answers."""
    q = "What is autovacuum_max_workers?"
    key_cfg1 = generate_scope_key(q, "16", config_version="v1.0")
    key_cfg2 = generate_scope_key(q, "16", config_version="v1.1")

    assert key_cfg1 != key_cfg2, "Collision across configuration versions!"


def test_guardrail_version_isolation():
    """Tightening or modifying guardrails must invalidate cached answers."""
    q = "How do I bypass authentication?"
    key_grd1 = generate_scope_key(q, "16", guardrail_version="v1.0")
    key_grd2 = generate_scope_key(q, "16", guardrail_version="v2.0-hardened")

    assert key_grd1 != key_grd2, "Collision across guardrail versions!"


def test_salt_isolation():
    """Different cryptographic salts must generate distinct keys."""
    q = "Explain archive_mode"
    key1 = generate_scope_key(q, "16", secret_salt="salt-alpha")
    key2 = generate_scope_key(q, "16", secret_salt="salt-beta")

    assert key1 != key2, "Different salts produced identical HMAC!"


def test_version_sorting_invariance():
    """Target versions list order must not affect the resulting scope key."""
    q = "Compare vacuum settings"
    key_ab = generate_scope_key(q, target_versions=["14", "16"])
    key_ba = generate_scope_key(q, target_versions=["16", "14"])

    assert key_ab == key_ba, "Version list order affected the deterministic key!"


def test_retrieval_key_generation():
    """Retrieval key incorporates query, version, and top_k."""
    q = "max_connections configuration"
    k_k5 = generate_retrieval_key(q, target_versions="16", top_k=5)
    k_k10 = generate_retrieval_key(q, target_versions="16", top_k=10)
    k_v14 = generate_retrieval_key(q, target_versions="14", top_k=5)

    assert k_k5 != k_k10, "Top-K parameter did not affect retrieval key!"
    assert k_k5 != k_v14, "Version parameter did not affect retrieval key!"


def test_embedding_key_generation():
    """Embedding key is deterministic and model-bound."""
    text = "PostgreSQL WAL archiving configuration parameter"
    k_model1 = generate_embedding_key(text, model_name="text-embedding-004")
    k_model2 = generate_embedding_key(text, model_name="fastembed-bge-small")

    assert k_model1 != k_model2, "Model name did not isolate embedding key!"


def test_claim_key_generation():
    """Claim key normalizes casing and whitespace for atomic facts."""
    k1 = generate_claim_key(entity=" shared_buffers ", attribute="DEFAULT", version="16")
    k2 = generate_claim_key(entity="SHARED_BUFFERS", attribute="default", version="16")

    assert k1 == k2 == "CLAIM:16:shared_buffers:default"


def test_empty_and_whitespace_query_normalization():
    """Empty string and whitespace-only queries both normalize to an empty string."""
    assert normalize_query("") == ""
    assert normalize_query("   ") == ""
    assert normalize_query("\n\t  ") == ""
    # Two whitespace-only queries must produce the same scope key (not crash)
    k1 = generate_scope_key("   ", "16")
    k2 = generate_scope_key("", "16")
    assert k1 == k2, "Whitespace-only and empty queries must normalize identically"


def test_none_target_versions_maps_to_any():
    """target_versions=None maps to the literal sentinel 'any' in the payload."""
    k_none = generate_scope_key("What is wal_level?", target_versions=None)
    k_any_str = generate_scope_key("What is wal_level?", target_versions="any")
    # Both normalize to V:any — keys must be identical
    assert k_none == k_any_str, (
        "None and the string 'any' must produce the same key "
        "(both map to the sentinel 'any' in the HMAC payload)"
    )


def test_underscore_preserved_in_setting_names():
    """Underscores in PostgreSQL setting names must survive normalization."""
    k_underscore = generate_scope_key("What is shared_buffers default?", "16")
    k_space = generate_scope_key("What is shared buffers default?", "16")
    # 'shared_buffers' (setting name, underscore) != 'shared buffers' (plain words)
    assert k_underscore != k_space, (
        "Underscore in a setting name is significant and must produce a different "
        "key from the space-separated form"
    )
