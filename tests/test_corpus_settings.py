"""
Synthetic unit tests for corpus-backed PostgreSQL setting extraction.
Verifies dynamic loading from corpus representations and extraction precedence.
NOTE: Uses strictly synthetic parameters (do not use h/f benchmark cases).
"""

import pytest
from evidenceflow.memory.corpus_settings import (
    set_corpus_settings_override,
    get_corpus_settings,
)
from evidenceflow.memory.graph_client import (
    extract_explicit_setting,
    extract_all_explicit_settings,
)


@pytest.fixture(autouse=True)
def setup_synthetic_corpus_settings():
    """Injects a synthetic set of corpus settings for test isolation."""
    synthetic_settings = {
        "synth_buffer_size",
        "synth_buffer",
        "synth_wal_keep_segments",
        "synth_checkpoint_delay",
        "synth_enable_fast_scan",
    }
    set_corpus_settings_override(synthetic_settings)
    yield
    set_corpus_settings_override(None)


def test_synthetic_setting_extraction_priority():
    """Verifies that longer compound setting names take precedence over substrings."""
    query = "How do I tune synth_buffer_size on synthetic worker 4?"
    detected = extract_explicit_setting(query)
    assert detected == "synth_buffer_size"
    assert detected != "synth_buffer"


def test_synthetic_multiple_setting_extraction():
    """Verifies extraction of all distinct settings in a multi-setting query."""
    query = "Compare synth_buffer_size and synth_checkpoint_delay for synthetic engine."
    all_settings = extract_all_explicit_settings(query)
    assert set(all_settings) == {"synth_buffer_size", "synth_checkpoint_delay"}


def test_synthetic_no_setting_present():
    """Verifies that queries with no corpus parameters return None / empty list."""
    query = "What is the capital city of synthetic land?"
    assert extract_explicit_setting(query) is None
    assert extract_all_explicit_settings(query) == []


def test_synthetic_word_boundary_matching():
    """Verifies that partial substrings do not trigger false positive parameter matches."""
    query = "The unsynth_buffer_size_max value is high."
    assert extract_explicit_setting(query) is None
