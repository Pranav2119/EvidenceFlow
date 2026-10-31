"""
Synthetic unit tests for verifier precision and safe sentence tokenization.
Verifies requirements (f) and (j):
  - Sentence splitter does not split inside decimals (4.0, 0.2), negative numbers (-1), or abbreviations (e.g., i.e.).
  - Everyday prose words ('true', 'yes', 'off' in prose) are not extracted as claimed values.
  - Single-setting chunks allow attribute word + value in one sentence without repeating setting name.
NOTE: Uses strictly synthetic parameters (do not use h/f benchmark cases).
"""

import pytest
from evidenceflow.graph.nodes.verifier import (
    extract_claimed_values,
    split_sentences_safely,
    verify_statement_value_presence,
)


def test_sentence_splitter_preserves_decimals_negatives_and_abbreviations():
    """Requirement (j): splitter does not break decimals, negatives, or abbreviations."""
    sample_text = (
        "Section: 20.4. Synthetic Architecture\n"
        "Parameter: synth_rate_limiter ( real )\n\n"
        "Configures the synthetic throughput. The default is 4.0MB (or 0.2GB in large clusters).\n"
        "For example (e.g. in test mode), setting it to -1 disables the limiter.\n"
        "Does it require a restart? Yes! Please see e.g. section 4."
    )

    sentences = [s for s, _ in split_sentences_safely(sample_text)]

    # Decimals 4.0MB and 0.2GB must be preserved in a single sentence
    dec_sent = [s for s in sentences if "4.0MB" in s]
    assert len(dec_sent) == 1
    assert "0.2GB" in dec_sent[0]

    # Negative number -1 and abbreviation e.g. must be preserved in a single sentence
    neg_sent = [s for s in sentences if "-1" in s]
    assert len(neg_sent) == 1
    assert "e.g." in neg_sent[0]


def test_everyday_words_not_extracted_as_claimed_values():
    """Requirement (f, j): 'true', 'yes', 'off' in casual prose must not count as claimed values."""
    # Prose questions and answers
    assert extract_claimed_values("Is it true that synthetic parameter was removed?") == []
    assert extract_claimed_values("Yes, it was removed in synthetic version 16.") == []
    assert extract_claimed_values("It was turned off due to performance concerns.") == []
    assert extract_claimed_values("The team agreed that it is true.") == []

    # Genuine parameter defaults and state assertions
    assert extract_claimed_values("The default value of synth_param is off.") == ["off"]
    assert extract_claimed_values("synth_param defaults to on.") == ["on"]
    assert extract_claimed_values("The default is -1.") == ["-1"]
    assert extract_claimed_values("The default is 4.0MB.") == ["4.0mb"]


def test_single_setting_chunk_attribute_and_value_verification():
    """
    Requirement (j): for single-setting chunks, do not require the setting name
    in the same sentence as the value; attribute word + value in one sentence is enough.
    """
    chunk_content = (
        "Section: 10.1. Synthetic Configuration\n"
        "Parameter: synth_buffer_multiplier ( real )\n\n"
        "Specifies the allocation factor for synthetic workers.\n"
        "The default is 4.0MB.\n"
        "The minimum is 0.2GB."
    )

    # Claim asserts default without repeating 'synth_buffer_multiplier' in the same clause
    statement = "The default is 4.0MB."
    missing, err = verify_statement_value_presence(statement, chunk_content)
    assert missing == []
    assert err is None

    # Claim asserts minimum
    statement_min = "The minimum is 0.2GB."
    missing_min, err_min = verify_statement_value_presence(statement_min, chunk_content)
    assert missing_min == []
    assert err_min is None
