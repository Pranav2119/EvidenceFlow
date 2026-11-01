"""
test_verifier_clause_binding.py: Synthetic unit tests for clause-level attribute binding
and section reference stripping in the deterministic verifier.
"""

import pytest
from evidenceflow.graph.nodes.verifier import (
    extract_claimed_values,
    verify_statement_value_presence,
)

SAMPLE_PASSAGE = (
    "Section: 20.5. Write Ahead Log > 20.5.2. Checkpoints\n"
    "Parameter: checkpoint_timeout ( integer )\n\n"
    "Sets the maximum time between automatic WAL checkpoints.\n"
    "The default is five minutes ( 5min ).\n"
    "The valid range is between 30 seconds and one day.\n"
    "This parameter can only be set in the postgresql.conf file."
)


def test_clause_bound_range_and_default_passes():
    stmt = "The default is five minutes (5min), and the valid range is between 30 seconds and one day"
    missing, err = verify_statement_value_presence(stmt, SAMPLE_PASSAGE)
    assert not missing, f"Expected pass but got missing: {missing}"
    assert err is None, f"Expected pass but got error: {err}"


def test_wrong_attribute_binding_fails():
    stmt = "The default is 30s"
    missing, err = verify_statement_value_presence(stmt, SAMPLE_PASSAGE)
    assert err is not None
    assert "does not match claimed attribute 'default'" in err


def test_maximum_attribute_mismatch_fails():
    stmt = "The maximum is 5min"
    missing, err = verify_statement_value_presence(stmt, SAMPLE_PASSAGE)
    assert err is not None
    assert "does not match claimed attribute 'maximum'" in err


def test_section_references_stripped_and_real_decimals_preserved():
    stmt = (
        "In section 20.5., and references 20.10. and E.16.2., "
        "the default is 0.9, while other parameters are 4.0 and 0.2."
    )
    extracted = extract_claimed_values(stmt)
    # Section numbers 20.5., 20.10., E.16.2. must NOT be in extracted values
    assert "20.5" not in extracted
    assert "20.5." not in extracted
    assert "20.10" not in extracted
    assert "20.10." not in extracted
    assert "E.16.2" not in extracted
    assert "16.2" not in extracted

    # Real decimal values must be preserved
    assert "0.9" in extracted
    assert "4.0" in extracted
    assert "0.2" in extracted
