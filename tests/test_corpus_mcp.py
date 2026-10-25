"""
test_corpus_mcp.py: Comprehensive test suite for FastMCP read-only corpus server.
Tests:
  1. MCP ClientSession end-to-end integration over in-memory JSON-RPC streams.
  2. Read-only tool registration gate (asserting no write tools exist).
  3. search_passages retrieval and chunk citation guidance.
  4. get_passage Qdrant-only retrieval and graceful not-found / malformed handling.
  5. list_versions supported version scope.
  6. diff_sections cross-version parameter tracking on real PostgreSQL settings:
     - log_checkpoints (default changed: off in 14 -> on in 15/16)
     - vacuum_defer_cleanup_age (removed in 16)
     - vacuum_buffer_usage_limit (new in 16)
  7. Strict input validation guards (version set, length limits, chunk ID format).
"""

import json
import pytest
import anyio
from mcp.client.session import ClientSession
from evidenceflow.mcp.corpus_server import (
    mcp,
    search_passages,
    get_passage,
    list_versions,
    diff_sections,
)


@pytest.fixture
async def mcp_session():
    """
    Spawns an end-to-end in-memory MCP ClientSession connected to the FastMCP corpus server
    over bidirectional anyio memory streams.
    """
    client_send, server_receive = anyio.create_memory_object_stream(20)
    server_send, client_receive = anyio.create_memory_object_stream(20)

    async with anyio.create_task_group() as tg:
        tg.start_soon(
            mcp._mcp_server.run,
            server_receive,
            server_send,
            mcp._mcp_server.create_initialization_options(),
        )
        async with ClientSession(client_receive, client_send) as session:
            await session.initialize()
            yield session
        tg.cancel_scope.cancel()


# --- 1. Read-Only Invariant & Tool Registration Gate ---

@pytest.mark.anyio
async def test_no_write_tools_registered_on_corpus_mcp(mcp_session):
    """
    Day 6 Security Invariant: The corpus server must be strictly read-only.
    Fails if any registered tool name resembles a mutation or write operation.
    """
    write_forbidden_keywords = {
        "create", "insert", "update", "delete", "drop", "write",
        "modify", "alter", "mutate", "put", "patch", "remove",
        "add", "set", "save", "post", "truncate"
    }

    tools_response = await mcp_session.list_tools()
    tool_names = [t.name for t in tools_response.tools]

    assert len(tool_names) == 4
    assert set(tool_names) == {"search_passages", "get_passage", "list_versions", "diff_sections"}

    for name in tool_names:
        name_lower = name.lower()
        for forbidden in write_forbidden_keywords:
            assert forbidden not in name_lower, (
                f"Security Violation: Tool '{name}' contains forbidden mutation keyword '{forbidden}'."
            )


# --- 2. Tool Description Integrity & Guidance ---

@pytest.mark.anyio
async def test_tool_descriptions_mandate_citations(mcp_session):
    """
    Day 6 Requirement 7: Tool descriptions must clearly mandate citing chunk_id
    and declaring insufficient evidence when ungrounded.
    """
    tools_response = await mcp_session.list_tools()
    tools_map = {t.name: t.description for t in tools_response.tools}

    assert "chunk_id" in tools_map["search_passages"]
    assert "insufficient evidence" in tools_map["search_passages"].lower()
    assert "chunk_id" in tools_map["get_passage"]
    assert "not_found" in tools_map["get_passage"]
    assert "cross-version" in tools_map["diff_sections"].lower() or "versions" in tools_map["diff_sections"].lower()


# --- 3. list_versions Tool via ClientSession ---

@pytest.mark.anyio
async def test_list_versions_via_client_session(mcp_session):
    """
    Verifies list_versions returns supported versions 14, 15, and 16 with descriptions.
    """
    res = await mcp_session.call_tool("list_versions", {})
    assert len(res.content) == 1
    data = json.loads(res.content[0].text)

    assert data["status"] == "success"
    assert data["supported_versions"] == ["14", "15", "16"]
    assert data["default_version"] == "16"
    assert "14" in data["version_details"]
    assert "15" in data["version_details"]
    assert "16" in data["version_details"]


# --- 4. search_passages Tool via ClientSession ---

@pytest.mark.anyio
async def test_search_passages_via_client_session(mcp_session):
    """
    Executes search_passages through JSON-RPC ClientSession and verifies ranked passage schema.
    """
    res = await mcp_session.call_tool("search_passages", {"query": "work_mem default value", "version": "16", "limit": 3})
    assert len(res.content) == 1
    data = json.loads(res.content[0].text)

    assert data["status"] == "success"
    assert data["version"] == "16"
    assert data["count"] > 0
    assert len(data["passages"]) <= 3

    top_hit = data["passages"][0]
    assert "chunk_id" in top_hit
    assert "work-mem" in top_hit["chunk_id"]
    assert top_hit["score"] > 0.0
    assert len(top_hit["sha256"]) == 64
    assert "Always cite the returned chunk_id" in data["guidance"]


# --- 5. get_passage Tool (Qdrant Only) & Break-It Handling ---

@pytest.mark.anyio
async def test_get_passage_valid_chunk_id(mcp_session):
    """
    Retrieves full passage directly from Qdrant by exact chunk_id.
    """
    target_cid = "pg16-runtime-config-resource-work-mem"
    res = await mcp_session.call_tool("get_passage", {"chunk_id": target_cid})
    data = json.loads(res.content[0].text)

    assert data["status"] == "success"
    assert data["chunk_id"] == target_cid
    assert data["version"] == "16"
    assert "work_mem" in data["content"]
    assert len(data["sha256"]) == 64


@pytest.mark.anyio
async def test_get_passage_not_found_returns_graceful_json(mcp_session):
    """
    Break It on Purpose: Querying non-existent chunk ID returns clean status: 'not_found'
    without throwing unhandled 500 exceptions.
    """
    res = await mcp_session.call_tool("get_passage", {"chunk_id": "pg16-non-existent-phantom-chunk"})
    data = json.loads(res.content[0].text)

    assert data["status"] == "not_found"
    assert "not found in Qdrant collection" in data["message"]


@pytest.mark.anyio
async def test_get_passage_malformed_chunk_id_returns_error(mcp_session):
    """
    Break It on Purpose: Malformed chunk ID (violating pg<14|15|16>- format) returns status: 'error'.
    """
    res = await mcp_session.call_tool("get_passage", {"chunk_id": "malicious_chunk_id; DROP TABLE--"})
    data = json.loads(res.content[0].text)

    assert data["status"] == "error"
    assert "Malformed chunk_id format" in data["error"]


# --- 6. diff_sections Cross-Version Parameter Differences ---

def test_diff_sections_log_checkpoints_default_change():
    """
    Real Difference 1: log_checkpoints default changed from 'off' (PG14) to 'on' (PG15/16).
    """
    res = diff_sections(setting="log_checkpoints", v1="14", v2="15")
    assert res["status"] == "success"
    assert res["setting"] == "log_checkpoints"
    assert res["found_v1"] is True
    assert res["found_v2"] is True
    assert res["changed"] is True
    assert res["default_v1"] == "off"
    assert res["default_v2"] == "on"
    assert "changed default value" in res["summary"]
    assert "-Causes checkpoints and restartpoints" in res["diff"] or "The default is on" in res["diff"]


def test_diff_sections_vacuum_defer_cleanup_age_removed_in_pg16():
    """
    Real Difference 2: vacuum_defer_cleanup_age existed in PG14/15 (default: 0),
    but was removed in PostgreSQL 16.
    """
    res = diff_sections(setting="vacuum_defer_cleanup_age", v1="14", v2="16")
    assert res["status"] == "success"
    assert res["setting"] == "vacuum_defer_cleanup_age"
    assert res["found_v1"] is True
    assert res["found_v2"] is False
    assert res["changed"] is True
    assert res["default_v1"] is not None
    assert res["default_v2"] is None
    assert "was removed or absent in PostgreSQL 16" in res["summary"]
    assert "postgresql-16/(removed)" in res["diff"]


def test_diff_sections_vacuum_buffer_usage_limit_new_in_pg16():
    """
    Real Difference 3: vacuum_buffer_usage_limit was newly introduced in PostgreSQL 16
    (default: 256 kB) and did not exist in PostgreSQL 14.
    """
    res = diff_sections(setting="vacuum_buffer_usage_limit", v1="14", v2="16")
    assert res["status"] == "success"
    assert res["setting"] == "vacuum_buffer_usage_limit"
    assert res["found_v1"] is False
    assert res["found_v2"] is True
    assert res["changed"] is True
    assert res["default_v1"] is None
    assert res["default_v2"] == "256 kB"
    assert "newly introduced in PostgreSQL 16" in res["summary"]
    assert "postgresql-14/(non-existent)" in res["diff"]


def test_diff_sections_non_existent_setting_handles_gracefully():
    """
    Handles a parameter present in neither version gracefully.
    """
    res = diff_sections(setting="hyperdrive_warp_speed", v1="14", v2="16")
    assert res["status"] == "not_found"
    assert res["found_v1"] is False
    assert res["found_v2"] is False
    assert res["changed"] is False
    assert "was not found in either" in res["summary"]


# --- 7. Input Validation Guards ---

def test_input_validation_guards():
    """
    Verifies input boundaries: version sets, query length, and empty arguments.
    """
    # 1. Invalid version
    res_bad_ver = search_passages(query="work_mem", version="99")
    assert res_bad_ver["status"] == "error"
    assert "Invalid PostgreSQL version" in res_bad_ver["error"]

    # 2. Oversized query (>500 chars)
    long_query = "work_mem " * 70  # ~630 chars
    res_long_q = search_passages(query=long_query, version="16")
    assert res_long_q["status"] == "error"
    assert "exceeds maximum allowed" in res_long_q["error"]

    # 3. Empty query
    res_empty_q = search_passages(query="", version="16")
    assert res_empty_q["status"] == "error"
    assert "cannot be empty" in res_empty_q["error"]

    # 4. Limit clamped between 1 and 20
    res_high_lim = search_passages(query="work_mem", version="16", limit=100)
    assert res_high_lim["status"] == "success"
    assert res_high_lim["count"] <= 20
