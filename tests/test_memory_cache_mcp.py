"""
test_memory_cache_mcp.py: Comprehensive test suite for FastMCP memory and cache server.
Validates strict role-based write isolation, permission rejections, and MCP tool execution.
"""

from unittest.mock import MagicMock, patch
import pytest
import anyio
from mcp.client.session import ClientSession

from evidenceflow.cache.manager import ScopedCacheManager
from evidenceflow.cache.models import CachedCitation
from evidenceflow.graph.nodes.cache_gatekeeper import set_cache_manager
from evidenceflow.mcp.memory_cache_server import (
    mcp,
    lookup_cache,
    write_cache,
    query_entity_graph,
    commit_turn_memory,
)


@pytest.fixture(autouse=True)
def clean_cache():
    """Ensure clean isolated cache for every test."""
    cm = ScopedCacheManager()
    set_cache_manager(cm)
    yield cm
    set_cache_manager(None)


@pytest.fixture
async def mcp_session():
    """
    Spawns an end-to-end in-memory MCP ClientSession connected to the FastMCP memory-cache server
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


# =============================================================================
# 1. Role-Based Security & Write Isolation Tests
# =============================================================================

def test_unauthorized_cache_write_rejections():
    """Retriever, AnswerWriter, Planner, and anonymous clients CANNOT write to cache."""
    unauthorized_roles = ["retriever", "answer_writer", "planner", "guest", "", None]

    for role in unauthorized_roles:
        res = write_cache(
            query="What is wal_level?",
            answer="replica",
            caller_role=role,
        )
        assert res["status"] == "permission_denied"
        assert "PermissionDenied" in res["error"]
        assert "cannot write to cache" in res["error"]


def test_authorized_cache_write_and_lookup():
    """Only cache_gatekeeper and admin may commit verified answers to cache."""
    # Write as cache_gatekeeper
    write_res = write_cache(
        query="What is max_connections in v16?",
        answer="Default is 100 [^chunk1].",
        citations=[{"passage_id": "chunk1", "chunk_hash": "h1", "title": "Connections"}],
        version="16",
        caller_role="cache_gatekeeper",
    )
    assert write_res["status"] == "success"
    assert "key" in write_res

    # Lookup as planner
    lookup_res = lookup_cache(
        query="What is max_connections in v16?",
        version="16",
        caller_role="planner",
    )
    assert lookup_res["status"] == "hit"
    assert lookup_res["hit"] is True
    assert "100" in lookup_res["answer"]
    assert len(lookup_res["citations"]) == 1

    # Lookup as orchestrator
    orch_res = lookup_cache(
        query="What is max_connections in v16?",
        version="16",
        caller_role="orchestrator",
    )
    assert orch_res["hit"] is True


def test_unauthorized_memory_commit_rejections():
    """AnswerWriter, Retriever, and Planner CANNOT write to Neo4j conversational memory."""
    unauthorized_roles = ["answer_writer", "retriever", "planner", "analyst"]

    for role in unauthorized_roles:
        res = commit_turn_memory(
            turn_data={"question": "Tune shared_buffers"},
            caller_role=role,
        )
        assert res["status"] == "permission_denied"
        assert "PermissionDenied" in res["error"]
        assert "cannot write to memory" in res["error"]


@patch("evidenceflow.mcp.memory_cache_server.neo4j_record_turn", return_value="turn_uuid_123")
def test_authorized_memory_commit_and_query(mock_neo4j_record):
    """Only memory_curator and admin may commit to Neo4j conversational memory."""
    res = commit_turn_memory(
        turn_data={
            "turn_id": "turn_uuid_123",
            "thread_id": "thread_abc",
            "turn_number": 1,
            "question": "What is autovacuum_naptime?",
            "version_scope": "16",
            "entities": [{"name": "autovacuum_naptime", "type": "ConfigParam"}],
            "claims": [{"text": "1min", "verdict": "VERIFIED"}],
        },
        caller_role="memory_curator",
    )
    assert res["status"] == "success"
    assert res["turn_id"] == "turn_uuid_123"
    assert res["committed_entities_count"] == 1


@patch("evidenceflow.mcp.memory_cache_server.get_session_history", return_value=[{"turn_number": 1, "question": "q1"}])
def test_authorized_memory_query(mock_history):
    """Planner and memory_resolver can inspect conversational memory."""
    res = query_entity_graph(
        thread_id="thread_abc",
        caller_role="planner",
    )
    assert res["status"] == "success"
    assert res["turns_count"] == 1
    assert res["history"][0]["question"] == "q1"


def test_unauthorized_memory_query_rejection():
    """Unauthorized callers cannot inspect conversational memory."""
    res = query_entity_graph(
        thread_id="thread_abc",
        caller_role="untrusted_external_role",
    )
    assert res["status"] == "permission_denied"
    assert "PermissionDenied" in res["error"]


# =============================================================================
# 2. FastMCP Tool Registration & JSON-RPC ClientSession Tests
# =============================================================================

@pytest.mark.anyio
async def test_mcp_server_tools_registered(mcp_session):
    """Verifies all 4 memory and cache tools are registered on the FastMCP server."""
    tools = await mcp_session.list_tools()
    tool_names = {t.name for t in tools.tools}

    assert "lookup_cache" in tool_names
    assert "write_cache" in tool_names
    assert "query_entity_graph" in tool_names
    assert "commit_turn_memory" in tool_names


@pytest.mark.anyio
async def test_mcp_client_session_cache_write_and_read(mcp_session):
    """Executes write_cache and lookup_cache over standard MCP ClientSession JSON-RPC streams."""
    # 1. Attempt unauthorized write via MCP call_tool
    unauth_result = await mcp_session.call_tool(
        "write_cache",
        arguments={
            "query": "What is checkpoint_timeout in v16?",
            "answer": "5min",
            "caller_role": "retriever",
        },
    )
    unauth_content = unauth_result.content[0].text
    assert "PermissionDenied" in unauth_content

    # 2. Authorized write via MCP call_tool
    auth_result = await mcp_session.call_tool(
        "write_cache",
        arguments={
            "query": "What is checkpoint_timeout in v16?",
            "answer": "Default checkpoint_timeout is 5min [^chunk_ckpt].",
            "citations": [{"passage_id": "chunk_ckpt", "chunk_hash": "hash_ckpt"}],
            "version": "16",
            "caller_role": "cache_gatekeeper",
        },
    )
    auth_content = auth_result.content[0].text
    assert "success" in auth_content

    # 3. Read back via MCP call_tool
    read_result = await mcp_session.call_tool(
        "lookup_cache",
        arguments={
            "query": "What is checkpoint_timeout in v16?",
            "version": "16",
            "caller_role": "planner",
        },
    )
    read_content = read_result.content[0].text
    assert "hit" in read_content
    assert "5min" in read_content


# =============================================================================
# 3. Break It on Purpose Test
# =============================================================================

def test_break_it_on_purpose_role_guard_necessity():
    """
    Break It on Purpose (Roadmap Day 11 Section 6):
    Demonstrates that without role validation, an unverified intermediate retriever
    or hallucinating writer could poison the cache. The role guard actively blocks it.
    """
    poisonous_hallucination = "PostgreSQL 16 max_connections is 9999999 connections."

    # Attempt to write hallucination pretending to be retriever
    attempt = write_cache(
        query="What is max_connections in v16?",
        answer=poisonous_hallucination,
        caller_role="retriever",
    )
    assert attempt["status"] == "permission_denied"

    # Verify that cache remains unpoisoned
    check = lookup_cache(
        query="What is max_connections in v16?",
        version="16",
        caller_role="cache_gatekeeper",
    )
    assert check["hit"] is False  # Clean!
