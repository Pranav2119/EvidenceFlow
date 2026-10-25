# ADR 001: Model Context Protocol (MCP) Server Framework Selection

## Status
Accepted

## Date
2026-10-25

## Context
EvidenceFlow AI requires exposing a standardized Model Context Protocol (MCP) interface for external AI clients (Claude Desktop, Gemini, Cursor/IDE sidecars, and internal agent workflows).
- **Day 6**: Standalone read-only corpus server (`corpus-mcp`) providing search, retrieval, version listing, and cross-version diffing.
- **Tier 2 (Day 9+)**: `memory-cache-mcp` server requiring client session tracking and role-based write isolation (read-only queries vs. read-write memory updates).
- **Tier 3**: External API server requiring client rate-limiting and schema validation.

Two primary framework options exist in the Python ecosystem:
1. **Anthropic Official SDK (`mcp` package, `mcp.server.fastmcp.FastMCP`)**:
   - The official reference implementation of the Model Context Protocol maintained by Anthropic.
   - Pinned at `mcp>=1.3.0,<2.0.0` for FastMCP v1 API stability (in `mcp` 2.x, FastMCP was refactored into `MCPServer`).
   - Uses Starlette/anyio for async concurrency, supports both `stdio` and `sse` transports, and provides native `ClientSession` support for testing and client integration.
   - **Course Alignment**: Standard MCP courses (such as DeepLearning.AI's "Building Systems with the Model Context Protocol") specifically teach and import `from mcp.server.fastmcp import FastMCP` using Anthropic's official Python SDK.
2. **Standalone `fastmcp` (3.x)**:
   - Independent package by Jeremiah Lowin (Prefect).
   - Extends the protocol with built-in middleware abstractions and decorators.
   - Drifts from the official Anthropic SDK release cadence and introduces additional dependency trees.

## Decision
We select **Anthropic's Official MCP Python SDK (`mcp<2`)** (`from mcp.server.fastmcp import FastMCP`).

### Rationale
1. **Reference Compliance**: Zero risk of protocol drift or JSON-RPC schema mismatches when interfacing with official MCP hosts (Claude Desktop, Cursor, LangGraph MCP adapters).
2. **Standard Client Testing**: Anthropic's SDK includes `mcp.client.session.ClientSession`, enabling true end-to-end JSON-RPC client-server testing over in-memory streams (`anyio.create_memory_object_stream`) without spawning external processes.
3. **Course Alignment**: Direct fidelity to the MCP curriculum used in the project specifications.
4. **Architectural Path for Tiers 2 & 3**:
   - **Role Isolation (Tier 2)**: FastMCP exposes `FastMCP(name)` instances that can be partitioned into distinct server boundaries (`corpus-mcp` as read-only vs `memory-cache-mcp` with scoped tool registration). Role checks can be enforced via tool-level Python decorators inspecting context headers.
   - **Rate Limiting (Tier 3)**: FastMCP servers expose underlying Starlette apps (`mcp.sse_app()`) where standard ASGI token-bucket middleware can be mounted, or request-rate decorators applied to tool functions directly.

## Consequences
- Pyproject dependencies specify `mcp>=1.30.0,<2.0.0`.
- All logging within stdio servers must strictly target `sys.stderr` to prevent JSON-RPC framing corruption on `stdout`.
