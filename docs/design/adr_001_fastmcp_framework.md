# ADR 001: Model Context Protocol (MCP) Server Framework Selection

## Status
Accepted

## Date
2026-10-25 (Revisit Date: After Tier 3 completion, ~Q1 2027)

## Context
EvidenceFlow AI requires exposing a standardized Model Context Protocol (MCP) interface for external AI clients (Claude Desktop, Gemini, Cursor/IDE sidecars, and internal agent workflows).
- **Day 6**: Standalone read-only corpus server (`corpus-mcp`) providing search, retrieval, version listing, and cross-version diffing.
- **Tier 2 (Day 9+)**: `memory-cache-mcp` server requiring client session tracking and role-based write isolation (read-only queries vs. read-write memory updates).
- **Tier 3**: External API server requiring client rate-limiting and schema validation.

Two primary framework options exist in the Python ecosystem:
1. **Anthropic Official SDK (`mcp` package, `mcp.server.fastmcp.FastMCP`)**:
   - The official reference implementation of the Model Context Protocol maintained by Anthropic.
   - Pinned at `mcp>=1.3.0,<2.0.0` for FastMCP 1.0 API stability (chosen to match the official SDK's FastMCP 1.0 API; confirm with course notes).
   - Uses Starlette/anyio for async concurrency, supports both `stdio` and `sse` transports, and provides native `ClientSession` support for testing.
2. **Standalone `fastmcp` (3.x)**:
   - Independent package by Jeremiah Lowin (Prefect).
   - Extends the protocol with built-in conveniences, middleware abstractions, and authentication/rate-limiting decorators.
   - Separate release cadence from the official Anthropic SDK with differing internal abstractions.

---

## Comparison Matrix

| Evaluation Criteria | Official Anthropic SDK (`mcp.server.fastmcp`, `mcp<2`) | Standalone FastMCP (`fastmcp` 3.x) |
| :--- | :--- | :--- |
| **Protocol Conformity** | Reference implementation maintained directly by Anthropic / MCP Working Group. Zero schema drift. | Adheres to protocol, but abstractions and extensions wrap the underlying primitives. |
| **API Stability** | FastMCP 1.0 API is stable in `mcp<2` (`mcp==1.30.0`). | Active independent semantic releases; frequent feature evolution. |
| **Client / Test Tooling** | Official `mcp.client.session.ClientSession` and in-memory anyio stream transports available out of the box. | Focuses primarily on server authoring and developer CLI conveniences. |
| **Out-of-the-Box Auth & Rate Limiting** | **None.** Does not provide built-in rate-limiting or token-bucket middleware. | Includes built-in decorators and utilities for authentication and rate-limiting. |
| **Dependency Footprint** | Lightweight, official dependency (`pydantic`, `anyio`, `starlette`, `httpx`). | Additional third-party dependency tree and release cadence. |

---

## Decision
We select **Anthropic's Official MCP Python SDK (`mcp<2`)** (`from mcp.server.fastmcp import FastMCP`), chosen to match the official SDK's FastMCP 1.0 API; confirm with course notes.

### Rationale
1. **Reference Compliance**: Direct alignment with the official MCP reference implementation and host environments (Claude Desktop, Cursor, LangGraph MCP adapters).
2. **In-Memory Client Testing**: Anthropic's SDK includes `mcp.client.session.ClientSession`, enabling true end-to-end JSON-RPC client-server testing over in-memory streams (`anyio.create_memory_object_stream`) without spawning external processes.
3. **API Consistency**: Clean standard decorators (`@mcp.tool()`) matching foundational FastMCP 1.0 documentation.

### Trade-Offs & Tier 3 Architecture
- **Trade-Off (Hand-Built Auth & Rate Limiting)**: Unlike standalone `fastmcp` 3.x, Anthropic's FastMCP 1.0 in `mcp<2` does not provide turn-key rate limiting or authentication. In Tier 3, rate limiting (token-bucket algorithm) and authentication/role checking must be hand-built using custom ASGI middleware on `mcp.sse_app()` or Python decorators wrapping tool functions.
- **Role Isolation (Tier 2)**: FastMCP instances will be partitioned into distinct server boundaries (`corpus-mcp` as read-only vs. `memory-cache-mcp` with scoped tool registration).

### Storage Access & Guardrail Boundaries
- **Qdrant Credential Status**: In our current local development infrastructure, Qdrant is running with default local access without a configured read-only API key. Read-only enforcement in `corpus-mcp` is currently architectural / procedural (the server exposes and calls search and point query APIs only; it contains no indexing or point upsert routines).
- **Guardrail vs. Security Boundary**: The automated test checking tool names against mutation keywords (`test_no_write_tools_registered_on_corpus_mcp`) is a **guardrail (contract validation / linting)** to prevent accidental exposure of write tools. It is **not an OS-level or cryptographic security boundary**; defense-in-depth requires read-only vector store credentials and network/container sandboxing in production.

---

## Consequences & Revisit Plan
- Pyproject dependencies specify `mcp>=1.30.0,<2.0.0`.
- All logging within stdio servers must strictly target `sys.stderr` to prevent JSON-RPC framing corruption on `stdout`.
- **Revisit Date**: **After Tier 3 completion (estimated Q1 2027)**. At that milestone, we will evaluate whether to unpin or upgrade the `mcp` dependency once Tier 3 rate limiting and caching architectures are complete.
