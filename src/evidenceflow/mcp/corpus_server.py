"""
corpus_server.py: Standalone read-only FastMCP server for EvidenceFlow AI.
Exposes tools for searching, inspecting, and diffing PostgreSQL documentation
across versions (14, 15, 16) without write permissions.

Uses stdio transport for JSON-RPC communication; all server diagnostics strictly
log to sys.stderr to preserve protocol framing.
"""

import sys
import re
import difflib
import logging
from typing import Any, Dict, List, Literal, Optional, Annotated
from pydantic import Field
from mcp.server.fastmcp import FastMCP
from qdrant_client.http import models

from evidenceflow.config import get_settings
from evidenceflow.storage import get_qdrant_client
from evidenceflow.storage.qdrant_client import get_evidence_qdrant_client

# Configure logging exclusively to sys.stderr to avoid corrupting stdio transport
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [corpus-mcp] %(message)s"
)
logger = logging.getLogger("corpus-mcp")

VALID_VERSIONS = {"14", "15", "16"}
CHUNK_ID_REGEX = re.compile(r"^pg(14|15|16)-[a-z0-9_-]+$", re.IGNORECASE)

PostgresVersion = Literal["14", "15", "16"]
PassageLimit = Annotated[int, Field(ge=1, le=20, default=5, description="Maximum number of passages to return (1-20, default 5)")]
QueryStr = Annotated[str, Field(max_length=500, description="Technical search query or parameter name (max 500 chars)")]
SettingStr = Annotated[str, Field(max_length=100, description="Parameter or section name to compare (e.g. 'log_checkpoints', 'vacuum_defer_cleanup_age')")]

# FastMCP Server Instance
mcp = FastMCP("corpus-mcp")


def _extract_default_value(content: Optional[str]) -> Optional[str]:
    """Extracts the documented default value from parameter documentation content."""
    if not content:
        return None
    m = re.search(r"The default (?:value )?is\s+([^,.\n]+)", content, re.IGNORECASE)
    return m.group(1).strip() if m else None


def _find_parameter_chunk(setting: str, version: str) -> Optional[Dict[str, Any]]:
    """
    Locates the documentation chunk in Qdrant that defines a specific setting in a version.
    Prioritizes runtime-config parameter chunks over release note mentions.
    """
    client = get_evidence_qdrant_client()
    clean_setting = setting.strip().lower()
    clean_param_id = clean_setting.replace("_", "-")

    # 1. Search candidate chunks in Qdrant scoped to version
    candidates = client.search(query=clean_setting, version=version, limit=5)
    if not candidates:
        return None

    # Priority 1: Direct runtime-config chunk with setting name in chunk_id
    for h in candidates:
        cid = h.get("chunk_id", "").lower()
        if "runtime-config" in cid and clean_param_id in cid:
            return h

    # Priority 2: Chunk title or parameter header directly names the setting
    for h in candidates:
        title = h.get("title", "").lower()
        content = h.get("content", "").lower()
        if clean_setting in title or f"parameter: {clean_setting}" in content:
            if "release" not in h.get("doc_name", "").lower():
                return h

    # Priority 3: Fall back to best candidate if it contains the parameter name in ID
    for h in candidates:
        if clean_param_id in h.get("chunk_id", "").lower():
            return h

    return None


@mcp.tool()
def search_passages(
    query: QueryStr,
    version: PostgresVersion = "16",
    limit: PassageLimit = 5,
) -> Dict[str, Any]:
    """
    Searches official PostgreSQL technical documentation passages in Qdrant hybrid index.
    Always cite the returned chunk_id using footnote format [^chunk_id] when answering.
    If no relevant passages are found, declare insufficient evidence.

    Args:
        query: Technical search query or parameter name (max 500 chars).
        version: Target PostgreSQL documentation version (14, 15, or 16).
        limit: Maximum number of passages to return (1-20, default 5).
    """
    # 1. Input Validation
    if not query or not query.strip():
        return {"status": "error", "error": "Query cannot be empty."}

    query_clean = query.strip()
    if len(query_clean) > 500:
        return {"status": "error", "error": f"Query length ({len(query_clean)}) exceeds maximum allowed (500 chars)."}

    version_str = str(version).strip()
    if version_str not in VALID_VERSIONS:
        return {
            "status": "error",
            "error": f"Invalid PostgreSQL version '{version}'. Supported versions: {sorted(list(VALID_VERSIONS))}.",
        }

    clamped_limit = max(1, min(int(limit), 20))

    # 2. Execution via Qdrant interface
    try:
        client = get_evidence_qdrant_client()
        raw_results = client.search(query=query_clean, version=version_str, limit=clamped_limit)
        passages = [
            {
                "chunk_id": p.get("chunk_id"),
                "doc_name": p.get("doc_name"),
                "section": p.get("section"),
                "title": p.get("title"),
                "snippet": p.get("content", "")[:350] + ("..." if len(p.get("content", "")) > 350 else ""),
                "score": round(float(p.get("score", 0.0)), 4),
                "sha256": p.get("sha256"),
            }
            for p in raw_results
        ]
        return {
            "status": "success",
            "version": version_str,
            "query": query_clean,
            "count": len(passages),
            "passages": passages,
            "guidance": (
                "Always cite the returned chunk_id using footnote format [^chunk_id]. "
                "If no relevant passages were found, state that the documentation has insufficient evidence."
            ),
        }
    except Exception as e:
        logger.error("search_passages failed for query '%s': %s", query_clean, e)
        return {"status": "error", "error": str(e)}


@mcp.tool()
def get_passage(chunk_id: str) -> Dict[str, Any]:
    """
    Retrieves the full content and metadata for an exact documentation chunk by its chunk_id
    from the Qdrant vector database.
    Use this to inspect the complete context of a citation.
    Returns status 'not_found' if chunk_id does not exist.

    Args:
        chunk_id: Exact chunk identifier (e.g. 'pg16-runtime-config-resource-work-mem').
    """
    if not chunk_id or not isinstance(chunk_id, str):
        return {"status": "error", "error": "chunk_id must be a non-empty string."}

    cid = chunk_id.strip()
    if len(cid) > 128:
        return {"status": "error", "error": "chunk_id exceeds maximum length of 128 chars."}

    if not CHUNK_ID_REGEX.match(cid):
        return {
            "status": "error",
            "error": f"Malformed chunk_id format '{cid}'. Expected format matching 'pg<14|15|16>-<identifier>'.",
        }

    # Query Qdrant directly (strictly Qdrant only, no local disk fallback)
    try:
        q_client = get_qdrant_client()
        settings = get_settings()
        collection_name = settings.QDRANT_COLLECTION

        points, _ = q_client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="chunk_id", match=models.MatchValue(value=cid))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            return {
                "status": "not_found",
                "message": f"Passage '{cid}' not found in Qdrant collection '{collection_name}'.",
            }

        payload = points[0].payload or {}
        return {
            "status": "success",
            "chunk_id": cid,
            "version": payload.get("version"),
            "doc_name": payload.get("doc_name"),
            "section": payload.get("section"),
            "title": payload.get("title"),
            "content": payload.get("content"),
            "sha256": payload.get("sha256"),
        }
    except Exception as e:
        logger.error("get_passage failed for chunk_id '%s': %s", cid, e)
        return {"status": "error", "error": str(e)}


@mcp.tool()
def list_versions() -> Dict[str, Any]:
    """
    Lists the PostgreSQL versions currently indexed in the EvidenceFlow corpus (14, 15, 16)
    with status and description.
    """
    return {
        "status": "success",
        "supported_versions": sorted(list(VALID_VERSIONS)),
        "default_version": "16",
        "description": "PostgreSQL technical documentation versions indexed in EvidenceFlow.",
        "version_details": {
            "14": {"status": "indexed", "scope": "PostgreSQL 14.x official documentation"},
            "15": {"status": "indexed", "scope": "PostgreSQL 15.x official documentation"},
            "16": {"status": "indexed", "scope": "PostgreSQL 16.x official documentation"},
        },
    }


@mcp.tool()
def diff_sections(
    setting: SettingStr,
    v1: PostgresVersion = "14",
    v2: PostgresVersion = "16",
) -> Dict[str, Any]:
    """
    Compares a specific configuration parameter or documentation section across two PostgreSQL
    versions (e.g. 14 vs 16).
    Returns structured fields indicating whether the parameter was added, removed, or changed,
    including defaults and unified diff. Use to detect cross-version behavioral changes.

    Args:
        setting: Parameter or section name to compare (e.g. 'log_checkpoints', 'vacuum_defer_cleanup_age').
        v1: First PostgreSQL version (14, 15, 16; default '14').
        v2: Second PostgreSQL version (14, 15, 16; default '16').
    """
    if not setting or not setting.strip():
        return {"status": "error", "error": "Setting name cannot be empty."}

    setting_clean = setting.strip().lower()
    if len(setting_clean) > 100:
        return {"status": "error", "error": "Setting name exceeds 100 characters."}

    v1_str, v2_str = str(v1).strip(), str(v2).strip()
    if v1_str not in VALID_VERSIONS or v2_str not in VALID_VERSIONS:
        return {
            "status": "error",
            "error": f"Invalid version pair ('{v1}', '{v2}'). Supported versions: {sorted(list(VALID_VERSIONS))}.",
        }

    chunk_1 = _find_parameter_chunk(setting_clean, v1_str)
    chunk_2 = _find_parameter_chunk(setting_clean, v2_str)

    found_v1 = chunk_1 is not None
    found_v2 = chunk_2 is not None

    content_v1 = chunk_1.get("content", "") if chunk_1 else ""
    content_v2 = chunk_2.get("content", "") if chunk_2 else ""

    default_v1 = _extract_default_value(content_v1)
    default_v2 = _extract_default_value(content_v2)

    cid_1 = chunk_1.get("chunk_id") if chunk_1 else None
    cid_2 = chunk_2.get("chunk_id") if chunk_2 else None

    # Determine change status and summary
    if not found_v1 and not found_v2:
        return {
            "status": "not_found",
            "setting": setting_clean,
            "version_1": v1_str,
            "version_2": v2_str,
            "found_v1": False,
            "found_v2": False,
            "changed": False,
            "default_v1": None,
            "default_v2": None,
            "chunk_id_v1": None,
            "chunk_id_v2": None,
            "summary": f"Parameter '{setting_clean}' was not found in either PostgreSQL {v1_str} or PostgreSQL {v2_str} documentation.",
            "diff": "",
        }

    if found_v1 and not found_v2:
        changed = True
        summary = (
            f"Parameter '{setting_clean}' was present in PostgreSQL {v1_str} "
            f"(default: {default_v1 or 'unspecified'}) but was removed or absent in PostgreSQL {v2_str}."
        )
        diff_lines = list(
            difflib.unified_diff(
                content_v1.splitlines(keepends=True),
                [],
                fromfile=f"postgresql-{v1_str}/{cid_1}",
                tofile=f"postgresql-{v2_str}/(removed)",
            )
        )
        diff_text = "".join(diff_lines)

    elif not found_v1 and found_v2:
        changed = True
        summary = (
            f"Parameter '{setting_clean}' is newly introduced in PostgreSQL {v2_str} "
            f"(default: {default_v2 or 'unspecified'}) and did not exist in PostgreSQL {v1_str}."
        )
        diff_lines = list(
            difflib.unified_diff(
                [],
                content_v2.splitlines(keepends=True),
                fromfile=f"postgresql-{v1_str}/(non-existent)",
                tofile=f"postgresql-{v2_str}/{cid_2}",
            )
        )
        diff_text = "".join(diff_lines)

    else:
        # Both found — compare content and defaults
        is_text_diff = content_v1.strip() != content_v2.strip()
        is_default_diff = (default_v1 is not None and default_v2 is not None and default_v1 != default_v2)
        changed = is_text_diff or is_default_diff

        if changed:
            if is_default_diff:
                summary = (
                    f"Parameter '{setting_clean}' changed default value between PostgreSQL {v1_str} "
                    f"('{default_v1}') and PostgreSQL {v2_str} ('{default_v2}')."
                )
            else:
                summary = (
                    f"Parameter '{setting_clean}' documentation was modified between PostgreSQL {v1_str} "
                    f"and PostgreSQL {v2_str} (default remained '{default_v1}')."
                )
            diff_lines = list(
                difflib.unified_diff(
                    content_v1.splitlines(keepends=True),
                    content_v2.splitlines(keepends=True),
                    fromfile=f"postgresql-{v1_str}/{cid_1}",
                    tofile=f"postgresql-{v2_str}/{cid_2}",
                )
            )
            diff_text = "".join(diff_lines)
        else:
            summary = (
                f"Parameter '{setting_clean}' is unchanged between PostgreSQL {v1_str} and PostgreSQL {v2_str} "
                f"(default: {default_v1 or 'unspecified'})."
            )
            diff_text = ""

    return {
        "status": "success",
        "setting": setting_clean,
        "version_1": v1_str,
        "version_2": v2_str,
        "found_v1": found_v1,
        "found_v2": found_v2,
        "changed": changed,
        "default_v1": default_v1,
        "default_v2": default_v2,
        "chunk_id_v1": cid_1,
        "chunk_id_v2": cid_2,
        "summary": summary,
        "diff": diff_text,
    }


def main():
    """Runs the FastMCP corpus server using stdio transport."""
    logger.info("Starting EvidenceFlow FastMCP corpus server on stdio transport...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
