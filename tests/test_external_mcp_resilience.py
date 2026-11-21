"""
test_external_mcp_resilience.py: Unit, resilience, and fault-injection tests
for the External API FastMCP Server.
Tests Token Bucket Rate Limiting, Exponential Backoff with Full Jitter,
Pydantic Schema Validation, Fallback Degradation, and Break It on Purpose Timeout.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch
import httpx

from evidenceflow.mcp.external_server import (
    TokenBucketRateLimiter,
    execute_with_resilience,
    get_release_metadata,
    get_security_advisories,
    check_version_eol_status,
    PostgresReleaseMetadata,
    SecurityAdvisory,
)


@pytest.mark.anyio
async def test_token_bucket_rate_limiter_burst_and_throttle():
    """
    Verifies that the Token Bucket rate limiter permits immediate bursts up to capacity,
    and throttles/times out once tokens are exhausted.
    """
    limiter = TokenBucketRateLimiter(rate=2.0, capacity=3.0)

    # 1. Immediate burst of 3 tokens should succeed instantly
    assert await limiter.acquire(tokens=1.0, timeout=0.1) is True
    assert await limiter.acquire(tokens=1.0, timeout=0.1) is True
    assert await limiter.acquire(tokens=1.0, timeout=0.1) is True

    # 2. 4th token immediately requested with small timeout should be throttled (False)
    assert await limiter.acquire(tokens=1.0, timeout=0.05) is False

    # 3. After waiting for refill (0.6s at 2 tokens/s = ~1.2 tokens), should succeed
    await asyncio.sleep(0.6)
    assert await limiter.acquire(tokens=1.0, timeout=0.1) is True


@pytest.mark.anyio
async def test_get_release_metadata_schema_success():
    """
    Verifies that a valid release metadata request returns a validated Pydantic model
    with proper fields, success=True, and is_degraded=False.
    """
    resp = await get_release_metadata(major_version="16")
    assert resp["success"] is True
    assert resp["is_degraded"] is False
    assert resp["retry_count"] == 0

    data = resp["data"]
    # Schema validation verification
    model = PostgresReleaseMetadata(**data)
    assert model.major_version == "16"
    assert model.latest_minor == "16.4"
    assert model.is_supported is True
    assert "2028" in model.eol_date


@pytest.mark.anyio
async def test_get_security_advisories_schema():
    """Verifies that CVE advisories conform strictly to SecurityAdvisory Pydantic schema."""
    resp = await get_security_advisories(major_version="16")
    assert resp["success"] is True
    assert resp["is_degraded"] is False

    advisories = resp["data"]
    assert len(advisories) >= 1

    # Verify first CVE adheres to schema
    cve = SecurityAdvisory(**advisories[0])
    assert cve.cve_id.startswith("CVE-")
    assert cve.severity in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
    assert "16.4" in cve.fixed_in_versions


@pytest.mark.anyio
async def test_unsupported_version_clean_error():
    """Verifies that an unsupported version returns a clean failure envelope without crashing."""
    resp = await get_release_metadata(major_version="9.6")
    assert resp["success"] is False
    assert "Unsupported PostgreSQL major version: '9.6'" in resp["error_notice"]


@pytest.mark.anyio
async def test_retry_on_transient_http_429():
    """
    Simulates transient HTTP 429 rate limit on attempts 1 and 2,
    recovering successfully on attempt 3.
    Verifies that the retry loop triggers with jitter and succeeds.
    """
    attempts = 0

    async def flaky_api_call():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            # Raise HTTP 429 Rate Limited
            req = httpx.Request("GET", "https://api.postgresql.org/meta")
            resp = httpx.Response(429, request=req)
            raise httpx.HTTPStatusError("429 Too Many Requests", request=req, response=resp)
        return "SUCCESS_DATA"

    limiter = TokenBucketRateLimiter(rate=50.0, capacity=50.0)
    res, retries, is_err, err_msg = await execute_with_resilience(
        flaky_api_call,
        max_retries=3,
        base_delay=0.01,
        max_delay=0.05,
        rate_limiter=limiter,
    )

    assert is_err is False
    assert res == "SUCCESS_DATA"
    assert retries == 2  # Succeeded after 2 retries (attempt 3)
    assert attempts == 3


@pytest.mark.anyio
async def test_outage_fallback_degradation():
    """
    Simulates a total external API outage (HTTP 503).
    Verifies that the server exhausts retries, does NOT crash the process,
    and returns a graceful degradation envelope with verified local fallback data.
    """
    resp = await get_release_metadata(major_version="16", simulate_outage=True)

    # Invariants under failure:
    assert resp["success"] is True  # Gracefully served fallback
    assert resp["is_degraded"] is True
    assert resp["retries_exhausted"] is True
    assert resp["retry_count"] == 3
    assert "External API degraded" in resp["error_notice"]

    # Verify fallback payload is still structurally valid
    data = resp["data"]
    model = PostgresReleaseMetadata(**data)
    assert model.major_version == "16"
    assert model.latest_minor == "16.4"


@pytest.mark.anyio
async def test_check_version_eol_status():
    """Verifies that check_version_eol_status recommends upgrading PG14 to PG16."""
    resp = await check_version_eol_status(major_version="14")
    assert resp["success"] is True

    data = resp["data"]
    assert data["major_version"] == "14"
    assert data["status"] == "ACTIVE_SUPPORT"
    assert "PostgreSQL 16" in data["recommended_upgrade_target"]


@pytest.mark.anyio
async def test_break_it_on_purpose_timeout_degradation():
    """
    Break It on Purpose (Day 17 Section 6):
    Temporarily forces a TimeoutException by setting timeout_override=0.001s.
    Verifies that:
      1. TimeoutException is caught by the resilience layer.
      2. The retry loop triggers and exhausts retries.
      3. The tool gracefully emits a clean degradation response without an unhandled crash.
    """
    resp = await get_release_metadata(
        major_version="15",
        timeout_override=0.001,  # 1ms deadline forces immediate TimeoutException
    )

    # Verification invariants:
    assert resp["success"] is True  # Does not crash agent
    assert resp["is_degraded"] is True
    assert resp["retries_exhausted"] is True
    assert resp["retry_count"] == 3
    assert "timed out" in resp["error_notice"].lower()

    # Valid fallback payload preserved
    data = resp["data"]
    assert data["major_version"] == "15"
    assert data["latest_minor"] == "15.8"
