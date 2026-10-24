"""
download_docs.py: Downloader for official PostgreSQL documentation.
Downloads versioned release notes and runtime configuration manuals for versions 14, 15, and 16.

Review fixes applied (2026-10-24):
  [3.1] Added retry logic with exponential backoff for network resilience
  [3.2] Removed unused `import sys`
  [3.3] Replaced magic 1000-byte threshold with docContent marker validation
  [3.4] Expanded page list with connection, logging, locks, replication configs
"""

import argparse
import logging
import time
from pathlib import Path
import httpx

logger = logging.getLogger(__name__)

DOC_PAGES = [
    "runtime-config-autovacuum.html",
    "runtime-config-resource.html",
    "runtime-config-wal.html",
    "runtime-config-query.html",
    "runtime-config-client.html",
    "runtime-config-developer.html",
    # [Fix 3.4] Additional configuration sections for broader corpus coverage
    "runtime-config-connection.html",
    "runtime-config-logging.html",
    "runtime-config-locks.html",
    "runtime-config-replication.html",
]

BASE_URL = "https://www.postgresql.org/docs"

# [Fix 3.1] Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds


def _is_valid_doc_page(file_path: Path) -> bool:
    """
    [Fix 3.3] Validates that a downloaded file contains actual PostgreSQL documentation
    by checking for the docContent marker, instead of relying on a magic byte threshold.
    """
    if not file_path.exists():
        return False
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        return 'id="docContent"' in content
    except OSError:
        return False


def download_version_docs(version: str, output_dir: Path) -> int:
    """
    Downloads key documentation pages for a specific PostgreSQL version.
    Uses exponential backoff retry on failures.
    """
    target_dir = output_dir / version
    target_dir.mkdir(parents=True, exist_ok=True)

    pages = [f"release-{version}.html"] + DOC_PAGES
    downloaded = 0

    headers = {
        "User-Agent": "EvidenceFlow-DocIngestion/1.0 (Portfolio Project; Technical Documentation Parser)"
    }

    with httpx.Client(timeout=30.0, headers=headers) as client:
        for page in pages:
            url = f"{BASE_URL}/{version}/{page}"
            target_path = target_dir / page

            # [Fix 3.3] Validate existing file has real doc content, not just non-empty
            if _is_valid_doc_page(target_path):
                logger.info("  [v%s] Already exists: %s (%d bytes)", version, page, target_path.stat().st_size)
                downloaded += 1
                continue

            # [Fix 3.1] Retry loop with exponential backoff
            success = False
            for attempt in range(MAX_RETRIES):
                try:
                    logger.info("  [v%s] Downloading: %s ... (attempt %d/%d)", version, url, attempt + 1, MAX_RETRIES)
                    response = client.get(url)
                    if response.status_code == 200:
                        target_path.write_text(response.text, encoding="utf-8")
                        logger.info("  [v%s] Saved: %s (%d chars)", version, page, len(response.text))
                        downloaded += 1
                        success = True
                        break
                    elif response.status_code == 404:
                        logger.warning("  [v%s] NOT FOUND (404): %s — skipping", version, page)
                        break  # Don't retry 404s
                    else:
                        logger.warning("  [v%s] HTTP %d for %s, retrying...", version, response.status_code, page)
                except Exception as e:
                    logger.warning("  [v%s] ERROR downloading %s: %s", version, page, e)

                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF_BASE ** (attempt + 1)
                    logger.info("  [v%s] Waiting %ds before retry...", version, wait)
                    time.sleep(wait)

            if not success and response.status_code != 404:
                logger.error("  [v%s] FAILED after %d attempts: %s", version, MAX_RETRIES, page)

    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Download official PostgreSQL documentation pages.")
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["14", "15", "16"],
        help="PostgreSQL major versions to download (default: 14 15 16)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save raw HTML files (default: data/raw)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Starting PostgreSQL documentation download for versions: %s", args.versions)
    total_downloaded = 0
    for v in args.versions:
        count = download_version_docs(v, args.output_dir)
        total_downloaded += count

    logger.info("\nDownload completed! Total pages successfully fetched: %d", total_downloaded)


if __name__ == "__main__":
    main()
