"""
chunker.py: Hierarchy-aware HTML parser and chunker for PostgreSQL documentation.
Extracts sections, preserves hierarchy breadcrumbs, tags versions, and assigns SHA-256 hashes.

Review fixes applied (2026-10-24):
  [4.1] NFKD -> NFC normalization to preserve technical symbols
  [4.2] Stable chunk IDs derived from GUC HTML id attributes or content hash
  [4.3] Breadcrumb computed per <dt> instead of per <dl>
  [4.4] Release notes: extract only direct <li> text, not nested sub-lists
  [4.5] Title extraction uses word-based truncation instead of split(".")
  [4.6] Replaced print() with logging module
"""

import json
import hashlib
import logging
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Normalizes unicode (NFC to preserve technical symbols like >= and subscripts),
    collapses extra whitespaces, and strips stray anchor symbols.
    """
    if not text:
        return ""
    # [Fix 4.1] NFC preserves compatibility characters; NFKD was destroying them
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def compute_sha256(text: str) -> str:
    """
    Computes a deterministic SHA-256 hexadecimal hash for chunk content.
    Used for instant staleness detection and deduplication.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_guc_id(dt: Tag) -> str:
    """
    Extracts a stable GUC identifier from a <dt> tag's HTML id attribute.
    Example: <dt id="GUC-VACUUM-COST-LIMIT"> -> "vacuum-cost-limit"
    Falls back to a short content hash if no id attribute exists.
    """
    html_id = dt.get("id", "")
    if html_id:
        # Strip common prefixes and normalize
        stable = html_id.lower().replace("guc-", "").replace("_", "-")
        return stable

    # Fallback: short hash of the parameter name text
    text = clean_text(dt.get_text(" ", strip=True).replace("#", ""))
    return compute_sha256(text)[:10]


def find_section(tag: Tag, doc_root: Tag) -> str:
    """
    Finds the hierarchical section breadcrumb for a given DOM tag.
    First inspects parent section containers (sect1, sect2, sect3),
    and falls back to finding the nearest preceding heading (h1-h4).
    """
    crumbs: List[str] = []
    current = tag.parent

    # 1. Walk up ancestor tree within doc_root
    while current and current != doc_root:
        title_tag = None
        tp = current.find("div", class_="titlepage", recursive=False)
        if tp:
            title_tag = tp.find(["h1", "h2", "h3", "h4", "h5"])
        if not title_tag:
            title_tag = current.find(["h1", "h2", "h3", "h4", "h5"], class_="title", recursive=False)

        if title_tag:
            cleaned = clean_text(title_tag.get_text(" ", strip=True).replace("#", ""))
            if cleaned and cleaned not in crumbs:
                crumbs.append(cleaned)

        current = current.parent

    if crumbs:
        crumbs.reverse()
        return " > ".join(crumbs)

    # 2. Fallback: nearest preceding heading in the document
    prev_h = tag.find_previous(["h1", "h2", "h3", "h4"])
    if prev_h:
        return clean_text(prev_h.get_text(" ", strip=True).replace("#", ""))

    return "General"


def extract_chunks_from_html(html_text: str, doc_name: str, version: str) -> List[Dict[str, Any]]:
    """
    Parses a single PostgreSQL HTML manual and produces structured, hierarchy-aware chunks.
    Handles configuration parameter variable lists, adjacent <dt> elements with shared <dd>,
    and release notes change items.
    """
    soup = BeautifulSoup(html_text, "html.parser")
    doc_root = soup.find("div", id="docContent")

    if not doc_root:
        doc_root = soup.body if soup.body else soup

    chunks: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Strategy 1: Configuration Parameters (<dl class="variablelist">)
    # Handles 1 or more <dt> tags sharing a single <dd> explanation
    # -------------------------------------------------------------------------
    varlists = doc_root.find_all("dl", class_="variablelist")
    for dl in varlists:
        current_dts: List[Tag] = []
        for child in dl.children:
            if not isinstance(child, Tag):
                continue

            if child.name == "dt":
                current_dts.append(child)
            elif child.name == "dd" and current_dts:
                # [Fix 4.3] Compute breadcrumb from the FIRST <dt>, not the <dl>
                section_path = find_section(current_dts[0], doc_root)

                # Group multiple dt tags sharing this dd
                dt_titles = [
                    clean_text(dt.get_text(" ", strip=True).replace("#", ""))
                    for dt in current_dts
                ]
                combined_title = " / ".join([t for t in dt_titles if t])
                dd_text = clean_text(child.get_text(" ", strip=True))

                if combined_title and len(dd_text) >= 20:
                    # [Fix 4.2] Stable chunk ID from GUC HTML id or content hash
                    guc_id = _extract_guc_id(current_dts[0])
                    chunk_id = f"pg{version}-{Path(doc_name).stem}-{guc_id}"

                    full_content = f"Section: {section_path}\nParameter: {combined_title}\n\n{dd_text}"

                    chunks.append({
                        "id": chunk_id,
                        "version": version,
                        "doc_name": doc_name,
                        "section": section_path,
                        "title": combined_title,
                        "content": full_content,
                        "sha256": compute_sha256(full_content),
                        "char_count": len(full_content),
                    })

                current_dts = []

    # -------------------------------------------------------------------------
    # Strategy 2: Release Notes & Change Items (e.g. release-*.html)
    # -------------------------------------------------------------------------
    if not chunks:
        list_items = doc_root.find_all("li", class_="listitem") or doc_root.find_all("li")
        for li in list_items:
            # [Fix 4.4] Extract only direct text, not nested <ul>/<li> children
            nested_lists = li.find_all(["ul", "ol"], recursive=False)
            if nested_lists:
                # Clone the tag and remove nested lists to get only direct text
                li_copy = li.__copy__()
                for nested in li_copy.find_all(["ul", "ol"], recursive=False):
                    nested.decompose()
                text = clean_text(li_copy.get_text(" ", strip=True))
            else:
                text = clean_text(li.get_text(" ", strip=True))

            if len(text) < 40:
                continue

            section_path = find_section(li, doc_root)
            strong = li.find(["strong", "b"], recursive=False) or li.find(["strong", "b"])
            if strong:
                title = clean_text(strong.get_text(" ", strip=True))
            else:
                # [Fix 4.5] Word-based truncation instead of split(".") which breaks version numbers
                title = " ".join(text.split()[:12])

            # [Fix 4.2] Stable ID from content hash instead of positional counter
            content_hash = compute_sha256(text)[:10]
            chunk_id = f"pg{version}-{Path(doc_name).stem}-{content_hash}"

            full_content = f"Section: {section_path}\nFeature: {title}\n\n{text}"

            chunks.append({
                "id": chunk_id,
                "version": version,
                "doc_name": doc_name,
                "section": section_path,
                "title": title,
                "content": full_content,
                "sha256": compute_sha256(full_content),
                "char_count": len(full_content),
            })

    return chunks


def process_all_documentation(
    raw_dir: Path = Path("data/raw"),
    output_file: Path = Path("data/processed/chunks.jsonl"),
) -> int:
    """
    Iterates through all versioned raw HTML files, chunks them, and writes to JSON Lines (.jsonl).
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    all_chunks: List[Dict[str, Any]] = []

    for version_dir in sorted(raw_dir.glob("*")):
        if not version_dir.is_dir():
            continue
        version = version_dir.name

        for html_file in sorted(version_dir.glob("*.html")):
            html_text = html_file.read_text(encoding="utf-8")
            chunks = extract_chunks_from_html(html_text, html_file.name, version)
            all_chunks.extend(chunks)
            # [Fix 4.6] Use logging instead of print
            logger.info("[v%s] %s -> %d chunks", version, html_file.name, len(chunks))

    # Write out as JSON Lines (.jsonl)
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info("Processing complete! Generated %d chunks saved to %s", len(all_chunks), output_file)
    return len(all_chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    process_all_documentation()
