"""
diff_chunks.py: Deterministic staleness & change detector using SHA-256 chunk fingerprints.
Compares documentation revisions or cross-version parameters without calling an LLM.

Review fixes applied (2026-10-24):
  [5.1] Cross-version diff key now includes section to prevent title collisions
  [5.2] Removed unused Tuple import
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List


def load_chunks(file_path: Path) -> List[Dict[str, Any]]:
    """Loads chunks from a JSONL file."""
    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def diff_revisions(old_chunks: List[Dict[str, Any]], new_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compares two revisions of the same corpus by chunk ID.
    Identifies which chunks are UNCHANGED, CHANGED, ADDED, or REMOVED.
    """
    old_map = {c["id"]: c for c in old_chunks}
    new_map = {c["id"]: c for c in new_chunks}

    unchanged: List[str] = []
    changed: List[Dict[str, Any]] = []
    added: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []

    for cid, n_chunk in new_map.items():
        if cid not in old_map:
            added.append(n_chunk)
        else:
            o_chunk = old_map[cid]
            if n_chunk["sha256"] == o_chunk["sha256"]:
                unchanged.append(cid)
            else:
                changed.append({
                    "id": cid,
                    "title": n_chunk.get("title", ""),
                    "old_sha256": o_chunk["sha256"],
                    "new_sha256": n_chunk["sha256"],
                    "old_content": o_chunk.get("content", ""),
                    "new_content": n_chunk.get("content", ""),
                })

    for cid, o_chunk in old_map.items():
        if cid not in new_map:
            removed.append(o_chunk)

    return {
        "summary": {
            "total_old": len(old_chunks),
            "total_new": len(new_chunks),
            "unchanged": len(unchanged),
            "changed": len(changed),
            "added": len(added),
            "removed": len(removed),
        },
        "unchanged": unchanged,
        "changed": changed,
        "added": added,
        "removed": removed,
    }


def diff_cross_version(chunks: List[Dict[str, Any]], v1: str, v2: str) -> Dict[str, Any]:
    """
    Compares configuration parameters across two documentation versions (e.g. v15 vs v16).
    [Fix 5.1] Key includes section to prevent collisions on identical titles.
    """
    v1_chunks = {
        (c["title"], c["doc_name"], c.get("section", "")): c
        for c in chunks
        if c.get("version") == v1 and "runtime-config" in c.get("doc_name", "")
    }
    v2_chunks = {
        (c["title"], c["doc_name"], c.get("section", "")): c
        for c in chunks
        if c.get("version") == v2 and "runtime-config" in c.get("doc_name", "")
    }

    unchanged: List[str] = []
    changed: List[Dict[str, Any]] = []
    added: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []

    for key, c2 in v2_chunks.items():
        title, doc, section = key
        if key not in v1_chunks:
            added.append({"title": title, "doc_name": doc, "section": section, "sha256": c2["sha256"]})
        else:
            c1 = v1_chunks[key]
            if c1["sha256"] == c2["sha256"]:
                unchanged.append(title)
            else:
                changed.append({
                    "title": title,
                    "doc_name": doc,
                    "section": section,
                    f"v{v1}_sha256": c1["sha256"][:12],
                    f"v{v2}_sha256": c2["sha256"][:12],
                    f"v{v1}_content": c1["content"],
                    f"v{v2}_content": c2["content"],
                })

    for key, c1 in v1_chunks.items():
        if key not in v2_chunks:
            title, doc, section = key
            removed.append({"title": title, "doc_name": doc, "section": section, "sha256": c1["sha256"]})

    return {
        "v1": v1,
        "v2": v2,
        "summary": {
            f"v{v1}_total_params": len(v1_chunks),
            f"v{v2}_total_params": len(v2_chunks),
            "unchanged": len(unchanged),
            "changed": len(changed),
            "added_in_v2": len(added),
            "removed_in_v2": len(removed),
        },
        "unchanged": unchanged,
        "changed": changed,
        "added": added,
        "removed": removed,
    }


def main():
    parser = argparse.ArgumentParser(description="Deterministic SHA-256 chunk fingerprint diff tool.")
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"), help="Path to chunks.jsonl")
    parser.add_argument("--cross-version", nargs=2, metavar=("V1", "V2"), help="Compare parameters between two versions (e.g. 15 16)")
    parser.add_argument("--old-file", type=Path, help="Path to old chunks.jsonl for revision comparison")
    parser.add_argument("--new-file", type=Path, help="Path to new chunks.jsonl for revision comparison")
    args = parser.parse_args()

    if args.old_file and args.new_file:
        old_chunks = load_chunks(args.old_file)
        new_chunks = load_chunks(args.new_file)
        diff = diff_revisions(old_chunks, new_chunks)
        print("=== REVISION DIFF RESULT ===")
        print(json.dumps(diff["summary"], indent=2))
        return

    if args.cross_version:
        v1, v2 = args.cross_version
        chunks = load_chunks(args.chunks)
        diff = diff_cross_version(chunks, v1, v2)
        print(f"\n=======================================================")
        print(f"  CROSS-VERSION DIFF: PostgreSQL {v1} vs PostgreSQL {v2}")
        print(f"=======================================================")
        for k, v in diff["summary"].items():
            print(f"  {k:20s}: {v}")

        if diff["changed"]:
            print(f"\n[Parameters with Changed Definitions/Defaults (Staleness Trigger)]: {len(diff['changed'])}")
            for item in diff["changed"][:5]:
                print(f"  - {item['title']} in {item['doc_name']}")
                print(f"    v{v1} hash: {item[f'v{v1}_sha256']}  -->  v{v2} hash: {item[f'v{v2}_sha256']}")

        if diff["added"]:
            print(f"\n[New Parameters Introduced in v{v2}]: {len(diff['added'])}")
            for item in diff["added"][:5]:
                print(f"  + {item['title']} in {item['doc_name']}")

        if diff["removed"]:
            print(f"\n[Deprecated/Removed Parameters in v{v2}]: {len(diff['removed'])}")
            for item in diff["removed"]:
                print(f"  - {item['title']} in {item['doc_name']}")


if __name__ == "__main__":
    main()
