"""
run_followup_holdout.py: Evaluation runner for the held-out 20-conversation test set
(data/benchmarks/followup_holdout_20.json).

Schema adaptation [Rule 3]:
  - Reads top-level 'cases' array.
  - Turn 1 executed with turn_1['version_flag'].
  - Turn 2 executed in the SAME thread_id; if version_flag is null, version=None is passed [Rule 4].
  - Runs once with Gemini OFF and once with Gemini ON [Rule 4].
  - Zero code modifications to resolver, pronoun rules, or word lists [Rule 8].
"""

import argparse
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from evidenceflow.config import get_settings
from evidenceflow.graph.workflow import run_pipeline
from evidenceflow.memory.init_neo4j import init_neo4j_schema, drop_graph
from evidenceflow.memory.graph_client import resolve_entity_for_turn
from evidenceflow.eval.metrics import check_value_accuracy

logger = logging.getLogger("evidenceflow.eval.holdout")


def load_holdout_dataset(path: str = "data/benchmarks/followup_holdout_20.json") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_resolution(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
    exp_status = expected.get("status")
    act_status = actual.get("status")

    # Handling 'resolved_multi_or_ambiguous': either ambiguous or multi-resolved is acceptable
    if exp_status == "resolved_multi_or_ambiguous":
        return act_status in ("ambiguous", "resolved")

    if exp_status != act_status:
        return False

    if exp_status == "resolved":
        exp_setting = expected.get("setting")
        exp_version = expected.get("version")
        setting_ok = (exp_setting is None) or (actual.get("entity_name") == exp_setting)
        version_ok = (exp_version is None) or (actual.get("target_version") == exp_version)
        return setting_ok and version_ok

    if exp_status == "passthrough":
        exp_setting = expected.get("setting")
        exp_version = expected.get("version")
        setting_ok = (exp_setting is None) or (actual.get("entity_name") == exp_setting)
        version_ok = (exp_version is None) or (actual.get("target_version") == exp_version)
        return setting_ok and version_ok

    if exp_status == "ambiguous":
        return True

    return False


def run_holdout_exam(gemini_enabled: bool = False, delay_seconds: float = 1.5) -> Dict[str, Any]:
    settings = get_settings()
    orig_key = settings.GEMINI_API_KEY
    if not gemini_enabled:
        settings.GEMINI_API_KEY = ""

    init_neo4j_schema()
    data = load_holdout_dataset()
    cases = data["cases"]

    mode_label = "Gemini ON (Live LLM)" if gemini_enabled else "Gemini OFF (Deterministic Fallback)"
    print("\n" + "=" * 90)
    print(f" EVIDENCEFLOW AI — HELD-OUT FOLLOW-UP BENCHMARK ({mode_label})")
    print(f" Dataset SHA-256: A4633E20FB769579309DDBAA2C892FA77002E2EED1857784B67729C0E39F31D9")
    print("=" * 90)

    rows = []
    check_by_eye_items = []

    for idx, case in enumerate(cases, 1):
        cid = case["id"]
        category = case["category"]
        t1_spec = case["turn_1"]
        t2_spec = case["turn_2"]
        exp_res = case["expected_resolution"]
        exp_final = case["expected_final"]
        notes = case.get("notes", "")

        # Single continuous thread_id for both turns [Rule 4]
        thread_id = f"holdout_{cid}_{uuid.uuid4().hex[:6]}"
        drop_graph(thread_id=thread_id)

        # -------------------------------------------------------------
        # TURN 1 EXECUTION
        # -------------------------------------------------------------
        t1_state = run_pipeline(
            query=t1_spec["question"],
            version=t1_spec["version_flag"],
            thread_id=thread_id,
            turn_number=1,
            enable_memory=True,
            use_postgres_checkpointer=False,
        )

        if gemini_enabled and delay_seconds > 0:
            time.sleep(delay_seconds)

        # -------------------------------------------------------------
        # TURN 2 MEMORY RESOLUTION INSPECTION
        # -------------------------------------------------------------
        t2_query = t2_spec["question"]
        t2_version_flag = t2_spec["version_flag"]  # null means do not pass version [Rule 4]

        # Call memory resolver directly to capture intermediate resolution dict
        resolution = resolve_entity_for_turn(thread_id=thread_id, follow_up_query=t2_query)
        res_pass = evaluate_resolution(exp_res, resolution)

        # Is this a false trigger? (Expected passthrough, but memory rewrote it)
        is_false_trigger = (exp_res["status"] == "passthrough" and resolution["status"] != "passthrough")

        # -------------------------------------------------------------
        # TURN 2 FULL PIPELINE EXECUTION
        # -------------------------------------------------------------
        t2_state = run_pipeline(
            query=t2_query,
            version=t2_version_flag,
            thread_id=thread_id,
            turn_number=2,
            enable_memory=True,
            use_postgres_checkpointer=False,
        )

        # Determine actual final outcome
        if t2_state.is_abstained:
            if t2_state.abstention_code == "AMBIGUOUS_FOLLOWUP":
                actual_outcome = "clarify"
            else:
                actual_outcome = "abstain"
        else:
            actual_outcome = "answer"

        # Check values in draft answer
        expected_values = exp_final.get("expected_values", [])
        values_found = []
        for val in expected_values:
            if check_value_accuracy(t2_state.draft_answer, val):
                values_found.append(val)

        val_match = (len(values_found) == len(expected_values)) if expected_values else True

        # Check final outcome
        exp_outcome = exp_final.get("outcome")
        if exp_outcome == "either":
            outcome_match = True
        else:
            outcome_match = (actual_outcome == exp_outcome)

        # Check code match if specified
        exp_code = exp_final.get("abstention_code")
        if exp_code:
            code_match = (t2_state.abstention_code == exp_code)
        else:
            code_match = True

        # End-to-end correctness
        e2e_pass = res_pass and outcome_match and val_match and code_match

        # Save check_by_eye for manual inspection [Rule 7]
        if exp_final.get("check_by_eye", False):
            cited_chunks = [
                c.chunk_id if hasattr(c, "chunk_id") else c.get("chunk_id")
                for c in t2_state.citations
            ]
            check_by_eye_items.append({
                "id": cid,
                "category": category,
                "turn1_q": t1_spec["question"],
                "turn2_q": t2_query,
                "rewritten_q": resolution.get("rewritten_query"),
                "draft_answer": t2_state.draft_answer,
                "citations": cited_chunks,
                "is_abstained": t2_state.is_abstained,
                "abstention_code": t2_state.abstention_code,
                "notes": notes,
            })

        rows.append({
            "id": cid,
            "category": category,
            "exp_res": exp_res,
            "act_res": {
                "status": resolution.get("status"),
                "setting": resolution.get("entity_name"),
                "version": resolution.get("target_version"),
            },
            "rewritten_query": resolution.get("rewritten_query"),
            "resolution_pass": res_pass,
            "is_false_trigger": is_false_trigger,
            "exp_outcome": exp_outcome,
            "act_outcome": actual_outcome,
            "act_code": t2_state.abstention_code,
            "values_found": values_found,
            "expected_values": expected_values,
            "e2e_pass": e2e_pass,
            "notes": notes,
        })

        # Teardown session nodes
        drop_graph(thread_id=thread_id)

    # Restore setting
    settings.GEMINI_API_KEY = orig_key

    # Aggregations [Rule 6]
    total_cases = len(rows)
    total_res_pass = sum(1 for r in rows if r["resolution_pass"])
    total_false_triggers = sum(1 for r in rows if r["is_false_trigger"])
    total_e2e_pass = sum(1 for r in rows if r["e2e_pass"])

    # By category aggregations
    categories = sorted(list({r["category"] for r in rows}))
    cat_breakdown = {}
    for cat in categories:
        cat_rows = [r for r in rows if r["category"] == cat]
        c_tot = len(cat_rows)
        c_res = sum(1 for r in cat_rows if r["resolution_pass"])
        c_ft = sum(1 for r in cat_rows if r["is_false_trigger"])
        c_e2e = sum(1 for r in cat_rows if r["e2e_pass"])
        cat_breakdown[cat] = {
            "total": c_tot,
            "res_pass": c_res,
            "false_triggers": c_ft,
            "e2e_pass": c_e2e,
        }

    return {
        "mode": mode_label,
        "total_cases": total_cases,
        "resolution_accuracy_count": total_res_pass,
        "resolution_accuracy_pct": round((total_res_pass / total_cases) * 100.0, 1),
        "false_triggers_count": total_false_triggers,
        "false_triggers_pct": round((total_false_triggers / total_cases) * 100.0, 1),
        "e2e_correct_count": total_e2e_pass,
        "e2e_correct_pct": round((total_e2e_pass / total_cases) * 100.0, 1),
        "category_breakdown": cat_breakdown,
        "rows": rows,
        "check_by_eye": check_by_eye_items,
    }


def print_holdout_results(res: Dict[str, Any]) -> None:
    print("\n" + "=" * 90)
    print(f" RESULTS: {res['mode']}")
    print("=" * 90)
    print(f" Resolution Accuracy : {res['resolution_accuracy_count']} / {res['total_cases']} ({res['resolution_accuracy_pct']}%)")
    print(f" False Triggers      : {res['false_triggers_count']} / {res['total_cases']} ({res['false_triggers_pct']}%)")
    print(f" End-to-End Correct  : {res['e2e_correct_count']} / {res['total_cases']} ({res['e2e_correct_pct']}%)")
    print("-" * 90)
    print(" CATEGORY BREAKDOWN:")
    for cat, stats in res["category_breakdown"].items():
        print(f"  * {cat:<32}: Res {stats['res_pass']}/{stats['total']} | FalseTrig {stats['false_triggers']} | E2E {stats['e2e_pass']}/{stats['total']}")
    print("-" * 90)
    print(" DETAILED PER-CASE TABLE [Rule 5]:")
    print(f"{'ID':<4} | {'Category':<28} | {'Exp Res':<18} | {'Act Res':<18} | {'Res':<4} | {'Outcome':<8} | {'E2E':<4}")
    print("-" * 90)
    for r in res["rows"]:
        exp_r_str = f"{r['exp_res']['status']}:{r['exp_res'].get('setting') or '-'}"[:18]
        act_r_str = f"{r['act_res']['status']}:{r['act_res'].get('setting') or '-'}"[:18]
        res_mark = "PASS" if r["resolution_pass"] else "FAIL"
        e2e_mark = "PASS" if r["e2e_pass"] else "FAIL"
        print(f"{r['id']:<4} | {r['category']:<28} | {exp_r_str:<18} | {act_r_str:<18} | {res_mark:<4} | {r['act_outcome']:<8} | {e2e_mark:<4}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Holdout Benchmark.")
    parser.add_argument("--gemini", action="store_true", help="Run with Gemini ON")
    parser.add_argument("--delay", type=float, default=1.5, help="Query pacing delay in seconds")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    results = run_holdout_exam(gemini_enabled=args.gemini, delay_seconds=args.delay)
    print_holdout_results(results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to: {args.output}")
