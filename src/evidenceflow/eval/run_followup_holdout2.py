"""
run_followup_holdout2.py: Evaluation runner for the second held-out 20-conversation
test set (data/benchmarks/followup_holdout2_20.json).

Conforms strictly to evaluation rules:
1. Committed dataset SHA-256 recorded prior to execution.
2. Adapts runner to schema without modifying test cases.
3. Turn 1 executed with its version_flag.
4. Turn 2 executed in the SAME thread_id with version_flag=null passed as version=None.
5. Scores resolution accuracy, false triggers (/4), end-to-end correct, and category breakdowns.
6. Flags any case that fell back to the offline writer during Gemini ON.
7. Outputs detailed per-case tables, rewritten questions, and check_by_eye responses.
"""

import argparse
import json
import logging
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from evidenceflow.config import get_settings
from evidenceflow.graph.workflow import run_pipeline
from evidenceflow.memory.init_neo4j import init_neo4j_schema, drop_graph
from evidenceflow.memory.graph_client import resolve_entity_for_turn
from evidenceflow.eval.metrics import check_value_accuracy

logger = logging.getLogger("evidenceflow.eval.holdout2")


def load_dataset(path: str = "data/benchmarks/followup_holdout2_20.json") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def evaluate_resolution(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
    exp_status = expected.get("status")
    act_status = actual.get("status")

    # Spec: resolved_multi_or_ambiguous: resolve to ALL earlier settings or ask which;
    # picking ONE single setting is a failure.
    if exp_status == "resolved_multi_or_ambiguous":
        if act_status == "ambiguous":
            return True
        if act_status == "resolved_multi":
            return True
        # If actual resolution picked a single setting, that is a failure
        return False

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
        return act_status == "ambiguous"

    return False


def is_header_only_answer(answer: str) -> bool:
    """Detect if an answer contains only a section header without substantive sentences."""
    if not answer or not answer.strip():
        return True
    lines = [line.strip() for line in answer.split("\n") if line.strip()]
    if len(lines) == 1 and lines[0].startswith("According to PostgreSQL") and "documentation on" in lines[0]:
        return True
    return False


def run_holdout2_exam(
    gemini_enabled: bool = False,
    delay_seconds: float = 2.0,
    dataset_path: str = "data/benchmarks/followup_holdout2_20.json",
) -> Dict[str, Any]:
    settings = get_settings()
    orig_key = settings.GEMINI_API_KEY
    if not gemini_enabled:
        settings.GEMINI_API_KEY = ""

    init_neo4j_schema()
    data = load_dataset(dataset_path)
    cases = data["cases"]
    dataset_sha256 = "7982fd1fae22f586261bb94959dc15e371ec12edf9a1bd37c106a87c52902598"
    git_commit = get_git_commit()

    mode_label = "Gemini ON (Live LLM)" if gemini_enabled else "Gemini OFF (Deterministic Fallback)"
    print("\n" + "=" * 95)
    print(f" EVIDENCEFLOW AI — HELD-OUT 2 FOLLOW-UP BENCHMARK ({mode_label})")
    print(f" Dataset SHA-256: {dataset_sha256}")
    print(f" Code Under Test Commit: {git_commit}")
    print("=" * 95)

    rows = []
    check_by_eye_items = []
    fallback_count = 0

    for idx, case in enumerate(cases, 1):
        cid = case["id"]
        category = case["category"]
        t1_spec = case["turn_1"]
        t2_spec = case["turn_2"]
        exp_res = case["expected_resolution"]
        exp_final = case["expected_final"]
        notes = case.get("notes", "")

        # Single continuous thread_id for both turns
        thread_id = f"holdout2_{cid}_{uuid.uuid4().hex[:6]}"
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
        t2_version_flag = t2_spec["version_flag"]  # null means do not pass version flag

        # Call memory resolver directly to capture intermediate resolution dict
        resolution = resolve_entity_for_turn(thread_id=thread_id, follow_up_query=t2_query)
        res_pass = evaluate_resolution(exp_res, resolution)

        # False trigger: passthrough case (g12-g15) that memory rewrote
        is_passthrough_expected = (exp_res.get("status") == "passthrough")
        is_false_trigger = is_passthrough_expected and (resolution.get("status") != "passthrough")

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

        if gemini_enabled and delay_seconds > 0:
            time.sleep(delay_seconds)

        # Determine actual outcome
        if t2_state.is_abstained:
            if t2_state.abstention_code == "AMBIGUOUS_FOLLOWUP":
                actual_outcome = "clarify"
            else:
                actual_outcome = "abstain"
        else:
            actual_outcome = "answer"

        # Check synthesis mode for Gemini ON runs
        t2_synthesis_mode = None
        for audit in t2_state.audit_trail:
            if audit.get("node") == "answer_writer":
                t2_synthesis_mode = audit.get("synthesis_mode")
                break

        fell_back_to_offline = False
        if gemini_enabled and t2_synthesis_mode == "fallback" and not t2_state.is_abstained:
            fell_back_to_offline = True
            fallback_count += 1

        # Check expected values
        expected_values = exp_final.get("expected_values", [])
        values_found = []
        for val in expected_values:
            if check_value_accuracy(t2_state.draft_answer, val):
                values_found.append(val)

        val_match = (len(values_found) == len(expected_values)) if expected_values else True

        # Check outcome match
        exp_outcome = exp_final.get("outcome")
        if exp_outcome == "either":
            outcome_match = True
        else:
            outcome_match = (actual_outcome == exp_outcome)

        # Check abstention code
        exp_code = exp_final.get("abstention_code")
        if exp_code:
            code_match = (t2_state.abstention_code == exp_code)
        else:
            code_match = True

        # Section header check for answers
        header_only = False
        if actual_outcome == "answer":
            if is_header_only_answer(t2_state.draft_answer):
                header_only = True

        # Must not cite setting / must not assert checks
        prohibited_assert_violated = False
        must_not_assert = exp_final.get("must_not_assert")
        if must_not_assert and must_not_assert.lower() in t2_state.draft_answer.lower():
            prohibited_assert_violated = True

        must_not_cite = exp_final.get("must_not_cite_setting")
        if must_not_cite:
            cited_chunks = [c.chunk_id for c in t2_state.citations]
            if any(must_not_cite.lower() in cid.lower() for cid in cited_chunks):
                prohibited_assert_violated = True

        # Overall end-to-end correctness
        # For check_by_eye cases, resolution and outcome match are preliminary indicators;
        # human review of draft answer provides final judgment.
        e2e_pass = (
            res_pass
            and outcome_match
            and val_match
            and code_match
            and not header_only
            and not prohibited_assert_violated
            and not fell_back_to_offline
        )

        cited_chunk_ids = [c.chunk_id for c in t2_state.citations]

        # Record check-by-eye items
        if exp_final.get("check_by_eye", False):
            check_by_eye_items.append({
                "id": cid,
                "category": category,
                "turn1_q": t1_spec["question"],
                "turn2_q": t2_query,
                "rewritten_q": resolution.get("rewritten_query"),
                "draft_answer": t2_state.draft_answer,
                "citations": cited_chunk_ids,
                "is_abstained": t2_state.is_abstained,
                "abstention_code": t2_state.abstention_code,
                "notes": notes,
                "must_not_assert": exp_final.get("must_not_assert"),
                "must_not_cite_setting": exp_final.get("must_not_cite_setting"),
            })

        row = {
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
            "synthesis_mode": t2_synthesis_mode,
            "fell_back_to_offline": fell_back_to_offline,
            "header_only": header_only,
            "citations": cited_chunk_ids,
            "draft_answer_preview": (t2_state.draft_answer[:120] + "...") if len(t2_state.draft_answer) > 120 else t2_state.draft_answer,
            "e2e_pass": e2e_pass,
            "notes": notes,
        }
        rows.append(row)

        drop_graph(thread_id=thread_id)

    # Restore setting
    settings.GEMINI_API_KEY = orig_key

    # Aggregations
    total_cases = len(rows)
    total_res_pass = sum(1 for r in rows if r["resolution_pass"])

    # Rule: false triggers divided by passthrough cases (g12-g15 = 4)
    passthrough_cases = [r for r in rows if r["exp_res"].get("status") == "passthrough"]
    passthrough_count = len(passthrough_cases)
    total_false_triggers = sum(1 for r in rows if r["is_false_trigger"])
    ft_pct = round((total_false_triggers / passthrough_count) * 100.0, 1) if passthrough_count > 0 else 0.0

    total_e2e_pass = sum(1 for r in rows if r["e2e_pass"])

    # Breakdown by category
    categories = sorted(list({r["category"] for r in rows}))
    cat_breakdown = {}
    for cat in categories:
        cat_rows = [r for r in rows if r["category"] == cat]
        c_tot = len(cat_rows)
        c_res = sum(1 for r in cat_rows if r["resolution_pass"])
        c_pt = [r for r in cat_rows if r["exp_res"].get("status") == "passthrough"]
        c_ft = sum(1 for r in cat_rows if r["is_false_trigger"])
        c_e2e = sum(1 for r in cat_rows if r["e2e_pass"])
        cat_breakdown[cat] = {
            "total": c_tot,
            "res_pass": c_res,
            "res_pct": round((c_res / c_tot) * 100.0, 1),
            "false_triggers": c_ft,
            "passthrough_total": len(c_pt),
            "e2e_pass": c_e2e,
            "e2e_pct": round((c_e2e / c_tot) * 100.0, 1),
        }

    return {
        "dataset_file": dataset_path,
        "dataset_sha256": dataset_sha256,
        "git_commit": git_commit,
        "mode": mode_label,
        "gemini_enabled": gemini_enabled,
        "total_cases": total_cases,
        "fallback_count": fallback_count,
        "resolution_accuracy_count": total_res_pass,
        "resolution_accuracy_pct": round((total_res_pass / total_cases) * 100.0, 1),
        "false_triggers_count": total_false_triggers,
        "passthrough_denominator": passthrough_count,
        "false_triggers_pct": ft_pct,
        "e2e_correct_count": total_e2e_pass,
        "e2e_correct_pct": round((total_e2e_pass / total_cases) * 100.0, 1),
        "category_breakdown": cat_breakdown,
        "rows": rows,
        "check_by_eye": check_by_eye_items,
    }


def print_holdout2_results(res: Dict[str, Any]) -> None:
    print("\n" + "=" * 95)
    print(f" RESULTS: {res['mode']}")
    print(f" Commit: {res['git_commit']} | Dataset SHA: {res['dataset_sha256']}")
    if res["gemini_enabled"]:
        print(f" Offline Fallbacks: {res['fallback_count']} (Excluded from valid Gemini ON)")
    print("=" * 95)
    print(f" Resolution Accuracy : {res['resolution_accuracy_count']} / {res['total_cases']} ({res['resolution_accuracy_pct']}%)")
    print(f" False Triggers      : {res['false_triggers_count']} / {res['passthrough_denominator']} ({res['false_triggers_pct']}%)")
    print(f" End-to-End Correct  : {res['e2e_correct_count']} / {res['total_cases']} ({res['e2e_correct_pct']}%)")
    print("-" * 95)
    print(" CATEGORY BREAKDOWN:")
    for cat, stats in res["category_breakdown"].items():
        ft_info = f"FT {stats['false_triggers']}/{stats['passthrough_total']}" if stats['passthrough_total'] > 0 else "FT 0/0"
        print(f"  * {cat:<32}: Res {stats['res_pass']}/{stats['total']} ({stats['res_pct']}%) | {ft_info} | E2E {stats['e2e_pass']}/{stats['total']} ({stats['e2e_pct']}%)")
    print("-" * 95)
    print(" DETAILED PER-CASE TABLE:")
    print(f"{'ID':<4} | {'Category':<28} | {'Exp Res':<18} | {'Act Res':<18} | {'Res':<4} | {'Outcome':<8} | {'Act Code':<18} | {'Vals':<8} | {'E2E':<4}")
    print("-" * 95)
    for r in res["rows"]:
        exp_r_str = f"{r['exp_res']['status']}:{r['exp_res'].get('setting') or '-'}:{r['exp_res'].get('version') or '-'}"[:18]
        act_r_str = f"{r['act_res']['status']}:{r['act_res'].get('setting') or '-'}:{r['act_res'].get('version') or '-'}"[:18]
        res_mark = "PASS" if r["resolution_pass"] else "FAIL"
        e2e_mark = "PASS" if r["e2e_pass"] else "FAIL"
        code_str = r['act_code'] or "-"
        vals_str = f"{len(r['values_found'])}/{len(r['expected_values'])}" if r['expected_values'] else "-"
        print(f"{r['id']:<4} | {r['category']:<28} | {exp_r_str:<18} | {act_r_str:<18} | {res_mark:<4} | {r['act_outcome']:<8} | {code_str:<18} | {vals_str:<8} | {e2e_mark:<4}")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Holdout 2 Benchmark.")
    parser.add_argument("--gemini", action="store_true", help="Run with Gemini ON")
    parser.add_argument("--delay", type=float, default=2.0, help="Query pacing delay in seconds")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    results = run_holdout2_exam(gemini_enabled=args.gemini, delay_seconds=args.delay)
    print_holdout2_results(results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to: {args.output}")
