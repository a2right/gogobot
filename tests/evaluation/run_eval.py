# tests/evaluation/run_eval.py
"""
批量评估脚本：读取 test_set.json → 调用 agent → 评估结果 → 输出报告

使用方法：
  python3 tests/evaluation/run_eval.py
  python3 tests/evaluation/run_eval.py --cases tests/evaluation/test_set.json
  python3 tests/evaluation/run_eval.py --difficulty easy
  python3 tests/evaluation/run_eval.py --dry-run   # 只打印用例不调用agent
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.agent import run_with_all_tools
from tests.evaluation.evaluator import (
    evaluate_plan,
    summary_stats,
    print_report,
    print_summary,
    EvaluationResult,
)

RESULTS_FILE = "tests/evaluation/eval_results.json"
TEST_SET_FILE = "tests/evaluation/test_set.json"


# ─────────────────────────────────────────────────────────────
# 核心运行逻辑
# ─────────────────────────────────────────────────────────────

def run_single(case: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """运行单条测试用例，返回包含评估结果的完整记录。"""
    case_id    = case.get("id", str(uuid.uuid4())[:8])
    query      = case.get("query", "")
    difficulty = case.get("difficulty", "unknown")
    constraints = case.get("constraints", {})

    thread_id = f"eval_{case_id}_{uuid.uuid4().hex[:6]}"

    print(f"\n{'─'*60}")
    print(f"[{difficulty.upper()}] {case_id}: {query[:60]}...")

    start = time.time()
    try:
        response = run_with_all_tools(query, thread_id=thread_id)
        elapsed = round(time.time() - start, 1)
        print(f"  ✅ Agent responded in {elapsed}s")
    except Exception as e:
        elapsed = round(time.time() - start, 1)
        print(f"  ❌ Agent error after {elapsed}s: {e}")
        return {
            "id": case_id,
            "difficulty": difficulty,
            "query": query,
            "constraints": constraints,
            "thread_id": thread_id,
            "elapsed": elapsed,
            "error": str(e),
            "delivered": False,
            "eval": None,
        }

    # 从 chat_db.json 取回生成的 itinerary
    itinerary = _fetch_itinerary(thread_id)

    # 评估
    max_stops = constraints.get("max_stops_per_day", 6)
    result: EvaluationResult = evaluate_plan(itinerary, query, max_stops)

    if verbose:
        print_report(result)

    return {
        "id": case_id,
        "difficulty": difficulty,
        "query": query,
        "constraints": constraints,
        "thread_id": thread_id,
        "elapsed": elapsed,
        "error": None,
        "delivered": result.delivered,
        "eval": result.to_dict(),
        "itinerary": itinerary,
    }


def _fetch_itinerary(thread_id: str) -> Dict[str, Any]:
    """从 chat_db.json 读取最新生成的 itinerary。"""
    db_path = os.environ.get("GOGOBOT_DB_FILE", "chat_db.json")
    if not os.path.exists(db_path):
        return {}
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            db = json.load(f)
        container = db.get(thread_id, {})
        state = container.get("state", {}) if isinstance(container, dict) else {}
        versions = state.get("itinerary_versions", [])
        return versions[-1] if versions else {}
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────
# 批量运行
# ─────────────────────────────────────────────────────────────

def run_batch(
    cases: List[Dict[str, Any]],
    difficulty: Optional[str] = None,
    verbose: bool = False,
    dry_run: bool = False,
    delay: float = 1.0,
) -> List[Dict[str, Any]]:
    """批量运行所有测试用例。"""

    # 过滤难度
    if difficulty:
        cases = [c for c in cases if c.get("difficulty") == difficulty]

    print(f"\n{'═'*60}")
    print(f"  GoGoBot 批量评估")
    print(f"  用例数量: {len(cases)}"
          + (f" (仅 {difficulty})" if difficulty else ""))
    print(f"{'═'*60}")

    if dry_run:
        for c in cases:
            print(f"  [{c.get('difficulty','?').upper()}] "
                  f"{c.get('id','?')}: {c.get('query','')[:70]}")
        return []

    records = []
    for i, case in enumerate(cases, 1):
        print(f"\n进度: {i}/{len(cases)}")
        record = run_single(case, verbose=verbose)
        records.append(record)
        # 避免 Ollama 过热
        if i < len(cases):
            time.sleep(delay)

    return records


# ─────────────────────────────────────────────────────────────
# 报告生成
# ─────────────────────────────────────────────────────────────

def build_report(records: List[Dict[str, Any]]) -> None:
    """生成分难度的汇总报告，对标 TravelPlanner Table 3。"""

    # 按难度分组
    groups: Dict[str, List[EvaluationResult]] = {
        "easy": [], "medium": [], "hard": [], "all": []
    }

    for rec in records:
        if not rec.get("eval"):
            continue
        ev = rec["eval"]
        # 重建 EvaluationResult（简化版，只用 summary_stats 需要的字段）
        r = EvaluationResult(
            delivered=ev.get("delivered", False),
            commonsense_micro=ev.get("commonsense_micro", 0.0),
            commonsense_macro=ev.get("commonsense_macro", False),
            hard_micro=ev.get("hard_micro", 0.0),
            hard_macro=ev.get("hard_macro", False),
            final_pass=ev.get("final_pass", False),
            query=rec.get("query", ""),
        )
        diff = rec.get("difficulty", "unknown")
        if diff in groups:
            groups[diff].append(r)
        groups["all"].append(r)

    print(f"\n{'═'*60}")
    print("  分难度汇总报告（对标 TravelPlanner Table 3）")
    print(f"{'═'*60}")

    for label in ["easy", "medium", "hard", "all"]:
        results = groups[label]
        if not results:
            continue
        stats = summary_stats(results)
        n = stats["n_plans"]
        print(f"\n  ── {label.upper()} (n={n}) ──")
        print(f"  Delivery Rate      : {stats['delivery_rate']:.1%}")
        print(f"  Commonsense Micro  : {stats['commonsense_micro']:.1%}")
        print(f"  Commonsense Macro  : {stats['commonsense_macro']:.1%}")
        if stats.get("hard_micro", 0) > 0:
            print(f"  Hard Constraint Micro : {stats['hard_micro']:.1%}")
            print(f"  Hard Constraint Macro : {stats['hard_macro']:.1%}")
        print(f"  ⭐ Final Pass Rate  : {stats['final_pass_rate']:.1%}")

    # 找出失败的用例
    failed = [r for r in records if not (r.get("eval") or {}).get("final_pass")]
    if failed:
        print(f"\n  ── 失败用例 ({len(failed)}条) ──")
        for r in failed:
            ev = r.get("eval") or {}
            cs_fails = [c["name"] for c in ev.get("commonsense_details", [])
                        if not c["passed"]]
            hc_fails = [c["name"] for c in ev.get("hard_details", [])
                        if not c["passed"]]
            all_fails = cs_fails + hc_fails
            print(f"  [{r.get('difficulty','?').upper()}] {r.get('id','?')} "
                  f"→ 失败约束: {all_fails or ['未交付']}")

    print(f"\n{'═'*60}")


# ─────────────────────────────────────────────────────────────
# 保存 / 加载结果
# ─────────────────────────────────────────────────────────────

def save_results(records: List[Dict[str, Any]], path: str = RESULTS_FILE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        # itinerary 字段太大，存储时只保留评估相关字段
        slim = []
        for r in records:
            s = {k: v for k, v in r.items() if k != "itinerary"}
            slim.append(s)
        json.dump(slim, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存至: {path}")


def load_results(path: str = RESULTS_FILE) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GoGoBot 批量评估")
    parser.add_argument("--cases", default=TEST_SET_FILE,
                        help="测试集 JSON 文件路径")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"],
                        help="只运行指定难度的用例")
    parser.add_argument("--verbose", action="store_true",
                        help="打印每条用例的详细评估报告")
    parser.add_argument("--dry-run", action="store_true",
                        help="只列出用例，不实际调用 agent")
    parser.add_argument("--report-only", action="store_true",
                        help="从已有结果文件生成报告，不重新运行")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="每条用例之间的间隔秒数（默认1秒）")
    args = parser.parse_args()

    # 只生成报告
    if args.report_only:
        if not os.path.exists(RESULTS_FILE):
            print(f"找不到结果文件: {RESULTS_FILE}")
            print("请先运行评估：python3 tests/evaluation/run_eval.py")
            return
        records = load_results()
        build_report(records)
        return

    # 加载测试集
    with open(args.cases, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # 运行
    records = run_batch(
        cases,
        difficulty=args.difficulty,
        verbose=args.verbose,
        dry_run=args.dry_run,
        delay=args.delay,
    )

    if args.dry_run or not records:
        return

    # 保存 + 报告
    save_results(records)
    build_report(records)


if __name__ == "__main__":
    main()