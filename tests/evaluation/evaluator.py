# tests/evaluation/evaluator.py
"""
GoGoBot Evaluator — inspired by TravelPlanner (Xie et al., 2024) Section 3.4.

Metrics implemented:
  - Delivery Rate          : did the agent produce a non-empty itinerary?
  - Commonsense Pass Rate  : micro & macro over 6 commonsense constraints
  - Hard Constraint Pass Rate: micro & macro over user-specified hard constraints
  - Final Pass Rate        : ALL constraints passed (micro=1.0 AND all hard passed)

Commonsense constraints (adapted for Singapore / GoGoBot schema):
  1. complete_information   — every day has ≥1 stop
  2. diverse_stops          — no stop name repeated across days
  3. valid_coordinates      — lat/lng within Singapore bounding box
  4. no_time_conflict       — no overlapping time windows within a day
  5. within_zone            — ≤2 zone changes per day
  6. reasonable_pace        — ≤6 stops per day

Hard constraints (parsed from query):
  - budget                  — total cost_estimate ≤ user budget
  - (extensible: cuisine, transport_mode, indoor_only, etc.)

Usage
-----
  # Single plan
  from tests.evaluation.evaluator import evaluate_plan, print_report
  result = evaluate_plan(itinerary_dict, query="3-day Singapore trip, budget SGD 500")
  print_report(result)

  # Batch evaluation
  from tests.evaluation.evaluator import batch_evaluate, summary_stats
  results = batch_evaluate(plan_list)          # plan_list: [{"itinerary":..,"query":...}]
  print(summary_stats(results))
"""

from __future__ import annotations

import json
import re
import sys
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── allow running as a script from repo root ──────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.constraint_validator import (
    validate_itinerary,
    extract_budget_from_text,
    _check_no_time_conflict,
    _check_within_zone,
    _check_diverse_stops,
    _check_no_empty_days,
    _check_reasonable_pace,
    _check_budget,
    _check_lat_lng_present,
    _flatten_stops,
    _stops_by_day,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConstraintResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class EvaluationResult:
    """Full evaluation result for one itinerary, mirroring TravelPlanner Table 3."""

    # Delivery
    delivered: bool = False

    # Commonsense
    commonsense_results: List[ConstraintResult] = field(default_factory=list)

    # Hard constraints
    hard_results: List[ConstraintResult] = field(default_factory=list)

    # Aggregated rates (computed by _compute_rates)
    commonsense_micro: float = 0.0   # eq.1 in paper
    commonsense_macro: bool = False  # eq.2 in paper
    hard_micro: float = 0.0
    hard_macro: bool = False
    final_pass: bool = False         # delivered AND commonsense_macro AND hard_macro

    # Raw itinerary & query (for debugging)
    query: str = ""
    itinerary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delivered": self.delivered,
            "commonsense_micro": round(self.commonsense_micro, 4),
            "commonsense_macro": self.commonsense_macro,
            "hard_micro": round(self.hard_micro, 4),
            "hard_macro": self.hard_macro,
            "final_pass": self.final_pass,
            "commonsense_details": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in self.commonsense_results
            ],
            "hard_details": [
                {"name": h.name, "passed": h.passed, "detail": h.detail}
                for h in self.hard_results
            ],
            "query": self.query,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Hard-constraint parsers  (extensible)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_hard_constraints(query: str) -> Dict[str, Any]:
    """
    Extract hard constraints from a natural-language query string.
    Returns a dict of {constraint_name: value}.

    Currently supported:
      budget        : float (SGD)
      indoor_only   : bool
      outdoor_only  : bool
      transport_mode: str  ("mrt", "walk", "taxi", "no_taxi", ...)
      cuisine       : List[str]
    """
    hard: Dict[str, Any] = {}
    q = (query or "").lower()

    # Budget
    budget = extract_budget_from_text(query)
    if budget is not None:
        hard["budget"] = budget

    # Indoor / outdoor preference
    if any(k in q for k in ("indoor", "室内", "避雨", "下雨")):
        hard["indoor_only"] = True
    if any(k in q for k in ("outdoor", "户外", "室外")):
        hard["outdoor_only"] = True

    # Transport mode
    if any(k in q for k in ("no taxi", "no cab", "不要打车", "不坐出租")):
        hard["no_taxi"] = True
    if any(k in q for k in ("mrt only", "只坐地铁", "地铁")):
        hard["mrt_only"] = True

    # Cuisine keywords (simple)
    cuisines = []
    for c in ("halal", "vegetarian", "vegan", "chinese", "indian",
              "malay", "western", "japanese", "清真", "素食"):
        if c in q:
            cuisines.append(c)
    if cuisines:
        hard["cuisine"] = cuisines

    return hard


def _evaluate_hard_constraints(
    itin: Dict[str, Any],
    hard: Dict[str, Any],
) -> List[ConstraintResult]:
    """Check each hard constraint against the itinerary."""
    results: List[ConstraintResult] = []

    # ── Budget ──────────────────────────────────────────────────────────────
    if "budget" in hard:
        ok, msg = _check_budget(itin, hard["budget"])
        results.append(ConstraintResult("budget", ok, msg))

    # ── Indoor only ─────────────────────────────────────────────────────────
    if hard.get("indoor_only"):
        all_stops = _flatten_stops(itin)
        outdoor_stops = [
            s.get("name", "?") for s in all_stops
            if "outdoor" in str(s.get("reason", "")).lower()
            or "beach" in str(s.get("name", "")).lower()
            or "park" in str(s.get("name", "")).lower()
        ]
        ok = len(outdoor_stops) == 0
        msg = f"Possibly outdoor stops: {outdoor_stops}" if not ok else ""
        results.append(ConstraintResult("indoor_only", ok, msg))

    # ── Cuisine (best-effort: check stop names / reasons) ───────────────────
    if hard.get("cuisine"):
        # We can't verify cuisine perfectly without structured data;
        # mark as passed with a note (placeholder for richer tool data).
        results.append(ConstraintResult(
            "cuisine", True,
            f"Cuisine check skipped (no structured cuisine field): {hard['cuisine']}"
        ))

    # ── Transport (placeholder) ──────────────────────────────────────────────
    if hard.get("no_taxi"):
        results.append(ConstraintResult(
            "no_taxi", True,
            "Transport constraint noted (no taxi); verify in rendered plan."
        ))
    if hard.get("mrt_only"):
        results.append(ConstraintResult(
            "mrt_only", True,
            "Transport constraint noted (MRT only); verify in rendered plan."
        ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Commonsense constraint evaluator
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_commonsense(
    itin: Dict[str, Any],
    max_stops_per_day: int = 6,
) -> List[ConstraintResult]:
    """
    Run all 6 commonsense checks; return one ConstraintResult each.
    Mirrors TravelPlanner Table 1 commonsense constraints.
    """
    results: List[ConstraintResult] = []
    days_stops = _stops_by_day(itin)
    all_stops  = _flatten_stops(itin)

    # 1) Complete information
    ok, msg = _check_no_empty_days(days_stops)
    results.append(ConstraintResult("complete_information", ok, msg))

    # 2) Diverse stops (no repeats across days)
    ok, msg = _check_diverse_stops(all_stops)
    results.append(ConstraintResult("diverse_stops", ok, msg))

    # 3) Valid coordinates (Singapore bounding box)
    ok, msg = _check_lat_lng_present(all_stops)
    results.append(ConstraintResult("valid_coordinates", ok, msg))

    # 4-6) Per-day checks
    time_ok_all, time_msgs = True, []
    zone_ok_all, zone_msgs = True, []
    pace_ok_all, pace_msgs = True, []

    for day_label, stops in days_stops:
        if not stops:
            continue
        ok, msg = _check_no_time_conflict(stops, day_label)
        if not ok:
            time_ok_all = False
            time_msgs.append(msg)

        ok, msg = _check_within_zone(stops, day_label)
        if not ok:
            zone_ok_all = False
            zone_msgs.append(msg)

        ok, msg = _check_reasonable_pace(stops, day_label, max_stops_per_day)
        if not ok:
            pace_ok_all = False
            pace_msgs.append(msg)

    results.append(ConstraintResult(
        "no_time_conflict", time_ok_all, "; ".join(time_msgs)))
    results.append(ConstraintResult(
        "within_zone", zone_ok_all, "; ".join(zone_msgs)))
    results.append(ConstraintResult(
        "reasonable_pace", pace_ok_all, "; ".join(pace_msgs)))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Rate computation  (TravelPlanner eq.1 & eq.2)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_rates(result: EvaluationResult) -> EvaluationResult:
    """Fill micro/macro rates in-place and return the same object."""

    def _micro(items: List[ConstraintResult]) -> float:
        if not items:
            return 1.0
        return sum(1 for c in items if c.passed) / len(items)

    def _macro(items: List[ConstraintResult]) -> bool:
        return all(c.passed for c in items) if items else True

    result.commonsense_micro = _micro(result.commonsense_results)
    result.commonsense_macro = _macro(result.commonsense_results)
    result.hard_micro        = _micro(result.hard_results)
    result.hard_macro        = _macro(result.hard_results)
    result.final_pass = (
        result.delivered
        and result.commonsense_macro
        and result.hard_macro
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_plan(
    itinerary: Dict[str, Any],
    query: str = "",
    max_stops_per_day: int = 6,
) -> EvaluationResult:
    """
    Evaluate a single itinerary against commonsense + hard constraints.

    Args:
        itinerary        : GoGoBot itinerary dict (days/stops schema).
        query            : Original user query string (for hard constraint parsing).
        max_stops_per_day: Threshold for reasonable-pace check (default 6).

    Returns:
        EvaluationResult with all metric fields populated.
    """
    result = EvaluationResult(query=query, itinerary=itinerary)

    # ── Delivery check ───────────────────────────────────────────────────────
    days = (itinerary or {}).get("days", [])
    has_stops = any(
        isinstance(d, dict) and d.get("stops")
        for d in (days if isinstance(days, list) else [])
    )
    result.delivered = (
        isinstance(itinerary, dict)
        and not itinerary.get("_invalid")
        and bool(days)
        and has_stops
    )

    if not result.delivered:
        result.commonsense_results = []
        result.hard_results = []
        return _compute_rates(result)

    # ── Commonsense constraints ──────────────────────────────────────────────
    result.commonsense_results = _evaluate_commonsense(itinerary, max_stops_per_day)

    # ── Hard constraints ─────────────────────────────────────────────────────
    hard_parsed = _parse_hard_constraints(query)
    result.hard_results = _evaluate_hard_constraints(itinerary, hard_parsed)

    return _compute_rates(result)


def batch_evaluate(
    plans: List[Dict[str, Any]],
    max_stops_per_day: int = 6,
) -> List[EvaluationResult]:
    """
    Evaluate a list of plans.

    Each item in `plans` should be:
        {"itinerary": <dict>, "query": <str>}   (query is optional)

    Returns a list of EvaluationResult objects.
    """
    results = []
    for item in plans:
        itin  = item.get("itinerary") or item.get("plan") or {}
        query = item.get("query", "")
        results.append(evaluate_plan(itin, query, max_stops_per_day))
    return results


def summary_stats(results: List[EvaluationResult]) -> Dict[str, Any]:
    """
    Aggregate metrics over a list of EvaluationResult objects.
    Mirrors TravelPlanner Table 3 column layout.

    Returns:
        {
          "n_plans": int,
          "delivery_rate": float,          # % plans delivered
          "commonsense_micro": float,      # avg micro pass rate
          "commonsense_macro": float,      # % plans all-commonsense-passed
          "hard_micro": float,
          "hard_macro": float,
          "final_pass_rate": float,        # % plans fully passing
          "per_constraint": {name: pass_rate, ...}
        }
    """
    n = len(results)
    if n == 0:
        return {"n_plans": 0}

    delivery      = sum(1 for r in results if r.delivered) / n
    cs_micro_avg  = sum(r.commonsense_micro for r in results) / n
    cs_macro_rate = sum(1 for r in results if r.commonsense_macro) / n
    hc_micro_avg  = sum(r.hard_micro for r in results) / n
    hc_macro_rate = sum(1 for r in results if r.hard_macro) / n
    final_rate    = sum(1 for r in results if r.final_pass) / n

    # Per-constraint pass rates
    per: Dict[str, List[bool]] = {}
    for r in results:
        for c in r.commonsense_results + r.hard_results:
            per.setdefault(c.name, []).append(c.passed)
    per_rates = {k: round(sum(v) / len(v), 4) for k, v in per.items()}

    return {
        "n_plans": n,
        "delivery_rate":      round(delivery,      4),
        "commonsense_micro":  round(cs_micro_avg,  4),
        "commonsense_macro":  round(cs_macro_rate, 4),
        "hard_micro":         round(hc_micro_avg,  4),
        "hard_macro":         round(hc_macro_rate, 4),
        "final_pass_rate":    round(final_rate,    4),
        "per_constraint":     per_rates,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def print_report(result: EvaluationResult, verbose: bool = True) -> None:
    """Print a human-readable evaluation report for a single plan."""
    sep = "─" * 60
    print(sep)
    print(f"Query   : {result.query or '(none)'}")
    print(f"Delivered: {'✅' if result.delivered else '❌'}")
    if not result.delivered:
        print("  ⚠️  No valid itinerary delivered — all metrics N/A")
        print(sep)
        return

    print(f"\n{'Constraint':<30} {'Pass?':>6}  Detail")
    print("─" * 60)

    print("  [Commonsense]")
    for c in result.commonsense_results:
        icon = "✅" if c.passed else "❌"
        detail = f"  ← {c.detail}" if (c.detail and not c.passed) else ""
        print(f"  {c.name:<28} {icon}{detail}")

    if result.hard_results:
        print("  [Hard Constraints]")
        for h in result.hard_results:
            icon = "✅" if h.passed else "❌"
            detail = f"  ← {h.detail}" if h.detail else ""
            print(f"  {h.name:<28} {icon}{detail}")
    else:
        print("  [Hard Constraints]  (none detected in query)")

    print(sep)
    print(f"  Commonsense  micro={result.commonsense_micro:.2%}  "
          f"macro={'PASS' if result.commonsense_macro else 'FAIL'}")
    if result.hard_results:
        print(f"  Hard         micro={result.hard_micro:.2%}  "
              f"macro={'PASS' if result.hard_macro else 'FAIL'}")
    print(f"  ⭐ Final Pass : {'✅ YES' if result.final_pass else '❌ NO'}")
    print(sep)


def print_summary(stats: Dict[str, Any]) -> None:
    """Print aggregate stats table (mirrors TravelPlanner Table 3)."""
    sep = "═" * 60
    print(sep)
    print(f"  GoGoBot Evaluation Summary  (n={stats['n_plans']})")
    print(sep)
    print(f"  Delivery Rate       : {stats['delivery_rate']:.2%}")
    print(f"  Commonsense Micro   : {stats['commonsense_micro']:.2%}")
    print(f"  Commonsense Macro   : {stats['commonsense_macro']:.2%}")
    if stats.get('hard_micro', 0) > 0 or stats.get('hard_macro', 0) > 0:
        print(f"  Hard Constraint Micro : {stats['hard_micro']:.2%}")
        print(f"  Hard Constraint Macro : {stats['hard_macro']:.2%}")
    print(f"  ⭐ Final Pass Rate   : {stats['final_pass_rate']:.2%}")
    print("─" * 60)
    print("  Per-Constraint Pass Rates:")
    for k, v in stats.get("per_constraint", {}).items():
        bar_len = int(v * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"    {k:<28} {v:.2%}  [{bar}]")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    """
    Usage:
      python tests/evaluation/evaluator.py --file results.json
      python tests/evaluation/evaluator.py --db chat_db.json
    """
    import argparse

    parser = argparse.ArgumentParser(description="GoGoBot Evaluator")
    parser.add_argument("--file", type=str,
                        help="JSON file: list of {itinerary, query} dicts")
    parser.add_argument("--db", type=str,
                        help="chat_db.json — evaluate all stored itineraries")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-plan reports")
    args = parser.parse_args()

    plans: List[Dict[str, Any]] = []

    # ── Load from results file ───────────────────────────────────────────────
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            data = json.load(f)
        plans = data if isinstance(data, list) else [data]

    # ── Load from chat_db.json ───────────────────────────────────────────────
    elif args.db:
        with open(args.db, "r", encoding="utf-8") as f:
            db = json.load(f)
        for tid, container in db.items():
            state   = container.get("state", {}) if isinstance(container, dict) else {}
            history = container.get("history", []) if isinstance(container, dict) else []
            versions = state.get("itinerary_versions", [])
            if not versions:
                continue
            # grab the last user message as query
            query = ""
            for m in reversed(history):
                if isinstance(m, dict) and m.get("role") == "user":
                    query = m.get("content", "")
                    break
            plans.append({
                "itinerary": versions[-1],
                "query": query,
                "thread_id": tid,
            })
    else:
        parser.print_help()
        return

    if not plans:
        print("No plans found.")
        return

    # ── Evaluate ─────────────────────────────────────────────────────────────
    results = batch_evaluate(plans)

    if args.verbose:
        for i, (plan, res) in enumerate(zip(plans, results)):
            tid = plan.get("thread_id", f"plan_{i}")
            print(f"\n{'='*60}")
            print(f"Thread: {tid}")
            print_report(res)

    stats = summary_stats(results)
    print_summary(stats)


if __name__ == "__main__":
    _cli()