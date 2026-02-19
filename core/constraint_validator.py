# core/constraint_validator.py
"""
Constraint validator inspired by TravelPlanner's evaluation framework.
Checks commonsense + hard constraints on a GoGoBot itinerary JSON.

Itinerary schema (your format):
{
  "days": [
    {
      "date": "D1",
      "stops": [
        {
          "id": "string",
          "name": "string",
          "zone": "string",
          "lat": number,
          "lng": number,
          "start": "HH:MM",
          "end": "HH:MM",
          "reason": "string",
          "evidence": [...]
        }
      ]
    }
  ],
  "notes": "string"
}

Changes vs original:
  [Fix 1] _check_budget: total==0 now returns a soft warning instead of
          silently passing, preventing false-pass when cost_estimate is absent.
  [Fix 2] _check_within_zone: added zone-continuity check so non-adjacent
          zone pairs are penalised even when len(unique) <= 2.
  [Fix 3] micro_pass_rate now aggregates per-day checks at the *category*
          level (7 fixed categories) so scores are comparable across trip
          lengths, matching TravelPlanner Eq.1 semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Singapore approximate bounding box (used in multiple checks)
_SG_LAT = (1.15, 1.50)
_SG_LNG = (103.60, 104.10)

# Zone adjacency map for Singapore planning regions.
# Only zones listed as neighbours are considered "adjacent".
# Extend this dict as your zone taxonomy grows.
_ZONE_ADJACENCY: Dict[str, Set[str]] = {
    "Central":   {"North", "East", "West", "Northeast"},
    "North":     {"Central", "Northeast", "Northwest"},
    "Northeast": {"Central", "North", "East"},
    "East":      {"Central", "Northeast"},
    "West":      {"Central", "Northwest"},
    "Northwest": {"North", "West"},
}


# ─────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────
@dataclass
class ValidationResult:
    passed: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)          # [Fix 1] soft warnings
    constraint_scores: Dict[str, bool] = field(default_factory=dict)
    micro_pass_rate: float = 1.0   # ratio of *categories* passed (7 fixed)
    penalty: float = 0.0           # quality penalty (feeds into _itinerary_quality)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": self.issues,
            "warnings": self.warnings,
            "constraint_scores": self.constraint_scores,
            "micro_pass_rate": self.micro_pass_rate,
            "penalty": self.penalty,
        }


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────
def _parse_time(t: str) -> Optional[int]:
    """Convert 'HH:MM' → minutes since midnight. Returns None on failure."""
    try:
        h, m = t.strip().split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return None


def _flatten_stops(itin: Dict[str, Any]) -> List[Dict[str, Any]]:
    stops = []
    for d in (itin or {}).get("days", []):
        stops.extend(d.get("stops", []) if isinstance(d, dict) else [])
    return [s for s in stops if isinstance(s, dict)]


def _stops_by_day(itin: Dict[str, Any]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    result = []
    for d in (itin or {}).get("days", []):
        if isinstance(d, dict):
            result.append((d.get("date", ""), d.get("stops", [])))
    return result


# ─────────────────────────────────────────────
# Individual constraint checks
# (each returns (passed: bool, issue_msg: str))
# ─────────────────────────────────────────────

def _check_no_time_conflict(stops: List[Dict[str, Any]], day_label: str) -> Tuple[bool, str]:
    """Stops within a day must not have overlapping time windows."""
    times = []
    for s in stops:
        st = _parse_time(s.get("start", ""))
        en = _parse_time(s.get("end", ""))
        if st is not None and en is not None:
            times.append((st, en, s.get("name", "?")))

    times.sort(key=lambda x: x[0])
    for i in range(1, len(times)):
        if times[i][0] < times[i - 1][1]:
            return False, (
                f"[{day_label}] Time conflict: '{times[i-1][2]}' ends "
                f"{times[i-1][1]//60:02d}:{times[i-1][1]%60:02d} "
                f"but '{times[i][2]}' starts "
                f"{times[i][0]//60:02d}:{times[i][0]%60:02d}"
            )
    return True, ""


def _check_within_zone(stops: List[Dict[str, Any]], day_label: str) -> Tuple[bool, str]:
    """
    Adapted from TravelPlanner's 'Within Current City'.

    [Fix 2] Two-level check:
      (a) Count distinct zones — more than 2 is always a hard fail.
      (b) If exactly 2 zones, verify they are adjacent in _ZONE_ADJACENCY;
          non-adjacent pairs (e.g. East ↔ West) are also flagged.
    This prevents cases like East+West passing simply because len(unique)==2,
    even though travelling between them takes 1 h+ on the MRT.
    """
    zones = [s.get("zone", "").strip() for s in stops if s.get("zone", "").strip()]
    if not zones:
        return True, ""

    unique = list(dict.fromkeys(zones))  # preserve visit order, deduplicate

    # (a) Hard fail: too many distinct zones
    if len(unique) > 2:
        return False, (
            f"[{day_label}] Too many zone switches: {unique}. "
            "Keep stops in ≤2 zones per day."
        )

    # (b) Soft fail: 2 zones but not adjacent
    if len(unique) == 2:
        z1, z2 = unique[0], unique[1]
        neighbours = _ZONE_ADJACENCY.get(z1, set())
        if z2 not in neighbours:
            return False, (
                f"[{day_label}] Non-adjacent zone jump: '{z1}' → '{z2}'. "
                "These zones are far apart in Singapore — consider splitting across days."
            )

    return True, ""


def _check_diverse_stops(all_stops: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Adapted from TravelPlanner's 'Diverse Attractions'.
    Same stop (by name) should not appear on multiple days.
    """
    seen: Dict[str, int] = {}
    duplicates = []
    for s in all_stops:
        name = (s.get("name") or "").strip().lower()
        if not name:
            continue
        seen[name] = seen.get(name, 0) + 1
        if seen[name] == 2:
            duplicates.append(s.get("name", name))

    if duplicates:
        return False, f"Duplicate stops across days: {duplicates}"
    return True, ""


def _check_no_empty_days(days_stops: List[Tuple[str, List[Dict]]]) -> Tuple[bool, str]:
    """
    Adapted from TravelPlanner's 'Complete Information'.
    Every day must have at least 1 stop.
    """
    empty = [label for label, stops in days_stops if not stops]
    if empty:
        return False, f"Days with no stops: {empty}"
    return True, ""


def _check_reasonable_pace(
    days_stops: List[Tuple[str, List[Dict[str, Any]]]],
    max_stops: int = 6,
) -> Tuple[bool, str]:
    """
    Penalise any day with too many stops (overly packed schedule).
    Aggregated across all days so it maps to one category score.
    """
    offenders = [
        f"{label}({len(stops)} stops)"
        for label, stops in days_stops
        if len(stops) > max_stops
    ]
    if offenders:
        return False, (
            f"Overly packed days (>{max_stops} stops): {offenders}"
        )
    return True, ""


def _check_budget(
    itin: Dict[str, Any],
    user_budget: Optional[float],
) -> Tuple[bool, str, Optional[str]]:
    """
    Adapted from TravelPlanner's 'Budget' hard constraint.

    [Fix 1] Returns a 3-tuple: (hard_pass, error_msg, warning_msg).
    - If no budget given → trivially passes.
    - If budget given but no cost data found → soft warning (not a hard fail),
      because the LLM may not populate cost_estimate for every stop.
    - If cost data found and total > budget → hard fail.
    """
    if user_budget is None:
        return True, "", None

    total = 0.0
    has_cost_data = False
    for s in _flatten_stops(itin):
        cost = s.get("cost_estimate") or s.get("cost")
        if cost is not None:
            try:
                total += float(cost)
                has_cost_data = True
            except (TypeError, ValueError):
                pass

    if not has_cost_data:
        # Soft warning: budget was set but no cost fields found in itinerary
        return True, "", (
            f"Budget constraint (${user_budget:.0f}) could not be verified — "
            "stops are missing 'cost_estimate' fields."
        )

    if total > user_budget:
        return False, (
            f"Estimated total cost ${total:.0f} exceeds budget ${user_budget:.0f}"
        ), None

    return True, "", None


def _check_lat_lng_present(all_stops: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Every stop should have valid lat/lng within the Singapore bounding box."""
    bad = []
    for s in all_stops:
        lat = s.get("lat")
        lng = s.get("lng")
        name = s.get("name", "?")
        if lat is None or lng is None:
            bad.append(f"{name}(missing)")
        elif not (
            _SG_LAT[0] <= lat <= _SG_LAT[1]
            and _SG_LNG[0] <= lng <= _SG_LNG[1]
        ):
            bad.append(f"{name}(out-of-SG: {lat},{lng})")

    if bad:
        return False, f"Stops with invalid coordinates: {bad}"
    return True, ""


# ─────────────────────────────────────────────
# Main validator
# ─────────────────────────────────────────────
def validate_itinerary(
    itin: Dict[str, Any],
    user_budget: Optional[float] = None,
    max_stops_per_day: int = 6,
) -> ValidationResult:
    """
    Run all constraint checks and return a ValidationResult.

    Micro pass rate is computed over 7 fixed *categories* regardless of
    trip length, so scores are directly comparable across 3/5/7-day itineraries.
    This matches TravelPlanner Eq.1 semantics at the category level.

    Categories (7):
      1. complete_information
      2. diverse_stops
      3. valid_coordinates
      4. within_budget
      5. no_time_conflict   ← any-day aggregation
      6. within_zone        ← any-day aggregation
      7. reasonable_pace    ← any-day aggregation

    Args:
        itin:              The itinerary dict (your JSON schema).
        user_budget:       Optional SGD budget extracted from user input.
        max_stops_per_day: Threshold for 'reasonable pace' check.

    Returns:
        ValidationResult with issues, warnings, per-category scores,
        micro_pass_rate, and penalty.
    """
    if not isinstance(itin, dict) or itin.get("_invalid"):
        return ValidationResult(
            passed=False,
            issues=["Itinerary is invalid or empty"],
            micro_pass_rate=0.0,
            penalty=-5.0,
        )

    days_stops = _stops_by_day(itin)
    all_stops = _flatten_stops(itin)
    issues: List[str] = []
    warnings: List[str] = []

    # ── Category scores (7 fixed keys) ──────────────────────────────────
    cat: Dict[str, bool] = {}

    # 1) Complete information (no empty days)
    ok, msg = _check_no_empty_days(days_stops)
    cat["complete_information"] = ok
    if not ok:
        issues.append(msg)

    # 2) Diverse stops (no repeat across days)
    ok, msg = _check_diverse_stops(all_stops)
    cat["diverse_stops"] = ok
    if not ok:
        issues.append(msg)

    # 3) Lat/lng validity (Singapore bounding box)
    ok, msg = _check_lat_lng_present(all_stops)
    cat["valid_coordinates"] = ok
    if not ok:
        issues.append(msg)

    # 4) Budget check [Fix 1: 3-tuple return]
    ok, err_msg, warn_msg = _check_budget(itin, user_budget)
    cat["within_budget"] = ok
    if not ok:
        issues.append(err_msg)
    if warn_msg:
        warnings.append(warn_msg)

    # 5) No time conflicts — aggregate across all days [Fix 3]
    day_conflict_results = [
        _check_no_time_conflict(stops, label)
        for label, stops in days_stops
        if stops
    ]
    cat["no_time_conflict"] = all(ok for ok, _ in day_conflict_results)
    for ok, msg in day_conflict_results:
        if not ok:
            issues.append(msg)

    # 6) Within zone — aggregate across all days [Fix 2 + Fix 3]
    day_zone_results = [
        _check_within_zone(stops, label)
        for label, stops in days_stops
        if stops
    ]
    cat["within_zone"] = all(ok for ok, _ in day_zone_results)
    for ok, msg in day_zone_results:
        if not ok:
            issues.append(msg)

    # 7) Reasonable pace — aggregated [Fix 3]
    ok, msg = _check_reasonable_pace(days_stops, max_stops_per_day)
    cat["reasonable_pace"] = ok
    if not ok:
        issues.append(msg)

    # ── Micro pass rate over 7 fixed categories ──────────────────────────
    n_total = len(cat)           # always 7
    n_passed = sum(cat.values())
    micro = round(n_passed / n_total, 4)

    # Penalty: each failed category subtracts 0.3
    penalty = round(-0.3 * (n_total - n_passed), 4)

    return ValidationResult(
        passed=(len(issues) == 0),
        issues=issues,
        warnings=warnings,
        constraint_scores=cat,
        micro_pass_rate=micro,
        penalty=penalty,
    )


# ─────────────────────────────────────────────
# Budget extraction helper
# ─────────────────────────────────────────────
def extract_budget_from_text(text: str) -> Optional[float]:
    """
    Best-effort extract a budget number from user input.
    Handles: '$500', 'SGD 1500', '预算500', '预算1500新元', etc.
    """
    import re
    patterns = [
        r"(?:sgd|s\$|\$|新元|新币|元)\s*([0-9,]+(?:\.[0-9]+)?)",
        r"预算\s*([0-9,]+(?:\.[0-9]+)?)",
        r"budget\s+(?:of\s+)?(?:sgd\s+)?([0-9,]+(?:\.[0-9]+)?)",
        r"([0-9,]+(?:\.[0-9]+)?)\s*(?:sgd|s\$|新元|新币)",
    ]
    text_lower = (text or "").lower()
    for pat in patterns:
        m = re.search(pat, text_lower)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                continue
    return None