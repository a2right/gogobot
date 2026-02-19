# core/stability.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from collections import Counter

Itinerary = Dict[str, Any]

@dataclass
class SwitchMetrics:
    added: int
    removed: int
    moved: int
    reordered: int
    cross_zone_jumps: int
    edit_distance: int
    total_changes: int

def _levenshtein(a: List[str], b: List[str]) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, start=1):
            cur = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[-1]

def _flatten_stops(itin: Itinerary) -> List[Dict[str, Any]]:
    if not itin:
        return []
    if isinstance(itin, dict) and "days" in itin and isinstance(itin["days"], list):
        out: List[Dict[str, Any]] = []
        for d in itin["days"]:
            if isinstance(d, dict):
                out.extend([s for s in d.get("stops", []) if isinstance(s, dict)])
        return out
    if isinstance(itin, dict) and "stops" in itin and isinstance(itin["stops"], list):
        return [s for s in itin["stops"] if isinstance(s, dict)]
    return []

def _stop_key(stop: Dict[str, Any]) -> str:
    return str(stop.get("id") or stop.get("poi_id") or stop.get("name") or repr(stop))

def _count_cross_zone_jumps(stops: List[Dict[str, Any]]) -> int:
    zones: List[str] = []
    for s in stops:
        z = s.get("zone") or s.get("district") or s.get("area")
        zones.append(str(z) if z is not None else "")
    jumps = 0
    for i in range(1, len(zones)):
        if zones[i] and zones[i - 1] and zones[i] != zones[i - 1]:
            jumps += 1
    return jumps

def itinerary_diff(prev: Optional[Itinerary], cur: Optional[Itinerary]) -> SwitchMetrics:
    prev_stops = _flatten_stops(prev or {})
    cur_stops = _flatten_stops(cur or {})

    prev_ids = [_stop_key(s) for s in prev_stops]
    cur_ids = [_stop_key(s) for s in cur_stops]

    prev_c = Counter(prev_ids)
    cur_c = Counter(cur_ids)

    added = sum((cur_c - prev_c).values())
    removed = sum((prev_c - cur_c).values())

    edit_distance = _levenshtein(prev_ids, cur_ids)
    moved = max(0, edit_distance - added - removed)

    reordered = 1 if (added == 0 and removed == 0 and prev_ids != cur_ids and len(prev_ids) > 1) else 0
    cross_zone_jumps = _count_cross_zone_jumps(cur_stops)

    total_changes = added + removed + moved + (2 * cross_zone_jumps) + (2 if reordered else 0)

    return SwitchMetrics(
        added=added,
        removed=removed,
        moved=moved,
        reordered=reordered,
        cross_zone_jumps=cross_zone_jumps,
        edit_distance=edit_distance,
        total_changes=total_changes,
    )

def switch_penalty(metrics: SwitchMetrics, lam: float = 1.0) -> float:
    return lam * float(metrics.total_changes)
