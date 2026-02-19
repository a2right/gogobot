# core/decision_profile.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict
from dataclasses import dataclass, asdict, field
@dataclass
class DecisionStats:
    win_stay: int = 0
    win_switch: int = 0
    lose_stay: int = 0
    lose_switch: int = 0

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DecisionStats":
        return DecisionStats(
            win_stay=int(d.get("win_stay", 0)),
            win_switch=int(d.get("win_switch", 0)),
            lose_stay=int(d.get("lose_stay", 0)),
            lose_switch=int(d.get("lose_switch", 0)),
        )

@dataclass
class DecisionProfile:
    status_quo: float = 0.25
    naive: float = 0.25
    strategic: float = 0.25
    exploratory: float = 0.25
    stats: DecisionStats = field(default_factory=DecisionStats)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["stats"] = self.stats.to_dict()
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DecisionProfile":
        stats = DecisionStats.from_dict(d.get("stats", {}) if isinstance(d.get("stats", {}), dict) else {})
        return DecisionProfile(
            status_quo=float(d.get("status_quo", 0.25)),
            naive=float(d.get("naive", 0.25)),
            strategic=float(d.get("strategic", 0.25)),
            exploratory=float(d.get("exploratory", 0.25)),
            stats=stats,
        )

def _safe_norm(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, v) for v in w.values())
    if s <= 1e-9:
        k = 1.0 / len(w)
        return {kk: k for kk in w}
    return {kk: max(0.0, vv) / s for kk, vv in w.items()}

def classify_from_conditional(stats: DecisionStats) -> Dict[str, float]:
    ws, wsw = stats.win_stay, stats.win_switch
    ls, lsw = stats.lose_stay, stats.lose_switch

    win = ws + wsw
    lose = ls + lsw
    p_win_stay = ws / win if win else 0.5
    p_lose_stay = ls / lose if lose else 0.5

    score_status_quo = 0.5 * (p_win_stay + p_lose_stay)
    score_exploratory = 0.5 * ((1 - p_win_stay) + (1 - p_lose_stay))
    score_naive = 0.5 * (p_win_stay + (1 - p_lose_stay))
    score_strategic = 0.5 * ((1 - p_win_stay) + p_lose_stay)

    return _safe_norm({
        "status_quo": score_status_quo,
        "naive": score_naive,
        "strategic": score_strategic,
        "exploratory": score_exploratory,
    })

def update_profile(profile: DecisionProfile, is_win: bool, is_switch: bool, alpha: float = 0.35) -> DecisionProfile:
    if is_win and (not is_switch):
        profile.stats.win_stay += 1
    elif is_win and is_switch:
        profile.stats.win_switch += 1
    elif (not is_win) and (not is_switch):
        profile.stats.lose_stay += 1
    else:
        profile.stats.lose_switch += 1

    new_w = classify_from_conditional(profile.stats)
    old_w = {
        "status_quo": profile.status_quo,
        "naive": profile.naive,
        "strategic": profile.strategic,
        "exploratory": profile.exploratory,
    }
    mixed = _safe_norm({k: (1 - alpha) * old_w[k] + alpha * new_w[k] for k in old_w})

    profile.status_quo = mixed["status_quo"]
    profile.naive = mixed["naive"]
    profile.strategic = mixed["strategic"]
    profile.exploratory = mixed["exploratory"]
    return profile
