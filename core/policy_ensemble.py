# core/policy_ensemble.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PolicyConfig:
    name: str
    persona_instruction: str

POLICIES: List[PolicyConfig] = [
    PolicyConfig(
        name="status_quo",
        persona_instruction=(
            "你是“稳定优先”的旅行规划师：\n"
            "- 满足硬约束前提下，尽量复用上一版行程。\n"
            "- 只做最小必要改动（少换点、少换顺序、少跨区）。\n"
        ),
    ),
    PolicyConfig(
        name="naive",
        persona_instruction=(
            "你是“效率优先”的旅行规划师：\n"
            "- 优先最短路程/最少转场/最少排队。\n"
            "- 贪心但必须遵守营业时间与地理可达性。\n"
        ),
    ),
    PolicyConfig(
        name="strategic",
        persona_instruction=(
            "你是“稳健策略型”的旅行规划师：\n"
            "- 主动规避拥挤/峰值/高不确定性点。\n"
            "- 倾向同区域聚类、错峰、预留缓冲。\n"
        ),
    ),
    PolicyConfig(
        name="exploratory",
        persona_instruction=(
            "你是“探索型”的旅行规划师：\n"
            "- 在满足硬约束前提下，加入新奇/小众体验。\n"
            "- 允许更高多样性，但不得产生明显跨区跳跃与时间冲突。\n"
        ),
    ),
]

def stability_lambda_from_profile(profile: Dict[str, float]) -> float:
    sq = float(profile.get("status_quo", 0.25))
    ex = float(profile.get("exploratory", 0.25))
    lam = 1.0 + 1.2 * sq - 0.8 * ex
    return max(0.5, min(2.0, lam))
