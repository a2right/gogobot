# core/calibration.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate

from core.prompts import (
    CALIB_PSEUDOGRAD_SYSTEM_PROMPT,
    CALIB_DIRECTIONS_SYSTEM_PROMPT,
    CALIB_EDIT_SYSTEM_PROMPT,
    CALIB_EVAL_SYSTEM_PROMPT,
    CALIB_SMOOTH_SYSTEM_PROMPT,
)


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def _escape_braces(s: str) -> str:
    return (s or "").replace("{", "{{").replace("}", "}}")


def _safe_json_loads(text: str) -> Optional[dict]:
    """Best-effort JSON 解析：提取第一个 {...} 块。"""
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    import re
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _llm_invoke_json(llm, system_prompt: str, user_payload: str) -> Dict[str, Any]:
    system_prompt = _escape_braces(system_prompt)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    msg = prompt.format_messages(input=user_payload)
    resp = llm.invoke(msg)
    content = getattr(resp, "content", None) if not isinstance(resp, str) else resp
    return _safe_json_loads(content or "") or {}


# ─────────────────────────────────────────────
# 结果容器
# ─────────────────────────────────────────────

@dataclass
class CalibrationResult:
    updated: bool
    persona_text: str
    meta: Dict[str, Any]


# ─────────────────────────────────────────────
# CalibrationAgent
# ─────────────────────────────────────────────

class CalibrationAgent:
    """
    双代理校准（Traveler Agent + Calibration Agent），对应论文 Algorithm 2。

    流程：
      1. 评估当前 persona 的 loss
      2. 对每条短窗口 experience 计算伪梯度（文本反馈）
      3. 将多条反馈整合为 J 个改进方向
      4. 沿每个方向编辑 persona，生成候选集
      5. 评估候选 → 选最优
      6. 用长窗口基准 persona 做平滑，稳定输出
    """

    def __init__(self, llm, tw: int = 5, tm: int = 30, J: int = 3):
        self.llm = llm
        self.tw = max(1, int(tw))
        self.tm = max(self.tw + 1, int(tm))
        self.J = max(1, int(J))

    def should_calibrate(self, user_input: str, turn_index: int, every_n_turns: int) -> bool:
        kw = ("不满意", "重来", "完全不同", "换路线", "太赶", "太贵",
              "不要", "不想", "不行", "不合理")
        if any(k in (user_input or "") for k in kw):
            return True
        if every_n_turns <= 0:
            return False
        return (turn_index % every_n_turns) == 0

    def calibrate(
        self,
        persona_text: str,
        experiences_short: List[Dict[str, Any]],
        experiences_long: List[Dict[str, Any]],
    ) -> CalibrationResult:

        # 1) 评估当前 persona loss
        curr_eval = self._evaluate_persona(persona_text, experiences_short)
        curr_loss = float(curr_eval.get("loss", 0.0))

        # 2) 对每条短窗口 experience 计算伪梯度反馈
        feedbacks: List[str] = []
        for exp in experiences_short:
            fb = self._pseudo_gradient(persona_text, exp)
            if fb:
                feedbacks.append(fb)

        if not feedbacks:
            return CalibrationResult(updated=False, persona_text=persona_text, meta={
                "reason": "no_feedback",
                "curr_eval": curr_eval,
            })

        # 3) 整合反馈为 J 个改进方向
        directions = self._synthesize_directions(feedbacks, self.J)
        if not directions:
            return CalibrationResult(updated=False, persona_text=persona_text, meta={
                "reason": "no_directions",
                "curr_eval": curr_eval,
            })

        # 4) 沿每个方向编辑 persona，生成候选集
        candidates = []
        for d in directions:
            cand = self._edit_persona(persona_text, d)
            if cand and cand.strip():
                candidates.append({"direction": d, "persona": cand.strip()})

        if not candidates:
            return CalibrationResult(updated=False, persona_text=persona_text, meta={
                "reason": "no_candidates",
                "curr_eval": curr_eval,
            })

        # 5) 评估候选，选 loss 最低者
        scored = []
        for c in candidates:
            ev = self._evaluate_persona(c["persona"], experiences_short)
            loss = float(ev.get("loss", 0.0))
            scored.append({**c, "eval": ev, "loss": loss})

        scored.sort(key=lambda x: x["loss"])
        best = scored[0]

        # 只有严格更优才更新
        updated = best["loss"] + 1e-9 < curr_loss

        new_persona = persona_text
        smooth_meta: Dict[str, Any] = {}

        if updated:
            # 6) 用长窗口 baseline 做平滑，避免过拟合单次抱怨
            baseline = self._smooth_baseline_persona(persona_text, experiences_long)
            smoothed = self._smooth_persona(best["persona"], baseline)
            if smoothed and smoothed.strip():
                new_persona = smoothed.strip()
                smooth_meta = {"baseline": baseline, "smoothed": True}
            else:
                new_persona = best["persona"]

        meta = {
            "updated": updated,
            "curr_eval": curr_eval,
            "best": {k: best[k] for k in ("direction", "loss", "eval")},
            "all_candidates": [{k: x[k] for k in ("direction", "loss")} for x in scored],
            **smooth_meta,
        }

        return CalibrationResult(updated=updated, persona_text=new_persona, meta=meta)

    # ─────────────────────────────────────────
    # LLM 子程序
    # ─────────────────────────────────────────

    def _evaluate_persona(
        self, persona_text: str, experiences: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """用 CALIB_EVAL_SYSTEM_PROMPT 评估 persona 的 loss（0-10，越低越好）。"""
        payload = json.dumps(
            {"persona": persona_text, "experiences": experiences},
            ensure_ascii=False,
        )
        return _llm_invoke_json(self.llm, CALIB_EVAL_SYSTEM_PROMPT, payload)

    def _pseudo_gradient(
        self, persona_text: str, experience: Dict[str, Any]
    ) -> str:
        """对单条 experience 计算文本伪梯度反馈。"""
        payload = json.dumps(
            {"persona": persona_text, "experience": experience},
            ensure_ascii=False,
        )
        out = _llm_invoke_json(self.llm, CALIB_PSEUDOGRAD_SYSTEM_PROMPT, payload)
        return (out.get("feedback") or "").strip()

    def _synthesize_directions(
        self, feedbacks: List[str], J: int
    ) -> List[str]:
        """将多条反馈整合为 J 个互不相同的改进方向。"""
        payload = json.dumps({"feedbacks": feedbacks, "J": J}, ensure_ascii=False)
        out = _llm_invoke_json(self.llm, CALIB_DIRECTIONS_SYSTEM_PROMPT, payload)
        dirs = out.get("directions")
        if isinstance(dirs, list):
            return [str(x).strip() for x in dirs if str(x).strip()]
        # fallback：单条方向
        if isinstance(out.get("direction"), str) and out["direction"].strip():
            return [out["direction"].strip()]
        return []

    def _edit_persona(self, persona_text: str, direction: str) -> str:
        """沿指定方向编辑 persona 文本。"""
        payload = json.dumps(
            {"persona": persona_text, "direction": direction},
            ensure_ascii=False,
        )
        out = _llm_invoke_json(self.llm, CALIB_EDIT_SYSTEM_PROMPT, payload)
        return (out.get("persona") or "").strip()

    def _smooth_baseline_persona(
        self, persona_text: str, experiences_long: List[Dict[str, Any]]
    ) -> str:
        """从长窗口 experience 中提炼稳定的基准 persona。"""
        payload = json.dumps(
            {"persona": persona_text, "experiences": experiences_long},
            ensure_ascii=False,
        )
        out = _llm_invoke_json(self.llm, CALIB_SMOOTH_SYSTEM_PROMPT, payload)
        return (out.get("baseline_persona") or persona_text).strip()

    def _smooth_persona(self, best_persona: str, baseline_persona: str) -> str:
        """
        将候选最优 persona 与长窗口基准 persona 合并，防止过拟合。

        修复说明：
          原版两次均调用 CALIB_EVAL_SYSTEM_PROMPT（评估用），
          导致平滑步骤实际上在做评估而非合并，输出的 key 也对不上。
          现改为专用的 CALIB_SMOOTH_SYSTEM_PROMPT，并明确要求返回
          合并后的 persona 文本（key: "persona"）。
        """
        payload = json.dumps(
            {
                "best_persona": best_persona,
                "baseline_persona": baseline_persona,
                "instruction": (
                    "Merge best_persona with baseline_persona into a single, "
                    "stable final persona (6-10 bullet rules). "
                    "Preserve strong constraints from best_persona; keep stable "
                    "preferences from baseline_persona. "
                    "Output ONLY JSON with key 'persona'."
                ),
            },
            ensure_ascii=False,
        )
        out = _llm_invoke_json(self.llm, CALIB_SMOOTH_SYSTEM_PROMPT, payload)  # ← 修复点

        # 优先取明确的 persona 字段
        merged = (out.get("persona") or "").strip()
        if merged:
            return merged

        # 二次回退：baseline_persona 本身已经是稳定版本，直接用
        return baseline_persona.strip()