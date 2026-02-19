# core/memory.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

DB_FILE = os.environ.get("GOGOBOT_DB_FILE", "chat_db.json")


def _load_all() -> Dict[str, Any]:
    if not os.path.exists(DB_FILE):
        return {}
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_all(data: Dict[str, Any]) -> None:
    """持久化全部数据，超过200个线程时自动清理最旧的。"""
    # 自动清理：保留最近 200 个对话（按 history 长度倒序）
    if len(data) > 200:
        print(f"⚠️  Auto-cleanup: {len(data)} threads -> 200")
        threads = sorted(
            data.items(),
            key=lambda x: len(x[1].get("history", [])) if isinstance(x[1], dict) else 0,
            reverse=True,
        )
        data = dict(threads[:200])

    tmp = DB_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, DB_FILE)


def _ensure_thread_container(obj: Any) -> Dict[str, Any]:
    """
    Backward compatibility:
    - old format: thread_id -> [ {"role":..,"content":..}, ...]
    - new format: thread_id -> {"history":[...], "state":{...}}
    """
    if isinstance(obj, list):
        return {"history": obj, "state": {}}

    if isinstance(obj, dict):
        if "history" in obj and "state" in obj:
            history = obj.get("history")
            state = obj.get("state")
            if not isinstance(history, list):
                history = []
            if not isinstance(state, dict):
                state = {}
            return {"history": history, "state": state}
        # dict but not new shape → treat as state-only
        return {"history": [], "state": obj}

    # Unknown / corrupted format
    return {"history": [], "state": {}}


# ---- Chat history ----

def load_chat_history_from_db(thread_id: str) -> List[Dict[str, Any]]:
    data = _load_all()
    container = _ensure_thread_container(data.get(thread_id, []))
    history = container.get("history", [])
    return history if isinstance(history, list) else []


def save_chat_history_to_db(thread_id: str, user_text: str, ai_text: str) -> None:
    data = _load_all()
    container = _ensure_thread_container(data.get(thread_id, []))
    history = container.get("history", []) or []
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": ai_text})
    container["history"] = history
    data[thread_id] = container
    _save_all(data)


# ---- Thread state ----

def load_thread_state(thread_id: str) -> Dict[str, Any]:
    data = _load_all()
    container = _ensure_thread_container(data.get(thread_id, []))
    state_raw = container.get("state", {})
    state = state_raw if isinstance(state_raw, dict) else {}
    # Ensure required keys exist
    state.setdefault("itinerary_versions", [])
    state.setdefault("decision_profile", None)
    state.setdefault("last_rank_meta", None)
    state.setdefault("last_policy", None)
    # Dual-agent additions
    state.setdefault("persona_text", DEFAULT_PERSONA_TEXT())
    state.setdefault("persona_history", [])
    state.setdefault("experiences", [])
    state.setdefault("turn_index", 0)
    state.setdefault("calibration_meta", [])
    container["state"] = state
    data[thread_id] = container
    _save_all(data)
    return state


def save_thread_state(thread_id: str, state: Dict[str, Any]) -> None:
    data = _load_all()
    container = _ensure_thread_container(data.get(thread_id, []))
    container["state"] = state if isinstance(state, dict) else {}
    data[thread_id] = container
    _save_all(data)


def DEFAULT_PERSONA_TEXT() -> str:
    return (
        "- Prioritize practical, geographically coherent plans.\n"
        "- Respect user dislikes and budget; avoid unwanted categories.\n"
        "- Cluster nearby places; minimize cross-zone jumps.\n"
        "- Prefer relaxed pacing unless user asks for packed schedule.\n"
        "- When uncertain, make assumptions and label them.\n"
        "- Offer 1\u20132 alternatives when tradeoffs exist."
    )


# ---- Experience memory (Hi) ----

def append_experience(thread_id: str, exp: Dict[str, Any]) -> None:
    state = load_thread_state(thread_id)
    exps = state.get("experiences", []) or []
    exps.append(exp)
    if len(exps) > 500:
        exps = exps[-500:]
    state["experiences"] = exps
    save_thread_state(thread_id, state)


def get_experiences(thread_id: str) -> List[Dict[str, Any]]:
    state = load_thread_state(thread_id)
    return state.get("experiences", []) or []


def get_short_long_experiences(
    thread_id: str, tw: int, tm: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    exps = get_experiences(thread_id)
    short = exps[-tw:] if tw > 0 else []
    long = exps[-tm:] if tm > 0 else []
    return short, long


def summarize_experiences(exps: List[Dict[str, Any]]) -> str:
    """Deterministic template summary (cheap). Keep short to fit prompt."""
    if not exps:
        return "(none)"
    lines = []
    for i, e in enumerate(exps[-10:], start=1):
        ctx = e.get("context", {})
        out = e.get("outcome", {})
        title = ctx.get("title") or ctx.get("destination") or "trip"
        sat = out.get("user_satisfaction")
        issues = out.get("issues", [])
        cj = out.get("validator", {}).get("cross_zone_jumps")
        chg = out.get("validator", {}).get("total_changes")
        # 新增：显示约束通过率（如果存在）
        mpr = out.get("validator", {}).get("micro_pass_rate")
        mpr_str = f" | micro_pass_rate={mpr:.2f}" if mpr is not None else ""
        lines.append(
            f"{i}. {title} | sat={sat} | issues={issues} "
            f"| cross_zone_jumps={cj} | total_changes={chg}{mpr_str}"
        )
    return "\n".join(lines)