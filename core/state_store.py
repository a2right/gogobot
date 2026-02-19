# core/state_store.py
from __future__ import annotations
import json
import os
from typing import Any, Dict

STATE_PATH = os.environ.get("GOGOBOT_STATE_PATH", "agent_state.json")

def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_PATH)

def get_thread_state(thread_id: str) -> Dict[str, Any]:
    state = load_state()
    return state.get(thread_id, {}) if isinstance(state.get(thread_id, {}), dict) else {}

def set_thread_state(thread_id: str, thread_state: Dict[str, Any]) -> None:
    state = load_state()
    state[thread_id] = thread_state
    save_state(state)