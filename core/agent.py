# core/agent.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

from core.config import llm, CALIB_TW, CALIB_TM, CALIB_J, CALIB_EVERY_N_TURNS
from core.examples import select_examples
from tools import ALL_TOOLS

from core.memory import (
    load_chat_history_from_db,
    save_chat_history_to_db,
    load_thread_state,
    save_thread_state,
    append_experience,
    get_short_long_experiences,
    summarize_experiences,
)

from core.prompts import SYSTEM_PROMPT_TEXT, PLAN_JSON_SYSTEM_PROMPT, RENDER_SYSTEM_PROMPT
from core.policy_ensemble import POLICIES, stability_lambda_from_profile
from core.stability import itinerary_diff, switch_penalty
from core.decision_profile import DecisionProfile, update_profile

from core.calibration import CalibrationAgent

# ================================================================
# 修改1：新增约束验证器导入
# ================================================================
from core.constraint_validator import validate_itinerary, extract_budget_from_text


# -----------------------
# Prompt utilities
# -----------------------
def _escape_braces(s: str) -> str:
    """Escape literal braces so LangChain/Python format() won't treat JSON as placeholders."""
    return (s or "").replace("{", "{{").replace("}", "}}")

def _format_recent_chat(history: List[Dict[str, Any]], max_messages: int = 8, max_chars: int = 2000) -> str:
    """Format last N turns into a compact context block for the planner."""
    if not isinstance(history, list) or not history:
        return ""
    tail = history[-max_messages:]
    lines: List[str] = []
    for m in tail:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"USER: {content}")
        else:
            lines.append(f"ASSISTANT: {content}")
    out = "\n".join(lines).strip()
    if len(out) > max_chars:
        out = out[-max_chars:]
    return out


def _last_assistant_message(history: List[Dict[str, Any]]) -> str:
    """Return the last assistant message content from history."""
    if not isinstance(history, list):
        return ""
    for m in reversed(history):
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").lower()
        if role in ("assistant", "ai"):
            return m.get("content") or ""
    return ""


def _is_transform_request(text: str) -> bool:
    """Detect translate/rewrite/summarize requests (CN + EN)."""
    raw = text or ""
    t = raw.lower()

    kws_cn = (
        "翻译", "翻成", "用中文", "用英文",
        "改写", "润色", "校对", "总结", "概括", "提炼", "摘要",
        "把上面的内容", "将上面的内容", "把上述内容",
        "上一条你说", "你刚才说", "你前面说", "上一条内容",
    )

    kws_en = (
        "translate", "translation", "to chinese", "into chinese", "in chinese",
        "to english", "into english", "in english",
        "rewrite", "rephrase", "paraphrase", "polish", "proofread", "edit",
        "summarize", "summary", "tldr", "tl;dr",
        "convert", "convert to", "render in",
        "above text", "the above", "previous message", "last message", "what did you say",
    )

    return any(k in raw for k in kws_cn) or any(k in t for k in kws_en)



def _needs_prev_context(text: str) -> bool:
    """Detect follow-up planning requests that refer to previous outputs (CN + EN)."""
    raw = text or ""
    t = raw.lower()
    kws_cn = ("上一条", "上面", "刚才", "前面", "上一轮", "根据上面", "基于上面", "延续上面", "继续", "再优化", "调整一下")
    kws_en = ("previous", "above", "earlier", "last", "based on", "continue", "improve", "refine", "tweak")
    return any(k in raw for k in kws_cn) or any(k in t for k in kws_en)


# Detect plan modification/optimization requests (for continuity)
def _is_modify_plan_request(text: str) -> bool:
    """Detect requests to modify/optimize an existing itinerary (CN + EN)."""
    raw = text or ""
    t = raw.lower()

    kws_cn = (
        "修改", "改一下", "调整", "优化", "完善", "微调", "改进",
        "删掉", "增加", "替换", "改成", "换成", "重新安排",
        "更省钱", "更便宜", "更悠闲", "更紧凑", "别太赶", "少走路",
        "更适合", "更符合", "根据上面", "基于上面",
    )
    kws_en = (
        "modify", "change", "adjust", "update", "revise", "refine", "tweak", "optimize", "improve",
        "remove", "add", "replace", "swap", "rearrange", "reschedule",
        "make it cheaper", "make it more relaxed", "less walking", "more indoor", "not too rushed",
        "based on the above", "based on the previous", "using the previous plan",
    )

    return any(k in raw for k in kws_cn) or any(k in t for k in kws_en)


def _run_transform(local_llm, instruction: str, source_text: str, persona_text: str = "") -> str:
    """Transform SOURCE_TEXT according to INSTRUCTION (translate/rewrite/summarize)."""
    sys = (
        "You are a text transformation assistant.\n"
        "You will ONLY transform the provided SOURCE_TEXT following the INSTRUCTION.\n"
        "Rules:\n"
        "- Do not invent new content.\n"
        "- Preserve meaning. If something is ambiguous in SOURCE_TEXT, keep it ambiguous.\n"
        "- Output ONLY the transformed text (no JSON, no extra explanation).\n"
    )
    if persona_text:
        sys += "\nTone persona:\n" + persona_text + "\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", _escape_braces(sys)),
        ("human", "{input}")
    ])

    payload = f"INSTRUCTION:\n{instruction}\n\nSOURCE_TEXT:\n{source_text}"
    resp = local_llm.invoke(prompt.format_messages(input=payload))
    return getattr(resp, "content", resp) if resp is not None else ""


def _build_fewshot(user_input: str, k: int = 4, prefer_semantic: bool = True) -> List[Tuple[str, str]]:
    selected = select_examples(user_input, k=k, prefer_semantic=prefer_semantic)
    msgs: List[Tuple[str, str]] = []
    for ex in selected:
        msgs.append(("human", ex["question"]))
        msgs.append(("ai", ex["answer"]))
    return msgs

def _compose_persona(base_persona: str, policy_instruction: str) -> str:
    base_persona = (base_persona or "").strip()
    policy_instruction = (policy_instruction or "").strip()
    if not base_persona:
        return policy_instruction
    if not policy_instruction:
        return base_persona
    return base_persona + "\n" + policy_instruction

def _planner_prompt(persona_text: str, short_mem: str, long_mem: str, fewshot: List[Tuple[str, str]]) -> ChatPromptTemplate:
    sys = (
        SYSTEM_PROMPT_TEXT
        + "\n\n[PERSONA]\n" + persona_text
        + "\n\n[SHORT_MEMORY]\n" + short_mem
        + "\n\n[LONG_MEMORY]\n" + long_mem
        + "\n\nYou must follow the JSON output system prompt exactly."
        + "\nIf the user input includes a [CURRENT_ITINERARY] block, you MUST treat it as the existing plan to edit."
        + "\nMake the smallest necessary changes to satisfy the new request; keep unchanged days/stops as-is unless required."
        + "\nPreserve entity names/structure; do not rewrite everything from scratch."
    )
    messages: List[Tuple[str, str]] = [
        ("system", _escape_braces(PLAN_JSON_SYSTEM_PROMPT)),
        ("system", _escape_braces(sys)),
    ]
    messages += fewshot
    messages += [("human", "{input}")]
    return ChatPromptTemplate.from_messages(messages)

def _render_prompt(persona_text: str) -> ChatPromptTemplate:
    messages = [
        ("system", _escape_braces(RENDER_SYSTEM_PROMPT)),
        ("system", _escape_braces("Persona (for tone only):\n" + persona_text)),
        ("human", "{input}")
    ]
    return ChatPromptTemplate.from_messages(messages)


# -----------------------
# Tool argument planner (placeholder friendly)
# -----------------------
def _available_tool_names() -> List[str]:
    names = []
    for t in ALL_TOOLS:
        n = getattr(t, "name", None) or getattr(t, "__name__", None) or t.__class__.__name__
        names.append(str(n))
    return names

def _plan_tool_args(user_input: str) -> dict:
    tool_names = _available_tool_names()
    planner_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a strict tool-argument planner. Output ONLY valid JSON. "
            "Keys must be tool names, values are argument objects. "
            f"Available tools: {', '.join(tool_names)}"
        )),
        ("human", "{input}")
    ])
    try:
        resp = llm.invoke(planner_prompt.format_messages(input=user_input))
        content = getattr(resp, "content", "")
        return json.loads(content)
    except Exception:
        return {}

def _execute_tools(tool_args: dict) -> Dict[str, Any]:
    """
    Execute tools if available. If a tool is a placeholder, it should still be safe.
    Returns: {tool_name: tool_result}
    """
    results: Dict[str, Any] = {}
    if not isinstance(tool_args, dict):
        return results
    name_to_tool = {}
    for t in ALL_TOOLS:
        n = getattr(t, "name", None) or getattr(t, "__name__", None) or t.__class__.__name__
        name_to_tool[str(n)] = t

    for name, args in tool_args.items():
        tool = name_to_tool.get(name)
        if tool is None:
            continue
        try:
            if hasattr(tool, "invoke"):
                results[name] = tool.invoke(args if isinstance(args, dict) else {})
            elif callable(tool):
                results[name] = tool(args)
            else:
                results[name] = {"error": "tool_not_callable"}
        except Exception as e:
            results[name] = {"error": str(e)}
    return results


# -----------------------
# Itinerary generation + ranking (first-paper method)
# -----------------------
def _safe_parse_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # attempt extract first JSON object
    import re
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def _generate_candidate_itinerary(
    user_input: str,
    persona_text: str,
    policy_name: str,
    policy_instruction: str,
    short_mem: str,
    long_mem: str,
    fewshot_query: Optional[str] = None,
) -> Dict[str, Any]:
    # IMPORTANT: few-shot should be retrieved from the ORIGINAL user query, not from injected planning_input.
    # When editing an existing itinerary (CURRENT_ITINERARY provided), disable few-shot to avoid copying templates.
    if fewshot_query is None:
        fewshot_query = user_input
    k = 0 if "[CURRENT_ITINERARY]" in (user_input or "") else 4
    fewshot = _build_fewshot(fewshot_query, k=k)
    composed_persona = _compose_persona(persona_text, f"[POLICY={policy_name}]\n" + policy_instruction)
    prompt = _planner_prompt(composed_persona, short_mem, long_mem, fewshot)
    resp = llm.invoke(prompt.format_messages(input=user_input))
    raw = getattr(resp, "content", "") if not isinstance(resp, str) else resp
    data = _safe_parse_json(raw) or {"_invalid": True, "_raw": raw}
    return data

# ================================================================
# 修改2：替换 _itinerary_quality()，整合约束验证器的 penalty
# ================================================================
def _itinerary_quality(itin: Dict[str, Any],
                       user_budget: Optional[float] = None) -> float:
    """
    Quality proxy = structural score + constraint penalty.
    - Structural checks: missing days/stops/fields  (-ve score)
    - Constraint checks: time conflicts, zone jumps, duplicates, budget (-0.3 each)
    """
    if not isinstance(itin, dict) or itin.get("_invalid"):
        return -5.0
    days = itin.get("days", [])
    if not isinstance(days, list) or not days:
        return -3.0
    # Structural score (original logic)
    score = 0.0
    for d in days:
        stops = (d or {}).get("stops", [])
        if not isinstance(stops, list) or len(stops) < 1:
            score -= 1.0
            continue
        for s in stops:
            if not isinstance(s, dict):
                score -= 0.2
                continue
            for k in ("name", "zone", "start", "end"):
                if not s.get(k):
                    score -= 0.1
            for k in ("lat", "lng"):
                if s.get(k) is None:
                    score -= 0.1
    # Constraint penalty (TravelPlanner-style)
    vr = validate_itinerary(itin, user_budget=user_budget)
    score += vr.penalty   # vr.penalty is already negative (e.g. -0.3 per failed constraint)
    return score

def _rank_candidates(
    candidates: List[Tuple[str, Dict[str, Any]]],
    prev_itinerary: Optional[Dict[str, Any]],
    profile: DecisionProfile,
    user_budget: Optional[float] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    # priors from profile
    priors = {
        "status_quo": profile.status_quo,
        "naive": profile.naive,
        "strategic": profile.strategic,
        "exploratory": profile.exploratory,
    }
    lam = stability_lambda_from_profile(priors)

    best = None
    best_meta = None

    # Pass user_budget so quality scoring is budget-aware
    prev_quality = _itinerary_quality(prev_itinerary, user_budget=user_budget) if prev_itinerary else None

    for policy, itin in candidates:
        q = _itinerary_quality(itin, user_budget=user_budget)
        metrics = itinerary_diff(prev_itinerary, itin) if prev_itinerary else None
        pen = switch_penalty(metrics, lam) if metrics else 0.0

        prior = float(priors.get(policy, 0.25))
        # final_score: prior + quality - w*penalty
        w = 0.05
        final = prior + q - w * float(pen)

        meta = {
            "policy": policy,
            "prior": prior,
            "lambda": lam,
            "quality": q,
            "switch_metrics": metrics.__dict__ if metrics else None,
            "switch_penalty": pen,
            "final_score": final,
            "prev_quality": prev_quality,
        }

        if best is None or final > best_meta["final_score"]:
            best = (policy, itin)
            best_meta = meta

    assert best is not None and best_meta is not None
    return best[0], best[1], best_meta


# -----------------------
# Main entry
# -----------------------
def run_with_all_tools(user_input: str, thread_id: str) -> AIMessage:
    # ========== 新增: 调试日志 ==========
    print(f"\n{'='*60}")
    print(f"🔍 Processing request for thread_id: {thread_id}")
    print(f"📝 User input: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")
    
    # 0) load history + state
    history = load_chat_history_from_db(thread_id)
    state = load_thread_state(thread_id)

    # ========== 新增: 显示历史状态 ==========
    print(f"📚 Loaded {len(history)} messages from history")
    prev_versions = state.get("itinerary_versions", [])
    print(f"🗺️  Found {len(prev_versions)} previous itinerary versions")
    if prev_versions:
        last_itin = prev_versions[-1]
        days_count = len(last_itin.get("days", [])) if isinstance(last_itin, dict) else 0
        print(f"    Last itinerary had {days_count} days")
    print(f"{'='*60}\n")

    original_user_input = user_input
    prev_assistant = _last_assistant_message(history)
    recent_chat = _format_recent_chat(history, max_messages=8, max_chars=2000)

    # thread turn index
    turn_index = int(state.get("turn_index", 0)) + 1
    state["turn_index"] = turn_index

    persona_text = state.get("persona_text") or ""

    # previous itinerary (for edit continuity)
    prev_itin_for_edit = None
    try:
        versions0 = state.get("itinerary_versions") or []
        if isinstance(versions0, list) and versions0:
            prev_itin_for_edit = versions0[-1]
    except Exception:
        prev_itin_for_edit = None

    # --- Route translation/rewrite/summarize requests to transform pipeline ---
    if _is_transform_request(original_user_input):
        if prev_assistant.strip():
            response_text = _run_transform(llm, original_user_input, prev_assistant, persona_text)
        else:
            response_text = (
                "我没有找到上一条需要被翻译/改写/总结的内容。"
                "请将需要转换的文本直接粘贴出来，或先让我生成一段内容后再让我转换。"
            )

        # persist state (turn_index already incremented below) and save history
        save_thread_state(thread_id, state)
        save_chat_history_to_db(thread_id, original_user_input, response_text)
        return AIMessage(content=response_text)

    # --- Inject recent chat / previous assistant output into planning input ---
    planning_input = original_user_input
    if recent_chat.strip():
        planning_input = planning_input + "\n\n[RECENT_CHAT]\n" + recent_chat
    if _needs_prev_context(original_user_input) and prev_assistant.strip():
        planning_input = planning_input + "\n\n[PREVIOUS_ASSISTANT_OUTPUT]\n" + prev_assistant

    # If user is modifying/optimizing the plan, provide the current itinerary JSON explicitly
    if prev_itin_for_edit is not None and (_is_modify_plan_request(original_user_input) or _needs_prev_context(original_user_input)):
        try:
            cur_json = json.dumps(prev_itin_for_edit, ensure_ascii=False)
        except Exception:
            cur_json = str(prev_itin_for_edit)
        planning_input = (
            planning_input
            + "\n\n[CURRENT_ITINERARY]\n"
            + cur_json
            + "\n\n[EDIT_INSTRUCTIONS]\n"
            + "Edit the CURRENT_ITINERARY to satisfy the user request. Keep unchanged parts. Output full updated itinerary JSON."
        )

    # experience summaries
    short_exps, long_exps = get_short_long_experiences(thread_id, CALIB_TW, CALIB_TM)
    short_mem = summarize_experiences(short_exps)
    long_mem = summarize_experiences(long_exps)

    # 1) tool planning + execution (safe placeholders)
    tool_args = _plan_tool_args(planning_input)
    tool_results = _execute_tools(tool_args)

    # ================================================================
    # 修改3（前置）：在候选生成前提取预算，供 quality 评分使用
    # ================================================================
    user_budget = extract_budget_from_text(original_user_input)

    # 2) generate policy candidates (supports dict or list POLICIES)
    candidates: List[Tuple[str, Dict[str, Any]]] = []

    if isinstance(POLICIES, dict):
        policy_iter = list(POLICIES.items())
    else:
        # Expect list items to have `name` and `persona_instruction` attributes; fall back defensively.
        policy_iter = []
        for p in (POLICIES or []):
            name = getattr(p, "name", None)
            instr = getattr(p, "persona_instruction", None)
            if name is None:
                # last-resort: represent as string
                name = str(getattr(p, "policy", None) or getattr(p, "id", None) or p)
            if instr is None:
                instr = str(getattr(p, "instruction", "") or getattr(p, "prompt", "") or "")
            policy_iter.append((str(name), str(instr)))

    for policy_name, policy_instruction in policy_iter:
        itin = _generate_candidate_itinerary(
            user_input=planning_input,
            persona_text=persona_text,
            policy_name=policy_name,
            policy_instruction=policy_instruction,
            short_mem=short_mem,
            long_mem=long_mem,
            fewshot_query=original_user_input,
        )
        candidates.append((policy_name, itin))

    # 3) rank/select with switching penalty + profile priors
    profile_dict = state.get("decision_profile")
    profile = DecisionProfile.from_dict(profile_dict) if profile_dict else DecisionProfile()
    prev_itin = state.get("itinerary_versions", [])[-1] if state.get("itinerary_versions") else None

    # Pass user_budget so quality scoring is budget-aware during ranking
    best_policy, best_itin, rank_meta = _rank_candidates(candidates, prev_itin, profile, user_budget=user_budget)
    state["last_policy"] = best_policy
    state["last_rank_meta"] = rank_meta

    # 4) update itinerary_versions (keep last 2)
    versions = state.get("itinerary_versions", []) or []
    versions.append(best_itin)
    if len(versions) > 2:
        versions = versions[-2:]
    state["itinerary_versions"] = versions

    # 5) decide is_win / is_switch and update decision_profile (first-paper method)
    prev_quality = rank_meta.get("prev_quality")
    curr_quality = rank_meta.get("quality")
    is_win = (prev_quality is None) or (curr_quality >= (prev_quality or 0.0))
    metrics = rank_meta.get("switch_metrics") or {}
    total_changes = metrics.get("total_changes", 0) if isinstance(metrics, dict) else 0
    is_switch = bool(total_changes >= 6)

    profile = update_profile(profile, is_win=is_win, is_switch=is_switch, alpha=0.25)
    state["decision_profile"] = profile.to_dict()

    # ================================================================
    # 修改3：对 best_itin 做完整约束验证，并将结果注入 experience
    # ================================================================
    validation = validate_itinerary(best_itin, user_budget=user_budget)

    # 6) create and store experience for calibration (Traveler memory Hi)
    exp = {
        "context": {
            "user_input": original_user_input,
            "thread_id": thread_id,
            "user_budget": user_budget,
        },
        "action": best_itin,
        "outcome": {
            "tool_results": tool_results,
            "validator": {
                "cross_zone_jumps": (metrics.get("cross_zone_jumps") if isinstance(metrics, dict) else None),
                "total_changes": total_changes,
                "quality": curr_quality,
                # 新增：来自 constraint_validator 的详细结果
                "constraint_passed": validation.passed,
                "micro_pass_rate": validation.micro_pass_rate,
                "constraint_scores": validation.constraint_scores,
            },
            "user_satisfaction": None,
            # 原来始终是 []，现在填入真实问题列表，校准模块可以读到
            "issues": validation.issues,
        }
    }
    append_experience(thread_id, exp)

    # 7) calibration step (online persona optimization)
    calib = CalibrationAgent(llm, tw=CALIB_TW, tm=CALIB_TM, J=CALIB_J)
    if calib.should_calibrate(original_user_input, turn_index, CALIB_EVERY_N_TURNS):
        short_exps2, long_exps2 = get_short_long_experiences(thread_id, CALIB_TW, CALIB_TM)
        result = calib.calibrate(persona_text, short_exps2, long_exps2)
        if result.updated and result.persona_text.strip() != persona_text.strip():
            # record persona history
            ph = state.get("persona_history", []) or []
            ph.append({"turn": turn_index, "persona": persona_text})
            state["persona_history"] = ph[-50:]
            state["persona_text"] = result.persona_text
        # store calibration meta
        cm = state.get("calibration_meta", []) or []
        cm.append({"turn": turn_index, **result.meta})
        state["calibration_meta"] = cm[-200:]

    # persist state
    save_thread_state(thread_id, state)

    # 8) render to user-facing text
    renderer = _render_prompt(state.get("persona_text", persona_text))
    render_in = json.dumps(best_itin, ensure_ascii=False)
    resp = llm.invoke(renderer.format_messages(input=render_in))
    final_answer = getattr(resp, "content", "") if not isinstance(resp, str) else resp

    # save history
    save_chat_history_to_db(thread_id, original_user_input, final_answer)

    return AIMessage(content=final_answer)