"""
TravelPlanner Sole-Planning 评估脚本 v2
用法:
  python3 tp_run.py --mode parse   # 只测试解析逻辑
  python3 tp_run.py --mode test    # 跑前1条测试
  python3 tp_run.py --mode run     # 跑全部180条
"""

import os, sys, re, json, argparse, time
from pathlib import Path

# ── 路径 ──────────────────────────────────────────────
WORKSPACE   = Path("/usr1/home/s124mdg55_07/a1workspace")
GOGOBOT_DIR = WORKSPACE / "llm"
OUTPUT_DIR  = WORKSPACE / "tp_eval_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "gogobot_plans.jsonl"

sys.path.insert(0, str(GOGOBOT_DIR))
os.chdir(str(GOGOBOT_DIR))

# ── TravelPlanner标准Sole-Planning Prompt ─────────────
# 与论文Direct策略完全一致（来自eval脚本的prompt格式）
TP_SYSTEM_PROMPT = """You are a proficient travel planner. Based on the provided information and query, give a detailed plan with flight numbers, restaurant names, and accommodation names. All information must come from the provided data. Use '-' when information is not needed (e.g. no accommodation on return day).

Output format (strictly follow this):
Day 1:
Current City: from A to B
Transportation: Flight Number: FXXXXXXX, from A to B, Departure Time: HH:MM, Arrival Time: HH:MM
Breakfast: Restaurant Name, City
Attraction: Attraction Name, City
Lunch: Restaurant Name, City
Dinner: Restaurant Name, City
Accommodation: Accommodation Name, City

Day 2:
Current City: B
Transportation: -
...and so on"""


def format_reference_info(ref_info_str: str) -> str:
    """将reference_information压缩为关键信息，控制在1500字以内"""
    try:
        ref_list = eval(ref_info_str) if isinstance(ref_info_str, str) else ref_info_str
        lines = []
        for item in ref_list:
            desc = item.get("Description", "")
            content = item.get("Content", "")
            # 按描述类型做不同程度的压缩
            if "Flight" in desc:
                # 航班：只保留航班号、时间、价格，去掉距离等
                compressed = _compress_flights(content)
            elif "Accommodation" in desc:
                # 住宿：只保留名称、价格、房型、规则、最少住宿天数
                compressed = _compress_table(content, keep_cols=["NAME","price","room type","house_rules","minimum nights","maximum occupancy"])
            elif "Restaurant" in desc:
                # 餐厅：只保留名称、价格、菜系
                compressed = _compress_table(content, keep_cols=["Name","Average Cost","Cuisines"])
            elif "Attraction" in desc:
                # 景点：只保留名称
                compressed = _compress_table(content, keep_cols=["Name"])
            elif "driving" in desc.lower() or "taxi" in desc.lower() or "Self" in desc:
                # 驾车/出租：保留全部（本来就很短）
                compressed = content.strip()
            else:
                compressed = content.strip()[:300]
            lines.append(f"[{desc}]\n{compressed}")
        result = "\n\n".join(lines)
        # 最终硬截断保险（10000 chars ≈ 2500 tokens，给prompt留足空间）
        return result[:6000]
    except Exception:
        return str(ref_info_str)[:2000]


def _compress_flights(content: str) -> str:
    """航班数据不压缩，保留原始格式供评估脚本解析"""
    return content.strip()


def _compress_table(content: str, keep_cols: list) -> str:
    """从pandas风格的表格文本中只保留指定列"""
    lines = [l for l in content.strip().split("\n") if l.strip()]
    if not lines:
        return ""
    # 直接返回每行压缩版（去掉多余空白）
    result = []
    for line in lines[:50]:  # 最多50行
        # 清理多余空格
        cleaned = re.sub(r' {2,}', '  ', line.strip())
        result.append(cleaned[:200])
    return "\n".join(result)


def build_prompt(sample: dict) -> str:
    """构造TravelPlanner标准格式的prompt，加强约束减少幻觉"""
    query = sample.get("query", "")
    ref_info = format_reference_info(sample.get("reference_information", ""))
    instruction = """You are a travel planner. Create a detailed travel plan based ONLY on the information provided below.
STRICT RULES:
1. Only use flight numbers that appear in the given information. If no flight is listed, use self-driving or taxi instead.
2. Only use restaurant names from the given information.
3. Only use accommodation names from the given information.
4. Never invent or guess any information. If data is missing, use "-".
5. Follow the exact output format shown."""
    return f"{instruction}\n\nGiven information:\n{ref_info}\n\nQuery: {query}\n\nTravel Plan:"


# ── Plan文本解析器 ────────────────────────────────────
FIELD_PATTERNS = {
    "current_city":   r"Current City\s*[:\uff1a]\s*(.+)",
    "transportation": r"Transportation\s*[:\uff1a]\s*(.+)",
    "breakfast":      r"Breakfast\s*[:\uff1a]\s*(.+)",
    "attraction":     r"Attraction(?:s)?\s*[:\uff1a]\s*(.+)",
    "lunch":          r"Lunch\s*[:\uff1a]\s*(.+)",
    "dinner":         r"Dinner\s*[:\uff1a]\s*(.+)",
    "accommodation":  r"Accommodation\s*[:\uff1a]\s*(.+)",
}

def parse_plan_text(text: str, n_days: int) -> list:
    if not text or not text.strip():
        return [_empty_day(d) for d in range(1, n_days + 1)]

    blocks = re.split(r'\bDay\s+(\d+)\b\s*:?', text, flags=re.IGNORECASE)
    parsed = {}
    i = 1
    while i < len(blocks) - 1:
        try:
            day_num = int(blocks[i])
            content = blocks[i + 1]
            parsed[day_num] = _parse_day_block(day_num, content)
        except (ValueError, IndexError):
            pass
        i += 2

    result = []
    for d in range(1, n_days + 1):
        result.append(parsed.get(d, _empty_day(d)))
    return result


def _parse_day_block(day_num: int, content: str) -> dict:
    rec = _empty_day(day_num)
    for field, pat in FIELD_PATTERNS.items():
        m = re.search(pat, content, re.IGNORECASE)
        if m:
            val = m.group(1).strip().rstrip(".,")
            if val:
                rec[field] = val
    return rec


def _empty_day(d: int) -> dict:
    return {
        "days": d,
        "current_city": "-",
        "transportation": "-",
        "breakfast": "-",
        "attraction": "-",
        "lunch": "-",
        "dinner": "-",
        "accommodation": "-",
    }


# ── LLM直接调用（绕过GoGoBot的新加坡system prompt）─────
def call_llm_direct(prompt: str) -> str:
    """直接调用底层LLM，使用TravelPlanner标准prompt"""
    from core.config import get_llm
    from langchain_core.prompts import ChatPromptTemplate

    if args.model:
        import os
        os.environ["LOCAL_LLM_MODEL"] = args.model
        print(f"🔄 使用模型: {args.model}")
    llm = get_llm(temperature=0.1)  # 低temperature保证可复现

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", TP_SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    try:
        resp = llm.invoke(chat_prompt.format_messages(input=prompt))
        content = getattr(resp, "content", "") if not isinstance(resp, str) else resp
        return content or ""
    except Exception as e:
        print(f"  [ERROR] LLM call failed: {e}")
        return ""


# ── 主流程 ────────────────────────────────────────────
def run_eval(mode: str, limit: int = None):
    from datasets import load_dataset

    if mode == "parse":
        sample_text = """Day 1:
Current City: from Washington to Myrtle Beach
Transportation: Flight Number: F3927581, from Washington to Myrtle Beach, Departure Time: 11:03, Arrival Time: 13:31
Breakfast: -
Attraction: SkyWheel Myrtle Beach, Myrtle Beach
Lunch: Catfish Charlie's, Myrtle Beach
Dinner: Exotic India, Myrtle Beach
Accommodation: A WONDERFUL Place is Waiting 4U in Brooklyn!!!, Myrtle Beach

Day 2:
Current City: Myrtle Beach
Transportation: -
Breakfast: First Eat, Myrtle Beach
Attraction: WonderWorks Myrtle Beach, Myrtle Beach; Broadway at the Beach, Myrtle Beach
Lunch: Nagai, Myrtle Beach
Dinner: Twigly, Myrtle Beach
Accommodation: A WONDERFUL Place is Waiting 4U in Brooklyn!!!, Myrtle Beach

Day 3:
Current City: from Myrtle Beach to Washington
Transportation: Flight Number: F3791200, from Myrtle Beach to Washington, Departure Time: 11:36, Arrival Time: 13:06
Breakfast: La Pino'z Pizza, Myrtle Beach
Attraction: Myrtle Beach State Park, Myrtle Beach
Lunch: -
Dinner: -
Accommodation: -"""
        result = parse_plan_text(sample_text, n_days=3)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print("加载TravelPlanner validation数据集...")
    data = load_dataset('osunlp/TravelPlanner', 'validation')['validation']
    data = [dict(x) for x in data]
    print(f"共 {len(data)} 条样本")

    if limit:
        data = data[:limit]

    results = []
    for idx, sample in enumerate(data):
        if idx < args.start_idx:
            continue
        n_days = int(sample.get("days", 3))
        level  = sample.get("level", "?")
        query  = sample.get("query", "")
        print(f"\n[{idx+1}/{len(data)}] level={level}, days={n_days}")
        print(f"  query: {query[:70]}...")

        prompt = build_prompt(sample)
        approx_tokens = len(prompt) // 4
        print(f"  prompt ~{approx_tokens} tokens")
        raw_text = call_llm_direct(prompt)
        print(f"  raw ({len(raw_text)} chars): {raw_text[:120].strip()}...")

        plan = parse_plan_text(raw_text, n_days)
        day1 = plan[0]
        print(f"  day1: city={day1['current_city'][:50]}, transport={day1['transportation'][:50]}")

        results.append({"plan": plan})

        if (idx + 1) % 10 == 0:
            _save_jsonl(results, OUTPUT_FILE)
            print(f"  [checkpoint] 已保存 {len(results)} 条")

        time.sleep(0.3)

    _save_jsonl(results, OUTPUT_FILE)
    print(f"\n完成！{len(results)} 条 → {OUTPUT_FILE}")
    if len(results) < 180 and limit is None:
        print(f"[WARNING] 少于180条，评估脚本会报错")


def _save_jsonl(records: list, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["parse", "test", "run"], default="parse")
    parser.add_argument("--model", type=str, default=None,
                        help="覆盖默认模型，例如 qwen2.5:14b")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="从第N条开始跑，用于断点续跑")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "test" and args.limit is None:
        args.limit = 1
    run_eval(args.mode, args.limit)
def build_prompt_v2(sample: dict) -> str:
    """加强约束版prompt，减少幻觉"""
    query = sample.get("query", "")
    ref_info = format_reference_info(sample.get("reference_information", ""))
    instruction = """You are a travel planner. Create a detailed travel plan based ONLY on the information provided below.
STRICT RULES:
1. Only use flight numbers that appear in the given information. If no flight is listed, use self-driving or taxi instead.
2. Only use restaurant names from the given information.
3. Only use accommodation names from the given information.
4. Never invent or guess any information. If data is missing, use "-".
5. Follow the exact output format shown."""
    return f"{instruction}\n\nGiven information:\n{ref_info}\n\nQuery: {query}\n\nTravel Plan:"
