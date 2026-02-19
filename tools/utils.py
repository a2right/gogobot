# tools/utils.py
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_core.tools import tool

@tool()
def now_sg() -> str:
    """返回新加坡当前日期时间"""
    try:
        dt = datetime.now(ZoneInfo("Asia/Singapore"))
        return json.dumps({
            "iso": dt.isoformat(timespec="seconds"),
            "text": dt.strftime("%A, %B %d, %Y %H:%M"),
            "tz": "Asia/Singapore"
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

@tool()
def truncate_memory(checkpoint_json: str, limit_messages: int = 6) -> str:
    """裁剪对话历史内存"""
    try:
        msgs = json.loads(checkpoint_json) or []
        if not isinstance(msgs, list): return "[]"
        return json.dumps(msgs[-int(limit_messages):], ensure_ascii=False)
    except Exception:
        return "[]"