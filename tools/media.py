# tools/media.py
import json
from langchain_core.tools import tool

@tool()
def image_search(query: str) -> str:
    """Image Search Placeholder"""
    return json.dumps({"images": [], "note": "Image search disabled"}, ensure_ascii=False)