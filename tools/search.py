# tools/search.py
import os
import json
import requests
import re
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
# 即使不使用 Wiki 功能，保留 import 以免报错，或者你可以简化掉它
try:
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from core.config import embeddings
except ImportError:
    pass # 忽略缺少库的情况

@tool()
def google_search(query: str) -> str:
    """Google Search"""
    if not os.getenv("SERPER_API_KEY"):
        return "Error: Missing SERPER_API_KEY"
    serper = GoogleSerperAPIWrapper(serper_api_key=os.getenv("SERPER_API_KEY"))
    return serper.run(query)

@tool()
def tavily_search(query: str) -> str:
    """Tavily Search"""
    key = os.getenv("TAVILY_API_KEY")
    if not key: return "Error: Missing TAVILY_API_KEY"
    payload = {"api_key": key, "query": query, "search_depth": "advanced"}
    try:
        r = requests.post("https://api.tavily.com/search", json=payload, timeout=10)
        return json.dumps(r.json(), ensure_ascii=False)
    except Exception as e:
        return str(e)

@tool()
def wiki_sg_retrieve(query: str) -> str:
    """Wikipedia Placeholder (简化版，防止 FAISS 报错)"""
    return json.dumps({"snippet": "Wikipedia search is temporarily disabled in local mode.", "source": "wiki"}, ensure_ascii=False)