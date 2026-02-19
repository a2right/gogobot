# tools/__init__.py
from .search import google_search, tavily_search, wiki_sg_retrieve
from .places import get_distance_tool, is_open_now, get_detail_place, get_nearby_place
from .media import image_search
from .utils import now_sg, truncate_memory

# 导出工具列表供 Agent 使用
ALL_TOOLS = [
    now_sg,
    image_search,
    get_distance_tool,
    is_open_now,
    get_detail_place,
    get_nearby_place,
    wiki_sg_retrieve,
    tavily_search,
    google_search,
    truncate_memory,
]