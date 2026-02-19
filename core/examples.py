# core/examples.py
import re
from collections import Counter
from typing import List, Dict, Optional

from core.config import embeddings

EXAMPLES = [
    {
        "question": "I have only 2 days in Singapore with a tight budget...",
        "answer": "Perfect! Let me suggest some budget-friendly attractions..."
    },
    {
        "question": "Plan a 3-day Singapore itinerary for a family with kids.",
        "answer": "Day 1: Start with Singapore Zoo in the morning..."
    },
    {
        "question": "What are the best hawker centres in Singapore for local food?",
        "answer": "Top hawker centres include Maxwell Food Centre, Lau Pa Sat..."
    },
    {
        "question": "Recommend a romantic 2-day itinerary in Singapore.",
        "answer": "Day 1: Gardens by the Bay evening light show, dinner at Marina Bay Sands..."
    },
    {
        "question": "I want a culture-focused trip to Singapore covering heritage areas.",
        "answer": "Day 1: Chinatown, Sri Mariamman Temple, Buddha Tooth Relic Temple..."
    },
    {
        "question": "Best outdoor activities and nature spots in Singapore for 3 days.",
        "answer": "Day 1: MacRitchie Reservoir tree-top walk, Bukit Timah Nature Reserve..."
    },
    {
        "question": "Singapore shopping itinerary: Orchard Road and local markets.",
        "answer": "Day 1: Orchard Road, ION Orchard, Ngee Ann City..."
    },
    {
        "question": "It looks like rain during my trip. Suggest a 2-day mostly indoor plan.",
        "answer": "Step 1–3 Options:\n- ArtScience Museum..."
    },
    {
        "question": "Plan a 1-day Singapore itinerary around the Marina Bay area.",
        "answer": "Morning: Marina Bay Sands SkyPark, ArtScience Museum..."
    },
    {
        "question": "Budget trip to Singapore for 3 days under SGD 300.",
        "answer": "Day 1: Free attractions like Gardens by the Bay outdoor gardens..."
    },
]


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

_EMBED_CACHE: Dict[str, List[float]] = {}


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    return [t for t in re.findall(r"[a-z0-9]+", text) if len(t) > 2]


def _tf(text: str) -> Counter:
    return Counter(_tokenize(text))


def _cosine(v1: List[float], v2: List[float]) -> float:
    """通用余弦相似度（要求两个向量等长且顺序对齐）。"""
    if not v1 or not v2:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = sum(a * a for a in v1) ** 0.5
    n2 = sum(b * b for b in v2) ** 0.5
    return 0.0 if not n1 or not n2 else dot / (n1 * n2)


def _cosine_tf(tf1: Counter, tf2: Counter) -> float:
    """
    修复版 TF 余弦相似度。
    原问题：直接使用 Counter.values() 不保证词汇对齐，
    导致两个向量的维度对应不同的词，计算结果毫无意义。
    修复：先构建共同词汇表，再按统一顺序取值。
    """
    vocab = list(set(tf1) | set(tf2))
    if not vocab:
        return 0.0
    v1 = [tf1.get(w, 0) for w in vocab]
    v2 = [tf2.get(w, 0) for w in vocab]
    return _cosine(v1, v2)


def _is_chinese(text: str) -> bool:
    """检测文本是否以中文为主（中文字符占比 > 30%）。"""
    if not text:
        return False
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return chinese_chars / max(len(text), 1) > 0.3


# ─────────────────────────────────────────────
# 主检索函数
# ─────────────────────────────────────────────

def select_examples(
    user_input: str,
    k: int = 4,
    prefer_semantic: bool = True,
) -> List[Dict]:
    """
    智能选择最相关的 few-shot 示例。

    策略优先级：
    1. 语义检索（embedding 余弦）—— 英文输入时首选
    2. TF 余弦回退 —— 中文输入或 embedding 失败时使用
       （修复了原版向量不对齐的 bug）

    Args:
        user_input:      用户当前输入
        k:               返回示例数量
        prefer_semantic: 是否优先语义检索（中文输入会自动降级为 TF）

    Returns:
        最多 k 条示例 [{"question": ..., "answer": ...}]
    """
    if not EXAMPLES:
        return []

    k = min(k, len(EXAMPLES))
    selected: List[Dict] = []

    # ── 1. 语义检索（embedding） ──────────────────────────────────
    # 中文输入强制回退到 TF，避免 mxbai-embed-large 中文支持不稳定
    use_semantic = prefer_semantic and not _is_chinese(user_input)

    if use_semantic:
        try:
            u_vec = embeddings.embed_query(user_input)
            scored = []
            for ex in EXAMPLES:
                q = ex["question"]
                if q not in _EMBED_CACHE:
                    _EMBED_CACHE[q] = embeddings.embed_query(q)
                score = _cosine(u_vec, _EMBED_CACHE[q])
                scored.append((score, ex))
            scored.sort(key=lambda x: x[0], reverse=True)
            selected = [ex for _, ex in scored[:k]]
        except Exception:
            selected = []   # embedding 失败 → 降级到 TF

    # ── 2. TF 余弦回退 ────────────────────────────────────────────
    if not selected:
        u_tf = _tf(user_input)
        scored_tf = []
        for ex in EXAMPLES:
            score = _cosine_tf(u_tf, _tf(ex["question"]))  # ← 修复点
            scored_tf.append((score, ex))
        scored_tf.sort(key=lambda x: x[0], reverse=True)
        selected = [ex for _, ex in scored_tf[:k]]

    return selected