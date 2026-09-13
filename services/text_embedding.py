"""文本向量化与相似度匹配

统一封装 Embedding 能力，供 RAG 知识检索和技师专长匹配复用。

设计要点：
1. Embedding 模型懒加载单例：避免每次调用都重新创建模型客户端
   （原实现每次 embed_input 都 new 一个模型，带来连接/初始化开销）。
2. 内存缓存：对重复文本（例如固定候选技师专长、常见查询）命中缓存，
   减少重复向量计算与外部 API 调用。
3. 统一使用余弦相似度：对向量做 L2 归一化后使用内积索引（IndexFlatIP），
   归一化后的内积即为余弦相似度，排序语义更符合文本语义匹配。
"""

from __future__ import annotations

import logging
from typing import List

import faiss
import numpy as np

from config.model_provider import create_embedding_model

logger = logging.getLogger(__name__)

# 模型客户端懒加载单例
_embedding_model = None

# 文本 -> 向量的内存缓存（有界，避免无限增长）
_embedding_cache: dict = {}
_EMBEDDING_CACHE_MAX = 8192


def get_embedding_model():
    """获取全局共享的 Embedding 模型客户端（懒加载）。"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = create_embedding_model()
    return _embedding_model


def embed_input(input_text: str) -> List[float]:
    """将文本转为向量，带内存缓存。"""
    if not input_text:
        return []

    cached = _embedding_cache.get(input_text)
    if cached is not None:
        return cached

    embedding = get_embedding_model().embed_query(input_text)

    # 有界缓存：达到上限时整体清空，简单但可防止内存无界增长
    if len(_embedding_cache) >= _EMBEDDING_CACHE_MAX:
        _embedding_cache.clear()
    _embedding_cache[input_text] = embedding
    return embedding


def clear_embedding_cache() -> None:
    """清空内存缓存（知识库更新后，若需强制重新向量化可调用）。"""
    _embedding_cache.clear()
    logger.info("Embedding 内存缓存已清空")


def _normalize(vector) -> np.ndarray:
    """L2 归一化，使内积等价于余弦相似度。"""
    v = np.asarray(vector, dtype="float32")
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def find_best_match_indices(text: str, candidates: List[str]) -> List[int]:
    """
    在候选文本列表中，按与 text 的语义相似度（余弦）从高到低返回下标。

    返回：候选文本下标列表，按相似度从高到低排序。
    """
    if not candidates:
        return []

    candidate_embs = np.array(
        [_normalize(embed_input(c)) for c in candidates], dtype="float32"
    )
    dimension = candidate_embs.shape[1]

    # 归一化后使用内积索引，等价于余弦相似度排序
    index = faiss.IndexFlatIP(dimension)
    index.add(candidate_embs)

    query_emb = np.array([_normalize(embed_input(text))], dtype="float32")
    k = len(candidates)
    _, indices = index.search(query_emb, k)
    return indices[0][:k].tolist()
