"""RAG 检索质量与延迟评估

衡量指标：
1. hit-rate@k：Top-K 检索结果中是否命中目标知识分类（命中率）。
2. 检索延迟：单次向量检索耗时（ms），含 embedding + FAISS 搜索。

用法：
    python eval/evaluate_rag.py
"""

import asyncio
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from services.knowledge_service import KnowledgeService
from eval.datasets import RAG_DATASET

TOP_K = 3


async def main() -> int:
    ks = KnowledgeService()
    await ks.initialize()

    hits = 0
    total = len(RAG_DATASET)
    latencies_ms = []

    for i, (query, expected_cat) in enumerate(RAG_DATASET, 1):
        t0 = time.perf_counter()
        docs = await ks.search(query, top_k=TOP_K)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)

        hit_cats = [d.get("category") for d in docs]
        hit = expected_cat in hit_cats
        hits += bool(hit)

        mark = "✓" if hit else "✗"
        print(f"[{i:2d}] {mark} {query!r}  期望分类={expected_cat}  "
              f"Top{TOP_K}={hit_cats}  ({elapsed_ms:.0f}ms)")

    hit_rate = hits / total
    print(f"\nRAG Top-{TOP_K} 命中率: {hits}/{total} = {hit_rate:.1%}")

    if latencies_ms:
        latencies_ms.sort()
        p50 = statistics.median(latencies_ms)
        p95 = latencies_ms[int(len(latencies_ms) * 0.95) - 1] if len(latencies_ms) >= 20 else latencies_ms[-1]
        print(f"检索延迟: mean={statistics.mean(latencies_ms):.0f}ms  "
              f"p50={p50:.0f}ms  p95={p95:.0f}ms  n={len(latencies_ms)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
