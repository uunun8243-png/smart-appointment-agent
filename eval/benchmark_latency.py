"""端到端流式延迟基准

衡量用户输入到首个输出 token（first-token latency）以及完整响应（full-response latency）。
这是 RQ10「端到端延迟是多少」的直接数据来源。

用法：
    python eval/benchmark_latency.py
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from api.chat_handler import task_agent

SAMPLES = [
    "你们营业到几点？",
    "我想预约明天下午3点的按摩",
    "今天天气怎么样？",
]


async def measure(user_input: str):
    """返回 (first_token_ms, total_ms, output_chars)"""
    t0 = time.perf_counter()
    first_token_ms = None
    n_chars = 0

    async for token in task_agent.classify_task_stream(user_input):
        if first_token_ms is None:
            first_token_ms = (time.perf_counter() - t0) * 1000
        n_chars += len(token)

    total_ms = (time.perf_counter() - t0) * 1000
    return first_token_ms, total_ms, n_chars


async def main() -> int:
    print(f"样本数: {len(SAMPLES)}\n")
    for s in SAMPLES:
        try:
            ft, total, chars = await measure(s)
            print(f"{s!r}\n  first-token: {ft:.0f}ms  完整响应: {total:.0f}ms  "
                  f"输出字符: {chars}\n")
        except Exception as e:  # noqa: BLE001
            print(f"{s!r}\n  测量失败: {type(e).__name__}: {e}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
