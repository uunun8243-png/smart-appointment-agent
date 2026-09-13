"""意图路由准确率评估

衡量 TaskClassifier 是否把用户输入正确分类为 appointment / query / other。
这是多 Agent 系统的第一道关卡：路由错了，后面的预约/咨询/兜底全都会错。

用法：
    python eval/evaluate_intent.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from config.model_provider import create_chat_model
from agents.task_classification.task_classifier import TaskClassifier
from eval.datasets import INTENT_DATASET


async def main() -> int:
    llm = create_chat_model(temperature=0)
    classifier = TaskClassifier(llm)

    correct = 0
    total = len(INTENT_DATASET)
    per_class = {}

    for i, (text, expected) in enumerate(INTENT_DATASET, 1):
        try:
            pred = await classifier.classify_task(text)
        except Exception as e:  # noqa: BLE001
            pred = f"error: {type(e).__name__}"

        ok = pred == expected
        correct += bool(ok)

        key = (expected, pred)
        per_class[key] = per_class.get(key, 0) + 1

        mark = "✓" if ok else "✗"
        print(f"[{i:2d}] {mark} {text!r}  期望={expected}  实际={pred}")

    acc = correct / total
    print(f"\n意图路由准确率: {correct}/{total} = {acc:.1%}")

    # 分类混淆统计
    print("\n混淆统计 (期望 -> 实际):")
    for (exp, pred), count in sorted(per_class.items()):
        print(f"  {exp} -> {pred}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
