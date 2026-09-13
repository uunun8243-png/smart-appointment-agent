# 评估（Evaluation）

离线评估脚本，用于回归三个核心质量/性能指标。**运行前请先在 `.env` 中填好模型与 Embedding 配置**（`setup-environment` skill 可一键完成）。

| 脚本 | 指标 | 对应面试真题 |
|------|------|--------------|
| `python eval/evaluate_intent.py` | 意图路由准确率（appointment / query / other 分类） | 路由准确率、Agent 评估（RQ11） |
| `python eval/evaluate_rag.py` | RAG Top-K 命中率 + 检索延迟 | RAG 评估（RQ06）、延迟（RQ10） |
| `python eval/benchmark_latency.py` | 首 token 延迟 / 完整响应延迟 | 端到端延迟（RQ10） |

标注数据在 `eval/datasets.py`，可直接扩充。命中标准采用「分类命中」，不绑定具体文档 ID，知识库增删后仍可复跑。

> 注意：意图路由与 RAG 评估依赖真实 LLM / Embedding 服务，未配置 API Key 时会直接报模型调用错误；这是预期行为，配好 `.env` 即可。
