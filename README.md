# 智能预约 Agent（Smart Appointment Agent）

面向本地服务门店（按摩 / 推拿等）的智能预约与服务咨询系统。基于 FastAPI + LangChain + FAISS + SQLite 构建，采用**中心化多 Agent 编排**，把「意图识别 → 预约信息抽取 → 技师匹配 → 知识问答 → 用户偏好沉淀 → 回访推荐」串成一条自动化服务链路。

核心目标：把门店前台高频重复的工作（咨询服务项目、匹配技师、检查档期、生成预约、记录偏好、回访触达）自动化，降低人工沟通成本与排班冲突。

## 核心能力

- **智能任务分类与路由**：自动识别预约 / 咨询 / 无关请求，路由到对应 Agent。
- **多 Agent 协作**：任务分类、预约、咨询、用户行为分析四个 Agent 分工处理，降低单模块职责膨胀。
- **RAG 知识问答**：SQLite 存知识 + FAISS 向量检索 + LLM 生成，支持流式输出。
- **智能预约**：从自然语言抽取预约信息，基于技师专长 Embedding 相似度匹配，检查档期，处理指定技师不可用时的替代推荐。
- **用户行为与偏好**：记录预约/咨询行为，沉淀长期偏好，定时生成回访提醒与服务推荐。
- **流式响应**：FastAPI AsyncGenerator 输出，并以 `[THOUGHT]` / `[REPLY]` 标记展示思考过程。
- **Embedding 缓存**：模型客户端单例化 + 内存缓存，减少重复向量计算与外部调用。
- **离线评估**：提供意图路由准确率、RAG 命中率、延迟基准三套评估脚本。

## 系统架构

采用严格的五层架构，核心原则是**下层不能反向调用上层**，避免循环依赖：

```text
Web & Application Layer
    ↓  app.py, web/：页面、路由入口、系统启动
API Layer
    ↓  api/：外部接口、请求编排、响应封装
Agents Layer
    ↓  agents/：AI Agent、任务路由、对话流程控制
Services Layer
    ↓  services/：业务逻辑、推荐算法、向量处理
DB Layer
    ↓  db/：数据模型、数据库连接、Repository
```

- Web 层调用 API 层
- API 层调用 Agents 层或 Services 层
- Agents 层调用 Services 层
- Services 层调用 DB 层

禁止下层反向调用上层，禁止 Agents 层绕过 Services 直接访问 DB。

## Agent 设计

### Task Classification Agent（主调度器）

用户输入 → 意图分析 → Agent 路由 → 响应协调。维护对话状态，判断意图，把请求分发给预约 / 咨询 Agent，并处理无关请求的兜底。

### Consultation Agent（RAG 咨询）

任务分类 → 知识检索 → FAISS 相似度搜索 → 流式回答。区分咨询类型，检索知识库，构建提示词并生成自然语言回答。

### Appointment Agent（智能预约）

解析预约需求 → 技师匹配 → 预约确认。抽取时间、项目、时长、性别与技师偏好，匹配技师，处理信息缺失时的追问与技师不可用时的替代推荐。

### User Behavior Agent（行为分析）

行为记录 → 模式分析 → 偏好更新 → 个性化推荐。记录交互与预约行为，分析偏好模式，为后续推荐与回访提供依据。

## 核心改进

在开源基础版本之上完成的增强：

1. **Embedding 缓存与统一相似度**：`services/text_embedding.py` 将 Embedding 模型客户端改为懒加载单例（原实现每次调用都重新创建模型），并加入有界内存缓存；相似度匹配统一为「L2 归一化 + 内积」的余弦相似度，语义排序更稳定。
2. **补全推荐生成逻辑**：`RecommendationService.generate_recommendations_job` 原为空的 TODO 桩，现已实现基于行为统计的回访提醒与基于偏好的服务建议生成，并落库到 `UserRecommendation`，带待发送去重。
3. **离线评估体系**：新增 `eval/` 目录，可测量意图路由准确率、RAG Top-K 命中率与检索延迟、首 token / 完整响应延迟。
4. **代码清理**：删除 `TechnicianService` 中重复定义的方法、`db/models.py` 重复 import 等死代码。

## 技术栈

- 后端框架：FastAPI、Uvicorn
- AI 框架：LangChain
- 大模型接入：兼容 OpenAI 格式的模型提供商（Qwen、DeepSeek、Zhipu、OpenAI、Azure OpenAI）
- 向量检索：FAISS
- 数据库：SQLite、SQLAlchemy
- RAG：Embedding、向量索引、知识库检索、提示词构建
- 流式响应：Python AsyncGenerator
- 外部服务扩展：MCP（天气等外部信息接入）
- 配置管理：python-dotenv
- 后台任务：schedule

## 快速开始

### 1. 创建虚拟环境（推荐 Python 3.10–3.12）

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

> 注意：项目依赖的 LangChain 0.3.x 与 Python 3.13+ 不兼容（PEP 649），请使用 3.10–3.12。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

在 `.env` 中填写模型与 Embedding 配置（两者可分开配置，例如 chat 用 DeepSeek、Embedding 用 Qwen）：

```env
MODEL_PROVIDER=qwen
LLM_API_KEY=...
LLM_BASE_URL=...
LLM_MODEL=...

EMBEDDING_PROVIDER=qwen
EMBEDDING_API_KEY=...
EMBEDDING_BASE_URL=...
EMBEDDING_MODEL=...
```

### 4. 启动服务

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

访问：

- Web 页面：http://127.0.0.1:8000
- API 文档：http://127.0.0.1:8000/docs

## 评估

配置好 `.env` 后运行：

```bash
python eval/evaluate_intent.py    # 意图路由准确率
python eval/evaluate_rag.py       # RAG 命中率 + 检索延迟
python eval/benchmark_latency.py  # 首 token / 完整响应延迟
```

详见 [eval/README.md](eval/README.md)。

## 项目结构

```text
├── agents/                  # 多 Agent 智能层
│   ├── task_classification_agent.py
│   ├── consultant_agent.py
│   ├── appointment_agent.py
│   ├── user_behavior_agent.py
│   └── ...                  # 各 Agent 的子模块
├── api/                     # API 编排层
├── services/                # 业务逻辑层
├── db/                      # 数据持久化层（Repository 模式）
├── config/                  # 配置模块
├── web/                     # Web 页面层
├── eval/                    # 离线评估脚本
├── tests/                   # 测试用例
├── app.py                   # 应用入口
└── requirements.txt
```

## 测试

```bash
pytest
```

