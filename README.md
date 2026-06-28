# AI 研究助手 Agent (Week 4)

基于 ReAct 模式的多工具自主编排 AI Agent，能根据用户意图自主规划工具调用链，完成跨工具协作推理任务。

## 核心能力

- **自主决策**：Agent 自行判断用哪个工具、什么顺序、如何组合结果
- **ReAct 循环**：Thought → Action → Observation 循环，直至给出 Final Answer
- **SSE 流式输出**：实时推送每一步推理过程到前端
- **多模板对比**：3 种 ReAct Prompt 模板 + LLM-as-Judge 自动评估

## 内置工具

| 工具 | 功能 |
|------|------|
| `search_knowledge_base` | 查询本地 ChromaDB 多模态 RAG 知识库 |
| `search_web` | 模拟联网搜索（Mock 数据） |
| `analyze_image` | 调用 Vision LLM 分析图片内容 |
| `calculator` | 安全的数学表达式求值（AST 解析） |
| `get_current_time` | 获取当前日期时间 |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量（复制 .env.example 为 .env，填入 API Key）
cp .env.example .env

# 3. 启动服务
uvicorn main:app --reload --port 8000

# 4. 浏览器访问
http://localhost:8000
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/agent/chat` | POST | Agent 对话（SSE 流式输出） |
| `/api/v1/agent/sessions` | GET | 会话历史列表 |
| `/api/v1/agent/sessions/{id}/trace` | GET | 完整推理轨迹（JSON） |
| `/api/v1/agent/evaluate` | POST | 多模板对比评估 |

## 项目结构

```
├── app/
│   ├── routers/agent.py      # Agent API 路由
│   ├── schemas/agent.py      # Pydantic 数据模型
│   ├── services/
│   │   ├── agent.py          # ReAct Agent 引擎
│   │   ├── tools.py          # 工具注册中心
│   │   ├── prompts.py        # 3 种 ReAct Prompt 模板
│   │   ├── llm.py            # LLM 调用封装
│   │   ├── evaluator.py      # LLM-as-Judge 评估器
│   │   ├── vector_store.py   # ChromaDB 向量存储
│   │   ├── retriever.py      # 多模态检索器
│   │   ├── embedding.py      # Embedding 服务
│   │   ├── clip_embedding.py # CLIP 图片嵌入
│   │   └── sessions.py       # 会话管理
│   └── utils/config.py       # 配置管理
├── static/index.html         # Agent 前端交互界面
├── tests/                    # pytest 测试（10 个用例）
├── main.py                   # FastAPI 入口
└── requirements.txt
```

## 运行测试

```bash
pytest tests/ -v
# 10 passed
```

## 技术栈

- **后端**：FastAPI + SSE
- **Agent 引擎**：ReAct 模式 + OpenAI SDK Function Calling
- **LLM**：智谱 AI glm-4-flash（文本）/ glm-4.6v-flash（视觉）
- **向量库**：ChromaDB
- **测试**：pytest + httpx
