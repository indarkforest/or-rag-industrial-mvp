# 工业场景og-rag效果验证 MVP

验证目标：**混合检索架构（hybrid-rag）在工业场景下对 Agent 问答效果的提升**。

三种检索方式对照：
- **naive-rag**：纯向量检索原始文档 chunk
- **og-rag**：本体驱动检索，事实块 + 图谱扩展 + 实时数据
- **hybrid-rag**：混合架构，事实块 + 原文 chunk + 图谱扩展 + 实时数据

实现思路参考 [microsoft/ograg2](https://github.com/microsoft/ograg2)：
文档 + 工业本体（JSON-LD）→ LLM 本体映射抽取三元组 → 按主语聚合为事实块（超边）→
向量检索事实块 + 图谱多跳扩展 → LLM 回答。与纯向量检索（naive-RAG）做同题对照，LLM-as-judge 评分。

## 架构

```
data/docs (工业文档)              data/ontology (工业本体 JSON-LD)
        \                           /
         KGBuilder (LLM 本体映射抽取)
                  |
         SQLite (nodes/edges/chunks/facts)
            /        |        \
   NaiveRetriever  OGRAGRetriever  HybridRetriever
            \        |        /
              QAAgent (MiniMax-M3)
                  |
        evaluate (LLM-as-judge 对照评分)
```

- 模拟场景：苯乙烯聚合反应釜 R-101，含设备/工艺/控制/因果四类知识，覆盖多跳因果链
  （仪表风低压 → 阀门开度不足 → 冷却水流量下降 → 釜温上升 → 报警/联锁）。
- 本体对齐架构图中的工业知识底座：`Device` 设备本体、`ProcessParameter` 工艺本体、
  `ControlLoop` 控制本体、`causes` 因果模型。

## 快速开始

### 方式一：一键脚本（推荐）

**Windows:**
```bat
双击 start.bat
```

**Linux / macOS:**
```bash
chmod +x start.sh
./start.sh
```

脚本会自动：创建虚拟环境 → 安装依赖 → 启动 Streamlit

启动后浏览器访问 `http://localhost:8501`，在左侧侧边栏配置 API Key 即可使用。

### 方式二：手动启动

```bash
# 1. 创建虚拟环境并安装依赖
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt

# 2. 配置 API Key（任选一种）
#    a) 在应用侧边栏直接输入（推荐）
#    b) 设置环境变量
export MINIMAX_API_KEY="你的key"     # Linux/Mac
set MINIMAX_API_KEY=你的key           # Windows

# 3. 启动
streamlit run app.py
```

### CLI 方式

```bash
# 构建知识图谱
python main.py build

# 单次问答
python main.py query -q "仪表风压力低会对 R-101 温度产生什么影响？" -r hybrid-rag

# 对照评估，输出 eval_report.md
python main.py eval
```

## 使用流程

1. **配置**：侧边栏填写 API Key → 点击「应用配置」
2. **上传文档**：「构建知识库」页 → 上传 .md 文档
3. **构建**：点击「开始构建」（LLM 抽取实体/关系，约 1-3 分钟）
4. **问答**：「问答对比」页 → 输入问题 → 三路对比回答 → 对比分析
5. **评估**：「对照评估」页 → 运行评估 → 查看评分
6. **实时数据**：「实时数据」页 → 生成模拟数据 → 趋势查看

## 配置说明

### 侧边栏配置（页面内）
- **API Key**：MiniMax API Key（密码输入，不回显）
- **LLM 模型**：如 MiniMax-M3、MiniMax-M1
- **API Base URL**：OpenAI 兼容接口地址
- **Embedding 模型**：如 embo-01
- **检索参数**：top_k、图谱扩展跳数（滑块调整）

### config.yaml（可选）
侧边栏配置优先于 config.yaml。如需预配置，编辑 `config.yaml`：
- `model`：LLM 模型与 API 地址
- `embedding`：Embedding 模型（`api` 或 `hashing`）
- `retrieval`：检索参数默认值

### 环境变量（可选）
复制 `.env.example` 为 `.env` 并填写 API Key，或直接设置环境变量：
```
MINIMAX_API_KEY=你的key
```

## 验证方法

`data/questions.yaml` 内置 6 题（2 题简单事实 + 4 题多跳因果），每题带参考答案要点。
评估命令对三种检索方式同题作答，由 LLM 按准确性/完整性/忠实度打分（1~5）。

## 局限（MVP 范围）

- SQLite 代替图数据库，多跳遍历用邻接查询实现
- 评估为 LLM-as-judge + 小问题集，非 ragas 级别的严格评估
- KG 抽取质量依赖 LLM，未做人工校验环节
- 实时数据为模拟生成，非真实工况采集
