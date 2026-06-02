# OpenOpinion

OpenOpinion 是一个面向中文事件与舆情的 Stage-1 速读 Agent。当前版本以“提示词 + 规则”为核心：输入一个事件描述后，系统会完成事件理解、搜索规划、证据过滤、观点挖掘、时间线构建、风险评估，并生成带证据记录的 Markdown/PDF 简报。

项目目前更接近可验证原型，而不是完整生产系统。它已经具备清晰的流水线、结构化日志、训练数据导出和可替换的搜索/模型接口；下一步重点是补齐真实多平台数据采集、更强的文本分析与可视化、报告表达质量和前端工作台。

## 当前能力

- 事件理解：识别主体、关键词、事件类型、风险等级、地域范围、候选平台和检索意图。
- 多轮检索规划：按 FACT、OFFICIAL_RESPONSE、MEDIA_COVERAGE、PUBLIC_REACTION、RUMOR_CHECK、RISK_IMPACT 等意图生成查询，并根据覆盖度补搜。
- 搜索执行：默认使用 Tavily；没有 `TAVILY_API_KEY` 或请求失败时回退到 Mock provider，方便本地测试。
- 证据过滤：对相关性、可信度、新鲜度、意图匹配、重复度进行打分，并形成 coverage matrix。
- 舆情分析：基于规则做情感、立场、观点聚类、关键词、时间线和风险维度分析。
- 报告生成：支持 LLM 叙事报告，也支持规则模板报告；每次运行输出 Markdown、PDF、session JSON、trace JSON 和 SQLite 记录。
- 训练视图：可从 session 中导出 SFT、偏好样本和 RL trajectory 的 JSONL。

## 架构概览

```text
用户输入
  -> ResearchStage
     -> QueryUnderstandingEngine
     -> SearchPlanner / IterativeSearchController
     -> SearchExecutor
     -> EvidenceFilter
  -> AnalysisStage
     -> OpinionMiner
     -> TimelineBuilder
     -> RiskAssessor
     -> TextStatsAnalyzer
  -> ReportStage
     -> ReportGenerator
     -> PdfReportGenerator
  -> SessionLogger
     -> data/sessions/*.json|*.md|*.pdf|*.trace.json
     -> data/openopinion.sqlite3
```

主要代码目录：

```text
src/opinion_agent/
  agent/          # 总流水线 OpinionAnalysisPipeline
  research/       # 事件理解、检索规划、搜索执行、证据过滤
  analysis/       # 舆情、时间线、风险、文本统计
  reporting/      # Markdown 报告生成与 PDF 渲染
  logging/        # session 保存、trace、训练数据视图
  eval/           # benchmark 与 baseline 评估
  schemas.py      # 全流程 Pydantic 数据结构
```

## 快速开始

```powershell
Copy-Item .env.example .env
# 编辑 .env，填入 Tavily 和 OpenAI-compatible 模型配置

uv run opinion-agent analyze "某奶茶品牌被曝使用过期原料" --max-rounds 3 --top-k 5 --out data/sessions
```

常用命令：

```powershell
uv run opinion-agent analyze "事件描述" --max-rounds 3 --top-k 5 --query-budget 12 --out data/sessions
uv run opinion-agent eval --cases data/benchmark_cases.json --out data/eval_results.json
uv run opinion-agent export-training --sessions data/sessions
uv run pytest
```

## 配置

复制 `.env.example` 到 `.env` 后配置：

```dotenv
TAVILY_API_KEY=

LLM_API_KEY=
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4.1-mini

EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
```

说明：

- LLM 与 embedding 可以使用不同的 OpenAI-compatible provider。
- Shell 或 CI 中已有的环境变量优先级高于 `.env`。
- 旧的 `OPENAI_*` 变量仍可作为 fallback alias。
- `REPORT_USE_LLM=false` 可关闭 LLM 报告生成，使用规则模板报告。

## 输出产物

默认输出到 `data/sessions` 和 `data/openopinion.sqlite3`：

- `<session_id>.json`：完整结构化 session。
- `<session_id>.md`：Markdown 速读报告。
- `<session_id>.pdf`：由 Markdown 渲染出的 PDF 报告。
- `<session_id>.trace.json`：Agent 轨迹，包括理解、规划、搜索、过滤、分析和报告 prompt 信息。
- `data/openopinion.sqlite3`：sessions、queries、search_results、evidences、opinions、risk_metrics 等表。
- `data/training/*.jsonl`：执行 `export-training` 后生成的训练样本。

## 需要修改和增强的点

### 1. 多平台数据爬取

当前 `SearchExecutor` 只有 Tavily 与 Mock provider，`platform` 字段更多是规划标签，还没有真正按平台采集。建议：

- 新增 provider 抽象实现：微博、小红书、知乎、抖音/B站、新闻站点、政府/监管官网、品牌官方渠道。
- 为每个平台定义统一结果 schema：标题、正文、发布时间、作者、互动数、转发/评论链路、原始 URL、抓取时间。
- 将 `configs/platform_rules.yaml` 从配置说明扩展为平台路由、限流、可信度、字段映射和错误重试规则。
- 加入去重与 canonical URL 归一化，避免同源转载、镜像页和 UTM 参数造成证据膨胀。
- 增加合规策略：robots、登录态、频率限制、敏感平台授权和失败回退。

### 2. 文本分析和可视化

当前 `TextStatsAnalyzer` 已有平台分布、来源类型、情感分布、关键词、风险词、质量均值等统计，但可视化只作为 `visual_evidence` 数据写入报告 trace，尚未形成独立前端图表。建议：

- 增强中文 NLP：分词、实体识别、话题聚类、摘要、谣言/事实断言抽取、评论噪声过滤。
- 引入 embedding 聚类，替代纯规则观点聚类，支持相似观点合并、争议面向发现和观点演化。
- 增加传播分析：时间热度曲线、平台扩散路径、关键节点账号、媒体与社交平台联动。
- 将 `visual_evidence` 落地为图表数据 API：情感分布、立场分布、来源-情感矩阵、风险雷达、证据覆盖表、关键词趋势。
- 为报告和前端共享一套 chart spec，避免报告、PDF 和 UI 各自拼数据。

### 3. 报告写作优化

当前报告生成可用，但代码里部分中文 prompt/模板出现编码异常，且报告更偏“结果罗列”。建议：

- 修复源码中乱码的中文 prompt、规则词典和模板，统一为 UTF-8。
- 把报告结构拆成可配置模板：核心判断、事实脉络、证据分层、舆情焦点、风险与不确定性、后续关注。
- 强化引用约束：每个关键判断必须绑定 evidence id，区分事实、媒体报道、网友评论和推测。
- 增加报告质量评估：faithfulness、coverage、citation density、重复度、事实/观点混写检测。
- 支持不同报告类型：管理层摘要、PR 应对简报、监管/法务关注版、日报/持续追踪版。

### 4. 前端

当前项目只有 CLI，没有前端工作台。建议新增一个轻量前端：

- 输入事件、设置检索轮数/top-k/query budget、选择平台与报告类型。
- 实时展示流水线状态：理解结果、检索计划、每轮证据数量、过滤原因、覆盖度。
- 提供证据面板：按意图、平台、可信度、时间排序，支持查看原文、摘要、分数和引用关系。
- 提供舆情看板：情感、立场、关键词、风险维度、时间线和来源分布。
- 支持报告预览、编辑、导出 Markdown/PDF，并能回看历史 session。
- 后端可先用 FastAPI 暴露现有 `OpinionAnalysisPipeline`，前端用 React/Vite 或 Streamlit 快速验证。

## 当前注意事项

- 没有配置 Tavily 时会使用 Mock 数据，适合测试流程，但不代表真实舆情结果。
- 当前舆情分析主要依赖规则词典，面对复杂中文表达、反讽、跨平台语境时会有偏差。
- PDF 渲染依赖本机中文字体；缺少中文字体时可能显示效果较差。
- 代码中存在中文字符串编码异常，应优先修复，否则会影响规则匹配、prompt 质量和报告可读性。
