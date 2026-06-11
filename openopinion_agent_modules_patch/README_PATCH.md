# OpenOpinion Agent Modules Patch

这个压缩包不是完整 OpenOpinion 项目，而是本次新增和修改的 Agent 模块补丁。

## 包含模块

1. 统一主题运行入口
- src/openopinion_case_runner/

2. 校园多平台采集相关模块
- src/campus_opinion_agent/
- scripts/fetch_shuiyuan.py
- scripts/get_shuiyuan_api_key.py
- scripts/run_mediacrawler.py

3. 多平台文本统一输入与本地模型分析
- src/openopinion_text_analysis/

4. 报告生成链路
- src/openopinion_report_enhancer/

5. 修改过的原项目文件
- src/opinion_agent/llm.py
- pyproject.toml
- uv.lock
- .gitignore

## 不包含内容

- data/
- external/
- .venv/
- .venv_mediacrawler/
- configs/campus_config.local.json
- openopinion.sqlite3
- 任何 API key、cookie、session 结果或生成报告

## 使用方式

把本压缩包解压到原始 OpenOpinion 项目根目录，覆盖同名文件。

然后运行：

uv sync --extra dev

测试：

uv run --extra dev pytest

主要命令：

uv run openopinion-run-topic
uv run openopinion-build-text-input
uv run openopinion-local-text-analysis
uv run openopinion-build-report-context
uv run openopinion-generate-report-insights
uv run openopinion-render-final-report
