# Campus Opinion Agent

面向交大校园语境的舆情分析 Agent 原型。当前版本先实现水源社区文本抓取能力：可申请水源 Discourse User API Key，抓取热帖、搜索帖子，并读取指定帖子的评论，为后续舆情监控、风险评分和报告生成做数据准备。

## 当前结构

```text
src/campus_opinion_agent/collectors/shuiyuan.py  # 水源抓取核心逻辑
scripts/get_shuiyuan_api_key.py                  # 申请水源 User API Key
scripts/fetch_shuiyuan.py                        # 抓取热帖/搜索/读帖 CLI
scripts/run_mediacrawler.py                      # 调用外部 MediaCrawler
external/MediaCrawler/                           # 校外平台爬虫项目
data/raw/                                        # 水源原始抓取 JSONL
data/external/mediacrawler/                      # MediaCrawler 校外平台输出
data/sessions/                                   # 后续分析 session
data/reports/                                    # 后续报告
```

## 配置

安装依赖：

```bash
pip install -r requirements.txt
```

申请并写入水源 API Key：

```bash
python3 scripts/get_shuiyuan_api_key.py --write-config
```

生成的 `config.json` 会包含：

```json
{
  "shuiyuan_user_api_key": "...",
  "shuiyuan_user_api_client_id": "...",
  "shuiyuan_cookies": {}
}
```

## 使用

抓取最近热帖候选：

```bash
python3 scripts/fetch_shuiyuan.py hot
```

搜索水源帖子：

```bash
python3 scripts/fetch_shuiyuan.py search "关键词"
```

读取指定帖子及评论：

```bash
python3 scripts/fetch_shuiyuan.py topic 471260 --max-posts 100
```

也可以传入帖子 URL。结果会保存到 `data/raw/*.jsonl`，其中第一行通常是帖子信息，后续行为评论数据。

## 校外平台抓取

`MediaCrawler` 已作为外部工具放在 `external/MediaCrawler`。建议通过 wrapper 运行，这样输出会统一保存到 `data/external/mediacrawler`：

```bash
python3 scripts/run_mediacrawler.py --platform zhihu --lt qrcode --type search --keywords "樊思睿" --crawler_max_notes_count 3 --get_comment true --max_comments_count_singlenotes 100
```

当前约定：

```text
水源数据：data/raw/
校外数据：data/external/mediacrawler/
```

## 当前运行逻辑

水源抓取模块使用 Discourse JSON API，不解析网页 HTML：

```text
top/latest/search/topic URL
-> ShuiyuanCollector
-> 请求水源 JSON API
-> 清洗 cooked HTML 为纯文本
-> 保存为 JSONL
```

后续计划是在此基础上增加热度增长监控、情绪与观点分析、校内外对比和 Markdown/PDF 报告生成。
