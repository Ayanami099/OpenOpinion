# Campus Opinion Agent

面向交大校园语境的舆情分析 Agent 原型。当前版本先实现水源社区文本抓取能力：可申请水源 Discourse User API Key，抓取热帖、搜索帖子，并读取指定帖子的评论，为后续舆情监控、风险评分和报告生成做数据准备。

## 当前结构

```text
src/campus_opinion_agent/collectors/shuiyuan.py  # 水源抓取核心逻辑
scripts/get_shuiyuan_api_key.py                  # 申请水源 User API Key
scripts/fetch_shuiyuan.py                        # 抓取热帖/搜索/读帖 CLI
scripts/fetch_tikhub.py                          # 通过 TikHub 抓取校外平台数据
data/raw/                                        # 水源原始抓取 JSONL
data/external/tikhub/                            # TikHub 校外平台输出
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
  "shuiyuan_cookies": {},
  "tikhub": {
    "api_key": "...",
    "output_dir": "data/external/tikhub"
  }
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

当前先接入知乎、小红书和微博，输出到 `data/external/tikhub/{platform}/jsonl/`，字段尽量兼容后续统一证据结构。

配置 API Key 可以写入 `config.json` 的 `tikhub.api_key`，也可以使用环境变量：

```bash
export TIKHUB_API_KEY="YOUR_TIKHUB_API_KEY"
```

搜索知乎内容：

```bash
python scripts/fetch_tikhub.py zhihu-search "上海交通大学" --max-results 10
```

搜索并抓取回答评论：

```bash
python scripts/fetch_tikhub.py zhihu-search "上海交通大学" --max-results 10 --comments --max-comments-per-content 50
```

单独抓取某个知乎回答的评论：

```bash
python scripts/fetch_tikhub.py zhihu-comments 2017029304488318722 --max-comments 100
```

搜索小红书笔记：

```bash
python scripts/fetch_tikhub.py xiaohongshu-search "上海交通大学" --max-results 20
```

搜索并抓取小红书评论：

```bash
python scripts/fetch_tikhub.py xiaohongshu-search "上海交通大学" --max-results 10 --comments --max-comments-per-note 50
```

单独抓取某篇小红书笔记的评论：

```bash
python scripts/fetch_tikhub.py xiaohongshu-comments 686fc2c100000000110007de --max-comments 100
```

搜索微博内容：

```bash
python scripts/fetch_tikhub.py weibo-search "上海交通大学" --max-results 20
```

搜索并抓取微博评论：

```bash
python scripts/fetch_tikhub.py weibo-search "上海交通大学" --max-results 10 --comments --max-comments-per-post 50
```

单独抓取某条微博的评论：

```bash
python scripts/fetch_tikhub.py weibo-comments 5299961121999321 --max-comments 100
```

当前约定：

```text
水源数据：data/raw/
TikHub 知乎内容：data/external/tikhub/zhihu/jsonl/search_contents_YYYY-MM-DD.jsonl
TikHub 知乎评论：data/external/tikhub/zhihu/jsonl/search_comments_YYYY-MM-DD.jsonl
TikHub 小红书笔记：data/external/tikhub/xiaohongshu/jsonl/search_notes_YYYY-MM-DD.jsonl
TikHub 小红书评论：data/external/tikhub/xiaohongshu/jsonl/search_comments_YYYY-MM-DD.jsonl
TikHub 微博内容：data/external/tikhub/weibo/jsonl/search_posts_YYYY-MM-DD.jsonl
TikHub 微博评论：data/external/tikhub/weibo/jsonl/search_comments_YYYY-MM-DD.jsonl
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
