from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


PLATFORM_ZH = {
    "tavily": "全网搜索",
    "shuiyuan": "水源社区",
    "zhihu": "知乎",
}

SENTIMENT_ZH = {
    "negative": "负面",
    "neutral": "中性",
    "positive": "正面",
    "mixed": "混合",
}

RISK_ZH = {
    "low": "低",
    "medium": "中",
    "high": "高",
}

URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


CSS = r"""
:root {
  --bg: #f4f6fb;
  --panel: #ffffff;
  --text: #111827;
  --subtext: #344054;
  --muted: #667085;
  --line: #e5eaf3;
  --blue: #2f55d4;
  --blue2: #5d7ce3;
  --blue-soft: #eef3ff;
  --red: #df5b4f;
  --green: #34a576;
  --gray: #95a3b8;
  --shadow: 0 16px 42px rgba(30, 41, 59, 0.08);
  --radius: 22px;
}

* {
  box-sizing: border-box;
}

html {
  scroll-behavior: smooth;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
  line-height: 1.68;
  overflow-x: hidden;
}

.page {
  max-width: 1280px;
  margin: 0 auto;
  padding: 34px 28px 64px;
}

.hero {
  background:
    radial-gradient(circle at 85% 12%, rgba(255,255,255,.22), transparent 28%),
    linear-gradient(135deg, #101828 0%, #1d3b88 58%, #3159d4 100%);
  color: #fff;
  border-radius: 30px;
  padding: 38px 44px;
  box-shadow: var(--shadow);
  margin-bottom: 16px;
}

.hero-label {
  font-size: 14px;
  opacity: .78;
  letter-spacing: .08em;
  margin-bottom: 10px;
}

.hero h1 {
  margin: 0;
  font-size: 38px;
  line-height: 1.24;
  font-weight: 800;
  letter-spacing: -0.03em;
}

.hero-sub {
  margin-top: 12px;
  opacity: .86;
  font-size: 15px;
}

.navbar {
  position: sticky;
  top: 0;
  z-index: 50;
  margin: 0 0 18px;
  padding: 10px;
  background: rgba(244, 246, 251, .86);
  backdrop-filter: blur(12px);
  display: flex;
  gap: 8px;
  overflow-x: auto;
  border-radius: 999px;
}

.navbar a {
  flex: 0 0 auto;
  white-space: nowrap;
  text-decoration: none;
  color: #31405d;
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 8px 13px;
  font-size: 13px;
  font-weight: 650;
  box-shadow: 0 6px 18px rgba(30, 41, 59, .05);
}

.navbar a:hover {
  color: var(--blue);
  border-color: #c8d5ff;
  background: var(--blue-soft);
}

.section {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 26px 28px;
  box-shadow: var(--shadow);
  margin-top: 20px;
  scroll-margin-top: 90px;
}

.section h2 {
  margin: 0 0 16px;
  font-size: 24px;
  line-height: 1.3;
  letter-spacing: -0.02em;
}

.section-desc {
  color: var(--muted);
  margin-top: -8px;
  margin-bottom: 18px;
  font-size: 14px;
}

.toc-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}

.toc-card {
  display: block;
  text-decoration: none;
  color: var(--text);
  border: 1px solid var(--line);
  background: linear-gradient(180deg, #fff 0%, #f9fbff 100%);
  border-radius: 18px;
  padding: 18px;
  min-height: 104px;
}

.toc-num {
  color: var(--blue);
  font-size: 13px;
  font-weight: 780;
  margin-bottom: 7px;
}

.toc-title {
  font-weight: 780;
  font-size: 17px;
  margin-bottom: 4px;
}

.toc-desc {
  color: var(--muted);
  font-size: 13px;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 14px;
}

.kpi-card {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 18px;
  min-height: 118px;
}

.kpi-title {
  color: var(--muted);
  font-size: 13px;
  margin-bottom: 10px;
}

.kpi-value {
  font-size: 30px;
  line-height: 1;
  font-weight: 800;
  letter-spacing: -0.03em;
}

.kpi-sub {
  margin-top: 10px;
  color: var(--muted);
  font-size: 13px;
}

.grid-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
}

.chart-card,
.text-panel,
.finding-card,
.topic-card,
.sample-card {
  border: 1px solid var(--line);
  border-radius: 18px;
  background: #fff;
}

.chart-card {
  padding: 18px;
  min-height: 250px;
}

.chart-card h3,
.text-panel h3,
.finding-card h3,
.topic-card h3,
.sample-card h3 {
  margin: 0 0 10px;
  font-size: 17px;
  line-height: 1.42;
}

.text-panel,
.finding-card,
.topic-card,
.sample-card {
  padding: 18px;
}

.text-panel p,
.finding-card p,
.topic-card p,
.sample-card p {
  margin: 0;
  color: var(--subtext);
}

.summary-text p {
  margin: 0 0 12px;
  color: var(--subtext);
}

.report-text-grid,
.topic-grid,
.finding-grid,
.sample-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.donut-wrap {
  display: flex;
  align-items: center;
  gap: 24px;
  min-height: 190px;
}

.donut {
  width: 170px;
  height: 170px;
  border-radius: 50%;
  position: relative;
  flex: 0 0 auto;
}

.donut::after {
  content: "";
  position: absolute;
  inset: 34px;
  background: #fff;
  border-radius: 50%;
  box-shadow: inset 0 0 0 1px var(--line);
}

.legend {
  display: grid;
  gap: 10px;
  flex: 1;
}

.legend-row {
  display: grid;
  grid-template-columns: 14px 1fr auto;
  align-items: center;
  gap: 8px;
  font-size: 14px;
}

.legend-dot {
  width: 12px;
  height: 12px;
  border-radius: 999px;
}

.col-chart {
  height: 260px;
  display: flex;
  align-items: flex-end;
  gap: 28px;
  padding: 18px 8px 8px;
  border-bottom: 1px solid var(--line);
}

.col-item {
  flex: 1;
  min-width: 0;
  height: 220px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
}

.col-bar {
  width: 58%;
  min-height: 8px;
  border-radius: 12px 12px 0 0;
  background: linear-gradient(180deg, var(--blue2), var(--blue));
}

.col-value {
  color: var(--muted);
  font-size: 13px;
  font-variant-numeric: tabular-nums;
}

.col-label {
  width: 100%;
  text-align: center;
  color: var(--subtext);
  font-size: 13px;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.mini-bars {
  display: grid;
  gap: 12px;
}

.mini-bar-row {
  display: grid;
  grid-template-columns: 120px 1fr 54px;
  align-items: center;
  gap: 10px;
  font-size: 14px;
}

.mini-bar-label {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.mini-bar-track {
  height: 10px;
  background: #edf1f7;
  border-radius: 999px;
  overflow: hidden;
}

.mini-bar-fill {
  height: 100%;
  background: var(--blue);
  border-radius: 999px;
}

.mini-bar-value {
  text-align: right;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}

.heatmap {
  display: grid;
  gap: 10px;
}

.heat-row {
  display: grid;
  grid-template-columns: 112px repeat(3, 1fr);
  gap: 8px;
  align-items: stretch;
}

.heat-head {
  color: var(--muted);
  font-size: 13px;
  font-weight: 650;
  padding: 8px 0;
}

.heat-label {
  font-weight: 650;
  color: var(--subtext);
  display: flex;
  align-items: center;
  font-size: 14px;
}

.heat-cell {
  border-radius: 12px;
  padding: 10px 8px;
  min-height: 58px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  color: #101828;
}

.heat-cell strong {
  font-size: 18px;
  line-height: 1;
}

.heat-cell span {
  margin-top: 5px;
  color: rgba(16,24,40,.72);
  font-size: 12px;
}

.keyword-cloud {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
}

.keyword-cloud-item {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  background: var(--blue-soft);
  color: #263f92;
  padding: 8px 13px;
  font-weight: 700;
}

.topic-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 8px;
}

.weight {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 42px;
  height: 28px;
  padding: 0 10px;
  background: var(--blue-soft);
  color: var(--blue);
  border-radius: 999px;
  font-weight: 760;
  font-size: 13px;
}

.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 8px 0 10px;
}

.chip {
  background: #f0f3f9;
  color: #33415c;
  border-radius: 999px;
  padding: 4px 9px;
  font-size: 12px;
}

.meta-line {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  color: var(--muted);
  font-size: 13px;
  margin-bottom: 10px;
}

.supporting-data {
  margin-top: 12px;
  padding: 10px 12px;
  background: var(--blue-soft);
  border-radius: 12px;
  color: #243b7a;
  font-size: 13px;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 14px;
  font-size: 14px;
}

th, td {
  border-bottom: 1px solid var(--line);
  text-align: left;
  padding: 12px 10px;
  vertical-align: top;
}

th {
  color: var(--muted);
  font-weight: 680;
  background: #fafbfe;
}

.risk-box {
  border: 1px solid #ffd8d4;
  background: #fff2f0;
  border-radius: 18px;
  padding: 18px;
}

.risk-level {
  color: #b42318;
  font-size: 18px;
  font-weight: 780;
  margin-bottom: 8px;
}

.two-col {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
  margin-top: 10px;
}

.two-col h3 {
  margin: 0 0 8px;
  font-size: 15px;
}

ul {
  margin: 0;
  padding-left: 20px;
}

#samples .sample-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.sample-card {
  min-width: 0;
}

.sample-meta {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 8px;
}

.sample-card p {
  font-size: 14px;
  word-break: break-word;
  overflow-wrap: anywhere;
}

.sample-link {
  display: inline-block;
  margin-top: 10px;
  color: var(--blue);
  text-decoration: none;
  font-size: 13px;
  font-weight: 680;
}

.empty,
.muted {
  color: var(--muted);
}

@media (max-width: 1000px) {
  .page {
    padding: 18px 14px 40px;
  }

  .hero {
    padding: 28px 24px;
  }

  .hero h1 {
    font-size: 30px;
  }

  .toc-grid,
  .kpi-grid,
  .grid-2,
  .topic-grid,
  .finding-grid,
  .sample-grid,
  .report-text-grid,
  .two-col {
    grid-template-columns: 1fr;
  }

  .donut-wrap {
    flex-direction: column;
    align-items: flex-start;
  }

  .heat-row {
    grid-template-columns: 88px repeat(3, 1fr);
  }
}
"""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def esc(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def clean_text(value: Any, remove_url: bool = False) -> str:
    if value is None:
        return ""
    text = str(value)
    text = html.unescape(text)
    if remove_url:
        text = URL_RE.sub("", text)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return " ".join(text.split()).strip()


def safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def zh_platform(value: Any) -> str:
    key = clean_text(value)
    return PLATFORM_ZH.get(key, key or "未知平台")


def zh_sentiment(value: Any) -> str:
    key = clean_text(value)
    return SENTIMENT_ZH.get(key, key or "未知")


def zh_risk(value: Any) -> str:
    key = clean_text(value)
    return RISK_ZH.get(key, key or "未知")


def pct(part: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{part / total * 100:.1f}%"


def render_kpi_card(title: str, value: Any, sub: str = "") -> str:
    return f"""
    <div class="kpi-card">
      <div class="kpi-title">{esc(title)}</div>
      <div class="kpi-value">{esc(value)}</div>
      <div class="kpi-sub">{esc(sub)}</div>
    </div>
    """


def render_donut(sentiment_counts: dict[str, Any]) -> str:
    neg = safe_int(sentiment_counts.get("negative"))
    neu = safe_int(sentiment_counts.get("neutral"))
    pos = safe_int(sentiment_counts.get("positive"))
    total = max(1, neg + neu + pos)

    neg_deg = neg / total * 360
    neu_deg = neu / total * 360
    pos_deg = pos / total * 360

    a = neg_deg
    b = neg_deg + neu_deg
    c = neg_deg + neu_deg + pos_deg

    bg = (
        f"conic-gradient(#df5b4f 0deg {a:.2f}deg, "
        f"#95a3b8 {a:.2f}deg {b:.2f}deg, "
        f"#34a576 {b:.2f}deg {c:.2f}deg)"
    )

    return f"""
    <div class="donut-wrap">
      <div class="donut" style="background:{esc(bg)}"></div>
      <div class="legend">
        <div class="legend-row">
          <span class="legend-dot" style="background:#df5b4f"></span>
          <span>负面</span>
          <strong>{pct(neg, total)}</strong>
        </div>
        <div class="legend-row">
          <span class="legend-dot" style="background:#95a3b8"></span>
          <span>中性</span>
          <strong>{pct(neu, total)}</strong>
        </div>
        <div class="legend-row">
          <span class="legend-dot" style="background:#34a576"></span>
          <span>正面</span>
          <strong>{pct(pos, total)}</strong>
        </div>
      </div>
    </div>
    """


def render_column_chart(data: dict[str, Any], label_func=None) -> str:
    if not data:
        return '<div class="empty">暂无数据</div>'

    label_func = label_func or (lambda x: x)
    max_value = max([safe_float(v) for v in data.values()] + [1.0])
    max_bar_height = 170
    items = []

    for key, value in data.items():
        v = safe_float(value)
        height_px = max(8, v / max_value * max_bar_height)

        items.append(f"""
        <div class="col-item">
          <div class="col-value">{esc(value)}</div>
          <div class="col-bar" style="height:{height_px:.0f}px"></div>
          <div class="col-label">{esc(label_func(key))}</div>
        </div>
        """)

    return f'<div class="col-chart">{"".join(items)}</div>'


def render_mini_bars(data: dict[str, Any], label_func=None) -> str:
    if not data:
        return '<div class="empty">暂无数据</div>'

    label_func = label_func or (lambda x: x)
    max_value = max([safe_float(v) for v in data.values()] + [1.0])
    rows = []

    for key, value in data.items():
        v = safe_float(value)
        width = max(3, min(100, v / max_value * 100))

        rows.append(f"""
        <div class="mini-bar-row">
          <div class="mini-bar-label">{esc(label_func(key))}</div>
          <div class="mini-bar-track">
            <div class="mini-bar-fill" style="width:{width:.2f}%"></div>
          </div>
          <div class="mini-bar-value">{esc(value)}</div>
        </div>
        """)

    return f'<div class="mini-bars">{"".join(rows)}</div>'


def heat_color(sentiment: str, value: int, max_value: int) -> str:
    ratio = 0 if max_value <= 0 else value / max_value
    alpha = 0.12 + 0.78 * ratio

    if sentiment == "negative":
        return f"rgba(223, 91, 79, {alpha:.2f})"
    if sentiment == "positive":
        return f"rgba(52, 165, 118, {alpha:.2f})"
    return f"rgba(149, 163, 184, {alpha:.2f})"


def render_platform_heatmap(data: dict[str, Any]) -> str:
    if not data:
        return '<div class="empty">暂无数据</div>'

    sentiments = ["negative", "neutral", "positive"]
    all_values = []

    for counts in data.values():
        counts = safe_dict(counts)
        for sentiment in sentiments:
            all_values.append(safe_int(counts.get(sentiment)))

    max_value = max(all_values + [1])

    rows = ["""
    <div class="heat-row">
      <div></div>
      <div class="heat-head">负面</div>
      <div class="heat-head">中性</div>
      <div class="heat-head">正面</div>
    </div>
    """]

    for key, counts in data.items():
        counts = safe_dict(counts)
        row_total = sum(safe_int(counts.get(s)) for s in sentiments)
        cells = []

        for sentiment in sentiments:
            value = safe_int(counts.get(sentiment))
            bg = heat_color(sentiment, value, max_value)
            cells.append(f"""
            <div class="heat-cell" style="background:{bg}">
              <strong>{value}</strong>
              <span>{pct(value, row_total)}</span>
            </div>
            """)

        rows.append(f"""
        <div class="heat-row">
          <div class="heat-label">{esc(zh_platform(key))}</div>
          {''.join(cells)}
        </div>
        """)

    return f'<div class="heatmap">{"".join(rows)}</div>'


def render_keyword_cloud(groups: list[Any]) -> str:
    valid = [item for item in groups if isinstance(item, dict)]

    if not valid:
        return '<div class="empty">暂无关键词组。</div>'

    max_weight = max([safe_float(item.get("weight")) for item in valid] + [1.0])
    chips = []

    for item in valid:
        name = clean_text(item.get("name"))
        weight = safe_float(item.get("weight"))
        size = 14 + 14 * weight / max_weight
        opacity = 0.72 + 0.28 * weight / max_weight
        chips.append(
            f'<span class="keyword-cloud-item" style="font-size:{size:.1f}px;opacity:{opacity:.2f}">{esc(name)}</span>'
        )

    return f'<div class="keyword-cloud">{"".join(chips)}</div>'


def render_summary(summary_items: list[Any]) -> str:
    items = [clean_text(item) for item in summary_items if clean_text(item)]

    if not items:
        return '<p class="muted">暂无摘要。</p>'

    return '<div class="summary-text">' + "\n".join(f"<p>{esc(item)}</p>" for item in items) + "</div>"


def render_report_sections(report_sections: dict[str, Any]) -> str:
    mapping = [
        ("event_overview", "事件背景"),
        ("public_opinion_focus", "舆论焦点"),
        ("emotion_structure", "情绪结构"),
        ("platform_spread", "平台传播"),
        ("risk_and_observation", "风险观察"),
    ]

    panels = []

    for key, title in mapping:
        text = clean_text(report_sections.get(key))
        if not text:
            continue

        panels.append(f"""
        <div class="text-panel">
          <h3>{esc(title)}</h3>
          <p>{esc(text)}</p>
        </div>
        """)

    if not panels:
        return '<div class="empty">暂无报告正文。</div>'

    return f'<div class="report-text-grid">{"".join(panels)}</div>'


def render_core_findings(findings: list[Any]) -> str:
    cards = []

    for item in findings:
        if not isinstance(item, dict):
            continue

        title = clean_text(item.get("title"))
        desc = clean_text(item.get("description"))
        data = clean_text(item.get("supporting_data"))

        cards.append(f"""
        <div class="finding-card">
          <h3>{esc(title)}</h3>
          <p>{esc(desc)}</p>
          <div class="supporting-data">{esc(data)}</div>
        </div>
        """)

    if not cards:
        return '<div class="empty">暂无核心发现。</div>'

    return "\n".join(cards)


def render_keyword_groups(groups: list[Any]) -> str:
    cards = []

    for item in groups:
        if not isinstance(item, dict):
            continue

        name = clean_text(item.get("name"))
        keywords = [clean_text(x) for x in safe_list(item.get("keywords")) if clean_text(x)]
        desc = clean_text(item.get("description"))
        weight = safe_int(item.get("weight"))

        chips = "".join(f'<span class="chip">{esc(x)}</span>' for x in keywords)

        cards.append(f"""
        <div class="topic-card">
          <div class="topic-head">
            <h3>{esc(name)}</h3>
            <span class="weight">{weight}</span>
          </div>
          <div class="chips">{chips}</div>
          <p>{esc(desc)}</p>
        </div>
        """)

    if not cards:
        return '<div class="empty">暂无关键词组。</div>'

    return "\n".join(cards)


def render_topic_clusters(clusters: list[Any]) -> str:
    cards = []

    for item in clusters:
        if not isinstance(item, dict):
            continue

        name = clean_text(item.get("name"))
        summary = clean_text(item.get("summary"))
        sentiment = zh_sentiment(item.get("dominant_sentiment"))
        platforms = [zh_platform(x) for x in safe_list(item.get("platforms")) if clean_text(x)]
        weight = safe_int(item.get("weight"))

        platform_text = " / ".join(platforms)

        cards.append(f"""
        <div class="topic-card">
          <div class="topic-head">
            <h3>{esc(name)}</h3>
            <span class="weight">{weight}</span>
          </div>
          <div class="meta-line">
            <span>情绪：{esc(sentiment)}</span>
            <span>平台：{esc(platform_text)}</span>
          </div>
          <p>{esc(summary)}</p>
        </div>
        """)

    if not cards:
        return '<div class="empty">暂无主题簇。</div>'

    return "\n".join(cards)


def render_platform_comparison(platform_comparison: dict[str, Any]) -> str:
    summary = clean_text(platform_comparison.get("summary"))
    items = safe_list(platform_comparison.get("items"))

    rows = []

    for item in items:
        if not isinstance(item, dict):
            continue

        rows.append(f"""
        <tr>
          <td>{esc(zh_platform(item.get("platform")))}</td>
          <td>{esc(item.get("observation"))}</td>
          <td>{esc(item.get("sentiment_pattern"))}</td>
        </tr>
        """)

    if not rows:
        table = '<div class="empty">暂无平台差异数据。</div>'
    else:
        table = f"""
        <table>
          <thead>
            <tr>
              <th>平台</th>
              <th>讨论特点</th>
              <th>情绪特点</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
        """

    return f"""
    <p>{esc(summary)}</p>
    {table}
    """


def render_sample_cards(items: list[Any], max_items: int = 6) -> str:
    cards = []
    seen = set()
    cleaned_items = []

    for item in items:
        if not isinstance(item, dict):
            continue

        raw_text = clean_text(item.get("text"), remove_url=True)
        raw_title = clean_text(item.get("title"), remove_url=True)
        url = clean_text(item.get("url"))

        key = (
            clean_text(item.get("platform")),
            url,
            raw_title[:80],
            raw_text[:120],
        )

        if key in seen:
            continue

        seen.add(key)

        if not raw_text and not raw_title:
            continue

        cleaned_items.append(item)

    for display_index, item in enumerate(cleaned_items[:max_items], start=1):
        text = clean_text(item.get("text"), remove_url=True)
        if len(text) > 180:
            text = text[:180].rstrip() + "……"

        platform = zh_platform(item.get("platform"))
        sentiment_level = clean_text(item.get("sentiment_level"))
        score = item.get("sentiment_score")
        engagement = item.get("engagement_score")
        url = clean_text(item.get("url"))

        link = ""
        if url:
            link = f'<a class="sample-link" href="{esc(url)}" target="_blank" rel="noreferrer">查看来源</a>'

        cards.append(f"""
        <div class="sample-card">
          <div class="sample-meta">
            <span>{esc(platform)}</span>
            <span>{esc(sentiment_level)} · {esc(score)} · 热度 {esc(engagement)}</span>
          </div>
          <h3>代表样本 {display_index}</h3>
          <p>{esc(text)}</p>
          {link}
        </div>
        """)

    if not cards:
        return '<div class="empty">暂无代表性样本。</div>'

    return "\n".join(cards)


def render_risk_block(risk: dict[str, Any]) -> str:
    risk_level = clean_text(risk.get("risk_level"))
    risk_summary = clean_text(risk.get("risk_summary"))
    signals = [clean_text(x) for x in safe_list(risk.get("risk_signals")) if clean_text(x)]
    uncertainties = [clean_text(x) for x in safe_list(risk.get("uncertainties")) if clean_text(x)]

    signal_html = "".join(f"<li>{esc(x)}</li>" for x in signals)
    uncertainty_html = "".join(f"<li>{esc(x)}</li>" for x in uncertainties)

    return f"""
    <div class="risk-box">
      <div class="risk-level">风险等级：{esc(zh_risk(risk_level))}</div>
      <p>{esc(risk_summary)}</p>
      <div class="two-col">
        <div>
          <h3>风险信号</h3>
          <ul>{signal_html}</ul>
        </div>
        <div>
          <h3>待观察点</h3>
          <ul>{uncertainty_html}</ul>
        </div>
      </div>
    </div>
    """


def render_toc() -> tuple[str, str]:
    items = [
        ("01", "事件概览", "摘要与正文概览", "#overview"),
        ("02", "核心指标", "样本规模与情绪概况", "#kpis"),
        ("03", "图表总览", "饼图、柱状图、关键词云", "#charts"),
        ("04", "平台情绪", "不同平台情绪对比", "#platform-emotion"),
        ("05", "舆论焦点", "关键词组与焦点解释", "#focus"),
        ("06", "主题簇", "主要讨论主题", "#topics"),
        ("07", "核心发现", "综合归纳结论", "#findings"),
        ("08", "平台差异", "不同平台讨论特征", "#platforms"),
        ("09", "风险观察", "风险信号与待观察点", "#risk"),
        ("10", "样本摘录", "高热与高负面内容", "#samples"),
    ]

    cards = []
    nav_links = []

    for num, title, desc, href in items:
        cards.append(f"""
        <a class="toc-card" href="{href}">
          <div class="toc-num">{esc(num)}</div>
          <div class="toc-title">{esc(title)}</div>
          <div class="toc-desc">{esc(desc)}</div>
        </a>
        """)
        nav_links.append(f'<a href="{href}">{esc(title)}</a>')

    return "\n".join(nav_links), "\n".join(cards)


def render_html(context: dict[str, Any], insights_doc: dict[str, Any]) -> str:
    insights = safe_dict(insights_doc.get("insights"))

    event_name = clean_text(context.get("event_name")) or clean_text(insights.get("event_name")) or "舆情事件"
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    data_overview = safe_dict(context.get("data_overview"))
    chart_data = safe_dict(context.get("chart_data"))
    sample_sets = safe_dict(context.get("sample_sets"))

    sentiment_counts = safe_dict(chart_data.get("sentiment_counts"))
    sentiment_level_counts = safe_dict(chart_data.get("sentiment_level_counts"))
    platform_counts = safe_dict(chart_data.get("platform_counts"))
    by_platform_sentiment = safe_dict(chart_data.get("by_platform_sentiment"))

    platform_source_text = " / ".join(
        zh_platform(platform)
        for platform, count in platform_counts.items()
        if safe_int(count) > 0
    ) or "未知"

    total = safe_int(data_overview.get("analyzed_text_count"))
    negative_count = safe_int(sentiment_counts.get("negative"))
    positive_count = safe_int(sentiment_counts.get("positive"))
    neutral_count = safe_int(sentiment_counts.get("neutral"))
    avg_score = safe_float(data_overview.get("average_sentiment_score"))

    executive_summary = safe_list(insights.get("executive_summary"))
    report_sections = safe_dict(insights.get("report_sections"))
    keyword_groups = safe_list(insights.get("keyword_groups"))
    topic_clusters = safe_list(insights.get("topic_clusters"))
    core_findings = safe_list(insights.get("core_findings"))
    platform_comparison = safe_dict(insights.get("platform_comparison"))
    risk_interpretation = safe_dict(insights.get("risk_interpretation"))
    insight_chart_data = safe_dict(insights.get("chart_data"))

    topic_chart = safe_list(insight_chart_data.get("topic_cluster_chart"))

    top_hot_items = safe_list(sample_sets.get("top_hot_items"))
    top_negative_items = safe_list(sample_sets.get("top_negative_items"))
    representative_items = safe_list(sample_sets.get("representative_samples"))

    nav_links, toc_cards = render_toc()

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(event_name)} - 舆情分析报告</title>
  <style>
{CSS}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="hero-label">舆情分析报告</div>
      <h1>{esc(event_name)}</h1>
      <div class="hero-sub">生成时间：{esc(generated_at)} · 样本文本数：{esc(total)} · 数据来源：{esc(platform_source_text)}</div>
    </section>

    <nav class="navbar">
      {nav_links}
    </nav>

    <section class="section" id="catalog">
      <h2>报告目录</h2>
      <div class="toc-grid">
        {toc_cards}
      </div>
    </section>

    <section class="section" id="overview">
      <h2>事件概览</h2>
      {render_summary(executive_summary)}
      <div style="height:12px"></div>
      {render_report_sections(report_sections)}
    </section>

    <section class="section" id="kpis">
      <h2>核心指标</h2>
      <div class="kpi-grid">
        {render_kpi_card("分析文本数", total, f"低价值短文本 {data_overview.get('low_value_count', 0)} 条")}
        {render_kpi_card("负面文本", negative_count, pct(negative_count, total))}
        {render_kpi_card("中性文本", neutral_count, pct(neutral_count, total))}
        {render_kpi_card("正面文本", positive_count, pct(positive_count, total))}
        {render_kpi_card("平均情感分数", f"{avg_score:.4f}", "越低表示样本语气越负面")}
      </div>
    </section>

    <section class="section" id="charts">
      <h2>图表总览</h2>
      <p class="section-desc">本部分只按平台维度展示样本结构，不再细分水源楼层、知乎评论等来源类型。</p>
      <div class="grid-2">
        <div class="chart-card">
          <h3>情感占比</h3>
          {render_donut(sentiment_counts)}
        </div>
        <div class="chart-card">
          <h3>平台样本量柱状图</h3>
          {render_column_chart(platform_counts, zh_platform)}
        </div>
        <div class="chart-card">
          <h3>五级情绪分布</h3>
          {render_column_chart(sentiment_level_counts)}
        </div>
        <div class="chart-card">
          <h3>舆论焦点关键词云</h3>
          {render_keyword_cloud(keyword_groups)}
        </div>
      </div>
    </section>

    <section class="section" id="platform-emotion">
      <h2>平台情绪结构</h2>
      <p class="section-desc">颜色越深表示该平台中的对应情绪文本越多。</p>
      <div class="grid-2">
        <div class="chart-card">
          <h3>平台情绪热力图</h3>
          {render_platform_heatmap(by_platform_sentiment)}
        </div>
        <div class="chart-card">
          <h3>主题簇权重</h3>
          {render_mini_bars({item.get("name"): item.get("value") for item in topic_chart if isinstance(item, dict)})}
        </div>
      </div>
    </section>

    <section class="section" id="focus">
      <h2>关键词组与舆论焦点</h2>
      <p class="section-desc">以下关键词组用于呈现主要讨论方向，权重表示相对关注度。</p>
      <div class="topic-grid">
        {render_keyword_groups(keyword_groups)}
      </div>
    </section>

    <section class="section" id="topics">
      <h2>主要讨论主题</h2>
      <div class="topic-grid">
        {render_topic_clusters(topic_clusters)}
      </div>
    </section>

    <section class="section" id="findings">
      <h2>核心发现</h2>
      <div class="finding-grid">
        {render_core_findings(core_findings)}
      </div>
    </section>

    <section class="section" id="platforms">
      <h2>平台差异</h2>
      {render_platform_comparison(platform_comparison)}
    </section>

    <section class="section" id="risk">
      <h2>风险观察</h2>
      {render_risk_block(risk_interpretation)}
    </section>

    <section class="section" id="samples">
      <h2>代表性样本摘录</h2>
      <p class="section-desc">以下内容用于展示样本中具有代表性的讨论表达，已做去重和截断处理。</p>
      <div class="sample-grid">
        {render_sample_cards(representative_items if representative_items else (top_hot_items + top_negative_items), max_items=6)}
      </div>
    </section>
  </main>
</body>
</html>
"""

    return html_text


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="openopinion-render-final-report",
        description="Render final HTML report from report_context.json and report_insights.json.",
    )
    parser.add_argument("--context", required=True, help="report_context.json 路径")
    parser.add_argument("--insights", required=True, help="report_insights.json 路径")
    parser.add_argument("--out", required=True, help="输出 HTML 路径")

    args = parser.parse_args()

    context = load_json(Path(args.context))
    insights = load_json(Path(args.insights))

    html_text = render_html(context, insights)

    output_path = Path(args.out)
    write_text(output_path, html_text)

    print("最终 HTML 报告生成完成：")
    print(output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
