from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, fontManager

from opinion_agent.schemas import SessionLog


PAGE_WIDTH_IN = 8.27
PAGE_HEIGHT_IN = 11.69
MARGIN_LEFT_IN = 0.72
MARGIN_RIGHT_IN = 0.72
MARGIN_TOP_IN = 0.78
MARGIN_BOTTOM_IN = 0.72
HEADER_GAP_IN = 0.2
INK = "#1d2433"
MUTED = "#667085"
ACCENT = "#1264a3"
LINE = "#d7dde8"
PREFERRED_FONT_FAMILIES = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
FONT_FAMILIES = [name for name in PREFERRED_FONT_FAMILIES if name in {font.name for font in fontManager.ttflist}]
if not FONT_FAMILIES:
    FONT_FAMILIES = ["DejaVu Sans"]


@dataclass(frozen=True)
class _MarkdownBlock:
    kind: str
    text: str = ""
    level: int = 0


class PdfReportGenerator:
    def generate(self, session: SessionLog, markdown_report: str, pdf_path: Path) -> str:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        _MarkdownPdfRenderer(session, markdown_report).render(pdf_path)
        return str(pdf_path)


class _MarkdownPdfRenderer:
    def __init__(self, session: SessionLog, markdown_report: str) -> None:
        self.session = session
        self.blocks = _parse_markdown(markdown_report)
        self.page_number = 0
        self.fig: plt.Figure | None = None
        self.y = 0.0
        self.normal = FontProperties(family=FONT_FAMILIES, weight="normal")
        self.bold = FontProperties(family=FONT_FAMILIES, weight="bold")

    def render(self, pdf_path: Path) -> None:
        plt.rcParams["font.sans-serif"] = FONT_FAMILIES
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42
        with PdfPages(pdf_path) as pdf:
            metadata = pdf.infodict()
            metadata["Title"] = f"OpenOpinion - {self.session.user_input}"
            metadata["Author"] = "OpenOpinion"
            self._start_page()
            self._draw_title(pdf)
            for block in self.blocks:
                self._draw_block(block, pdf)
            self._finish_page(pdf)

    def _start_page(self) -> None:
        self.page_number += 1
        self.fig = plt.figure(figsize=(PAGE_WIDTH_IN, PAGE_HEIGHT_IN), dpi=144)
        self.fig.patch.set_facecolor("white")
        self.y = 1 - (MARGIN_TOP_IN / PAGE_HEIGHT_IN)
        self._draw_page_header()

    def _finish_page(self, pdf: PdfPages) -> None:
        if self.fig is None:
            return
        footer_y = MARGIN_BOTTOM_IN * 0.45 / PAGE_HEIGHT_IN
        self.fig.text(
            0.5,
            footer_y,
            f"Page {self.page_number}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=MUTED,
            fontproperties=self.normal,
        )
        pdf.savefig(self.fig)
        plt.close(self.fig)
        self.fig = None

    def _ensure_space(self, needed_in: float, pdf: PdfPages) -> None:
        bottom = MARGIN_BOTTOM_IN / PAGE_HEIGHT_IN
        if self.y - (needed_in / PAGE_HEIGHT_IN) >= bottom:
            return
        self._finish_page(pdf)
        self._start_page()

    def _draw_page_header(self) -> None:
        if self.fig is None:
            return
        x_left = MARGIN_LEFT_IN / PAGE_WIDTH_IN
        x_right = 1 - (MARGIN_RIGHT_IN / PAGE_WIDTH_IN)
        header_y = 1 - (0.38 / PAGE_HEIGHT_IN)
        self.fig.text(
            x_left,
            header_y,
            "OpenOpinion",
            ha="left",
            va="top",
            fontsize=9,
            color=MUTED,
            fontproperties=self.bold,
        )
        self.fig.text(
            x_right,
            header_y,
            self.session.session_id,
            ha="right",
            va="top",
            fontsize=8,
            color=MUTED,
            fontproperties=self.normal,
        )
        line_y = 1 - ((0.58 + HEADER_GAP_IN) / PAGE_HEIGHT_IN)
        self.fig.add_artist(
            plt.Line2D(
                [x_left, x_right],
                [line_y, line_y],
                transform=self.fig.transFigure,
                color=LINE,
                linewidth=0.8,
            )
        )

    def _draw_title(self, pdf: PdfPages) -> None:
        if self.fig is None:
            return
        title = _plain_text(self.session.user_input)
        subtitle = f"Session {self.session.session_id} | Generated {self.session.created_at}"
        self._draw_wrapped_text(title, font_size=17, color=INK, weight="bold", top_gap_in=0.08, after_gap_in=0.08, pdf=pdf)
        self._draw_wrapped_text(subtitle, font_size=8.5, color=MUTED, top_gap_in=0, after_gap_in=0.2, pdf=pdf)

    def _draw_block(self, block: _MarkdownBlock, pdf: PdfPages) -> None:
        if block.kind == "spacer":
            self.y -= 0.08 / PAGE_HEIGHT_IN
            return
        if block.kind == "heading":
            size = {1: 16.0, 2: 13.8, 3: 12.2}.get(block.level, 11.4)
            top_gap = 0.22 if block.level <= 2 else 0.15
            self._draw_wrapped_text(
                block.text,
                font_size=size,
                color=ACCENT if block.level <= 2 else INK,
                weight="bold",
                top_gap_in=top_gap,
                after_gap_in=0.1,
                pdf=pdf,
            )
            return
        if block.kind == "bullet":
            self._draw_wrapped_text(
                f"- {block.text}",
                font_size=10.2,
                color=INK,
                left_indent_in=0.18,
                hanging_indent_in=0.16,
                after_gap_in=0.05,
                pdf=pdf,
            )
            return
        self._draw_wrapped_text(block.text, font_size=10.5, color=INK, after_gap_in=0.08, pdf=pdf)

    def _draw_wrapped_text(
        self,
        text: str,
        *,
        font_size: float,
        color: str,
        weight: str = "normal",
        top_gap_in: float = 0.0,
        after_gap_in: float = 0.0,
        left_indent_in: float = 0.0,
        hanging_indent_in: float = 0.0,
        pdf: PdfPages | None = None,
    ) -> None:
        text = _plain_text(text)
        if not text:
            return
        available_width = PAGE_WIDTH_IN - MARGIN_LEFT_IN - MARGIN_RIGHT_IN - left_indent_in
        max_units = max(18, int((available_width * 72) / (font_size * 0.55)))
        lines = _wrap_display(text, max_units)
        line_height_in = font_size * 1.42 / 72
        if pdf is not None and top_gap_in:
            self._ensure_space(top_gap_in + line_height_in, pdf)
        self.y -= top_gap_in / PAGE_HEIGHT_IN
        font = self.bold if weight == "bold" else self.normal
        for idx, line in enumerate(lines):
            if pdf is not None:
                self._ensure_space(line_height_in, pdf)
            if self.fig is None:
                return
            indent = left_indent_in + (hanging_indent_in if idx > 0 else 0.0)
            x = (MARGIN_LEFT_IN + indent) / PAGE_WIDTH_IN
            self.fig.text(
                x,
                self.y,
                line,
                ha="left",
                va="top",
                fontsize=font_size,
                color=color,
                fontproperties=font,
            )
            self.y -= line_height_in / PAGE_HEIGHT_IN
        self.y -= after_gap_in / PAGE_HEIGHT_IN


def _parse_markdown(markdown: str) -> list[_MarkdownBlock]:
    blocks: list[_MarkdownBlock] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(_MarkdownBlock(kind="paragraph", text=" ".join(paragraph).strip()))
            paragraph.clear()

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            if blocks and blocks[-1].kind != "spacer":
                blocks.append(_MarkdownBlock(kind="spacer"))
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            blocks.append(_MarkdownBlock(kind="heading", text=heading.group(2).strip(), level=len(heading.group(1))))
            continue
        bullet = re.match(r"^(?:[-*]|\d+[.)])\s+(.+)$", line)
        if bullet:
            flush_paragraph()
            blocks.append(_MarkdownBlock(kind="bullet", text=bullet.group(1).strip()))
            continue
        paragraph.append(line)

    flush_paragraph()
    return blocks or [_MarkdownBlock(kind="paragraph", text=markdown.strip())]


def _plain_text(text: str) -> str:
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _wrap_display(text: str, max_units: int) -> list[str]:
    lines: list[str] = []
    current: list[str] = []
    width = 0
    for char in text:
        if char.isspace():
            if current and current[-1] != " ":
                current.append(" ")
                width += 1
            continue
        char_width = _display_width(char)
        if current and width + char_width > max_units:
            lines.append("".join(current).rstrip())
            current = [char]
            width = char_width
        else:
            current.append(char)
            width += char_width
    if current:
        lines.append("".join(current).rstrip())
    return lines


def _display_width(char: str) -> int:
    if unicodedata.combining(char):
        return 0
    if unicodedata.east_asian_width(char) in {"F", "W"}:
        return 2
    return 1
