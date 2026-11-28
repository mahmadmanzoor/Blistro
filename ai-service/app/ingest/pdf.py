"""PDF to page-aware text blocks.

PyMuPDF gives us text blocks with font sizes, so headings can be detected from
typography instead of guessing from line text alone."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

import pymupdf


@dataclass(frozen=True)
class Block:
    page: int  # 1-based
    text: str
    is_heading: bool = False


@dataclass(frozen=True)
class ParsedPdf:
    page_count: int
    blocks: list[Block]

    def page_text(self, page: int) -> str:
        return "\n".join(b.text for b in self.blocks if b.page == page)


_NUMBERED_HEADING = re.compile(r"^(\d+(\.\d+)*|[IVXLC]+\.|[A-Z]\.)\s+\S")
_WS = re.compile(r"[ \t]+")


def _clean(text: str) -> str:
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join hyphenated line breaks
    lines = [_WS.sub(" ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def parse_pdf(data: bytes) -> ParsedPdf:
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        raw: list[tuple[int, str, float, bool]] = []  # page, text, size, bold
        for page_index, page in enumerate(doc, start=1):
            page_dict = page.get_text("dict", sort=True)
            for blk in page_dict.get("blocks", []):
                if blk.get("type") != 0:  # text blocks only
                    continue
                spans = [s for ln in blk.get("lines", []) for s in ln.get("spans", [])]
                if not spans:
                    continue
                text = "\n".join(
                    "".join(s["text"] for s in ln.get("spans", [])) for ln in blk["lines"]
                )
                text = _clean(text)
                if not text:
                    continue
                size = max(s["size"] for s in spans)
                bold = all((s.get("flags", 0) & 16) for s in spans if s["text"].strip())
                raw.append((page_index, text, size, bold))

        sizes = [size for _, text, size, _ in raw if len(text) > 40]
        body = statistics.median(sizes) if sizes else 10.0

        blocks: list[Block] = []
        for page_no, text, size, bold in raw:
            short = len(text) <= 120 and "\n" not in text.strip()
            heading = short and (
                size >= body * 1.15
                or (bold and len(text) <= 80)
                or bool(
                    _NUMBERED_HEADING.match(text) and len(text) <= 80 and not text.endswith(".")
                )
            )
            blocks.append(Block(page=page_no, text=text, is_heading=heading))
        return ParsedPdf(page_count=doc.page_count, blocks=blocks)
    finally:
        doc.close()
