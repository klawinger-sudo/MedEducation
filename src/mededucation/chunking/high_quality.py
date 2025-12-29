"""High-quality chunking for medical textbooks.

Goals:
- Remove repeated headers/footers and other page noise
- Preserve structure (headings, lists, tables)
- Prefer splitting on semantic boundaries (headings/paragraphs) rather than raw size
- Use a real tokenizer when available for accurate token limits

This module intentionally stays dependency-light; it uses `transformers` only
if installed, otherwise falls back to a character heuristic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from mededucation.models.content import ChunkResult, ExtractedPage


_TABLE_START_RE = re.compile(r"^\[TABLE\s+\d+\]$", re.IGNORECASE)
_TABLE_END_RE = re.compile(r"^\[/TABLE\]$", re.IGNORECASE)


@dataclass
class HQChunkerConfig:
    """Configuration for the high-quality chunker."""

    # Token sizing - stay well under 512-token embedding limit
    target_tokens: int = 380
    max_tokens: int = 450
    min_tokens: int = 150
    overlap_tokens: int = 70

    # Tokenization
    tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chars_per_token_fallback: float = 4.0

    # Repeated header/footer removal
    header_lines: int = 2
    footer_lines: int = 2
    boilerplate_threshold: float = 0.55


@dataclass(frozen=True)
class _Segment:
    text: str
    page_number: int
    kind: str  # heading|para|list|table


class _TokenCounter:
    def __init__(self, tokenizer_model: str, chars_per_token_fallback: float):
        self._tokenizer_model = tokenizer_model
        self._chars_per_token_fallback = chars_per_token_fallback
        self._tokenizer = None

    def _ensure_tokenizer(self) -> None:
        if self._tokenizer is not None:
            return
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_model, use_fast=True)
        except Exception:
            self._tokenizer = None

    def count(self, text: str) -> int:
        if not text:
            return 0

        self._ensure_tokenizer()
        if self._tokenizer is None:
            return int(len(text) / self._chars_per_token_fallback)

        return len(self._tokenizer.encode(text, add_special_tokens=False))


class HighQualityChunker:
    """Structure-aware, tokenizer-sized chunking for textbook PDFs."""

    def __init__(self, config: Optional[HQChunkerConfig] = None):
        self.config = config or HQChunkerConfig()
        self._counter = _TokenCounter(
            tokenizer_model=self.config.tokenizer_model,
            chars_per_token_fallback=self.config.chars_per_token_fallback,
        )

    def chunk_pages(self, pages: Sequence[ExtractedPage], source_name: str) -> List[ChunkResult]:
        if not pages:
            return []

        cleaned_pages = self._remove_repeated_headers_footers(list(pages))
        segments = self._pages_to_segments(cleaned_pages)
        return self._segments_to_chunks(segments, pages=cleaned_pages, source_name=source_name)

    def _remove_repeated_headers_footers(self, pages: List[ExtractedPage]) -> List[ExtractedPage]:
        """Drop repeated header/footer lines that occur across many pages."""

        def normalize(line: str) -> str:
            line = re.sub(r"\d+", "", line)
            line = re.sub(r"\s+", " ", line.strip().lower())
            line = re.sub(r"[^a-z ]+", "", line)
            return line.strip()

        if not pages:
            return pages

        header_counts: dict[str, int] = {}
        footer_counts: dict[str, int] = {}

        for page in pages:
            lines = [ln.strip() for ln in page.text.split("\n") if ln.strip()]
            if not lines:
                continue

            head = lines[: self.config.header_lines]
            foot = lines[-self.config.footer_lines :] if self.config.footer_lines > 0 else []

            for ln in head:
                key = normalize(ln)
                if key:
                    header_counts[key] = header_counts.get(key, 0) + 1
            for ln in foot:
                key = normalize(ln)
                if key:
                    footer_counts[key] = footer_counts.get(key, 0) + 1

        threshold = max(2, int(len(pages) * self.config.boilerplate_threshold))
        boilerplate = {k for k, v in header_counts.items() if v >= threshold} | {
            k for k, v in footer_counts.items() if v >= threshold
        }

        if not boilerplate:
            return pages

        cleaned: List[ExtractedPage] = []
        for page in pages:
            raw_lines = page.text.split("\n")
            kept_lines: List[str] = []
            for ln in raw_lines:
                stripped = ln.strip()
                if not stripped:
                    kept_lines.append(ln)
                    continue
                if normalize(stripped) in boilerplate:
                    continue
                kept_lines.append(ln)

            cleaned.append(page.model_copy(update={"text": "\n".join(kept_lines).strip()}))

        return cleaned

    def _pages_to_segments(self, pages: Sequence[ExtractedPage]) -> List[_Segment]:
        segments: List[_Segment] = []
        for page in pages:
            page_segments = self._page_text_to_segments(page.text, page.page_number)
            segments.extend(page_segments)
        return segments

    def _page_text_to_segments(self, text: str, page_number: int) -> List[_Segment]:
        if not text.strip():
            return []

        lines = [ln.rstrip() for ln in text.split("\n")]
        blocks: List[tuple[str, str]] = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if _TABLE_START_RE.match(line):
                table_lines = [lines[i]]
                i += 1
                while i < len(lines):
                    table_lines.append(lines[i])
                    if _TABLE_END_RE.match(lines[i].strip()):
                        break
                    i += 1
                blocks.append(("table", "\n".join(table_lines).strip()))
                i += 1
                continue

            chunk_lines: List[str] = []
            while i < len(lines) and lines[i].strip():
                if _TABLE_START_RE.match(lines[i].strip()):
                    break
                chunk_lines.append(lines[i])
                i += 1

            if chunk_lines:
                raw_block = "\n".join(chunk_lines).strip()
                blocks.extend(self._split_block_into_kinds(raw_block))

            while i < len(lines) and not lines[i].strip():
                i += 1

        segments: List[_Segment] = []
        for kind, block_text in blocks:
            block_text = self._unwrap_pdf_lines(block_text) if kind != "table" else block_text
            if not block_text.strip():
                continue

            if kind == "para" and self._is_heading(block_text):
                segments.append(_Segment(text=block_text.strip(), page_number=page_number, kind="heading"))
            else:
                segments.append(_Segment(text=block_text.strip(), page_number=page_number, kind=kind))

        return segments

    def _split_block_into_kinds(self, block_text: str) -> List[tuple[str, str]]:
        """Split a non-table block into paragraph/list sub-blocks."""
        lines = [ln.strip() for ln in block_text.split("\n") if ln.strip()]
        if not lines:
            return []

        bullet_re = re.compile(r"^(?:[-*•]|\d+\.|\([a-zA-Z0-9]+\))\s+\S+")
        if all(bullet_re.match(ln) for ln in lines) and len(lines) >= 2:
            return [("list", "\n".join(lines))]

        return [("para", "\n".join(lines))]

    def _unwrap_pdf_lines(self, block_text: str) -> str:
        """Remove hard line wraps while preserving true paragraph/list breaks."""
        lines = [ln.rstrip() for ln in block_text.split("\n")]
        if len(lines) <= 1:
            return block_text.strip()

        joined: List[str] = []
        current = lines[0].strip()
        for nxt in lines[1:]:
            nxt_s = nxt.strip()
            if not nxt_s:
                if current:
                    joined.append(current)
                    current = ""
                continue

            if not current:
                current = nxt_s
                continue

            if current.endswith("-") and nxt_s and nxt_s[0].islower():
                current = current[:-1] + nxt_s
                continue

            if not re.search(r"[.!?:;]$", current) and nxt_s and nxt_s[0].islower():
                current = current + " " + nxt_s
                continue

            if re.search(r"[,;]$", current):
                current = current + " " + nxt_s
                continue

            joined.append(current)
            current = nxt_s

        if current:
            joined.append(current)

        return "\n".join(joined).strip()

    def _is_heading(self, text: str) -> bool:
        t = text.strip()
        if not t:
            return False

        if _TABLE_START_RE.match(t) or _TABLE_END_RE.match(t):
            return False

        if re.match(r"^(?:chapter|section|module)\s+\d+\b", t, flags=re.IGNORECASE):
            return True

        if re.match(r"^\d+(?:\.\d+){0,3}\s+\S+", t) and len(t.split()) <= 12 and len(t) <= 120:
            return True

        letters = re.sub(r"[^A-Za-z]", "", t)
        if letters and letters.isupper() and len(t.split()) <= 10 and len(t) <= 80:
            return True

        if len(t) <= 90 and not t.endswith(".") and len(t.split()) <= 12:
            words = [w for w in re.split(r"\s+", t) if w]
            if words:
                cap = sum(1 for w in words if w[:1].isupper())
                if cap / max(1, len(words)) >= 0.7:
                    return True

        return False

    def _segments_to_chunks(
        self, segments: Sequence[_Segment], pages: Sequence[ExtractedPage], source_name: str
    ) -> List[ChunkResult]:
        if not segments:
            return []

        page_to_chapter = {p.page_number: p.chapter for p in pages}

        chunks: List[ChunkResult] = []
        current: List[_Segment] = []
        chunk_index = 0

        def current_text(seg_list: Sequence[_Segment]) -> str:
            return "\n\n".join(s.text for s in seg_list if s.text.strip())

        def segment_tokens(seg: _Segment) -> int:
            return self._counter.count(seg.text)

        def total_tokens(seg_list: Sequence[_Segment]) -> int:
            return self._counter.count(current_text(seg_list))

        def flush() -> None:
            nonlocal current, chunk_index
            if not current:
                return

            text = current_text(current).strip()
            if not text:
                current = []
                return

            tks = self._counter.count(text)
            if tks < self.config.min_tokens and chunks:
                prev = chunks[-1]
                merged_text = (prev.text + "\n\n" + text).strip()
                merged_tokens = self._counter.count(merged_text)
                if merged_tokens <= self.config.max_tokens:
                    chunks[-1] = prev.model_copy(
                        update={
                            "text": merged_text,
                            "end_page": max(prev.end_page, max(s.page_number for s in current)),
                            "token_estimate": merged_tokens,
                        }
                    )
                    current = []
                    return

            start_page = min(s.page_number for s in current)
            end_page = max(s.page_number for s in current)

            chapter_vals = [page_to_chapter.get(pn) for pn in range(start_page, end_page + 1)]
            chapter_vals = [c for c in chapter_vals if c]
            chapter = None
            if chapter_vals:
                chapter = max(set(chapter_vals), key=chapter_vals.count)

            chunk_id = f"hq_chunk{chunk_index:05d}"
            chunk_index += 1

            chunks.append(
                ChunkResult(
                    chunk_id=chunk_id,
                    text=text,
                    start_page=start_page,
                    end_page=end_page,
                    chapter=chapter,
                    section=None,
                    source_name=source_name,
                    token_estimate=tks,
                )
            )

            if self.config.overlap_tokens <= 0:
                current = []
                return

            overlap: List[_Segment] = []
            running = 0
            for seg in reversed(current):
                seg_t = segment_tokens(seg)
                if running + seg_t > self.config.overlap_tokens and overlap:
                    break
                overlap.append(seg)
                running += seg_t

            overlap.reverse()
            current = overlap

        for seg in segments:
            for subseg in self._split_oversize_segment(seg):
                if subseg.kind == "heading" and current:
                    if total_tokens(current) >= int(self.config.target_tokens * 0.6):
                        flush()

                prospective = current + [subseg]
                if total_tokens(prospective) > self.config.max_tokens and current:
                    flush()
                    current = [subseg]
                else:
                    current.append(subseg)

        flush()

        return chunks

    def _split_oversize_segment(self, seg: _Segment) -> List[_Segment]:
        """Split a single segment that exceeds max_tokens."""
        max_tokens = self.config.max_tokens
        if max_tokens <= 0:
            return [seg]

        if self._counter.count(seg.text) <= max_tokens:
            return [seg]

        text = seg.text.strip()
        if not text:
            return []

        if seg.kind == "table":
            parts = [ln for ln in text.splitlines() if ln.strip()]
            joiner = "\n"
        else:
            parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
            joiner = "\n\n"

        chunks: List[_Segment] = []
        buf: List[str] = []

        for part in parts:
            candidate = joiner.join(buf + [part]).strip() if buf else part
            if buf and self._counter.count(candidate) > max_tokens:
                chunks.append(_Segment(text=joiner.join(buf).strip(), page_number=seg.page_number, kind=seg.kind))
                buf = [part]
            else:
                buf.append(part)

        if buf:
            chunks.append(_Segment(text=joiner.join(buf).strip(), page_number=seg.page_number, kind=seg.kind))

        final: List[_Segment] = []
        for c in chunks:
            if self._counter.count(c.text) <= max_tokens:
                final.append(c)
                continue

            raw = c.text
            approx_tokens = max(1, self._counter.count(raw))
            safe_max = int(max_tokens * 0.9)
            window = max(400, int(len(raw) * (safe_max / approx_tokens)))
            for i in range(0, len(raw), window):
                piece = raw[i : i + window].strip()
                if piece:
                    final.append(_Segment(text=piece, page_number=seg.page_number, kind=seg.kind))

        return final
