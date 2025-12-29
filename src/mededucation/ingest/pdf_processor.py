"""
PDF text extraction with page tracking and chapter detection.

This module handles extracting text from medical textbook PDFs while
preserving page numbers for accurate citations.

Usage:
    from mededucation.ingest import PDFProcessor

    processor = PDFProcessor("textbook.pdf")
    result = processor.extract()
    processor.save_json("output/extracted.json")
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Union

import fitz  # PyMuPDF

from mededucation.models.content import ExtractedChapter, ExtractedPage, PDFMetadata


# Common chapter heading patterns in medical textbooks
CHAPTER_PATTERNS = [
    re.compile(r"^\s*(?:CHAPTER|Chapter)\s+(\d+)\s*[:\-\.]?\s*(.+)?$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*(?:SECTION|Section)\s+(\d+)\s*[:\-\.]?\s*(.+)?$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*(?:MODULE|Module)\s+(\d+)\s*[:\-\.]?\s*(.+)?$", re.MULTILINE | re.IGNORECASE),
]

STRICT_CHAPTER_PATTERN = re.compile(r"^\s*(\d{1,2})\s+([A-Z][A-Z\s]{3,})$", re.MULTILINE)


class PDFProcessor:
    """
    Extracts text and metadata from PDF files.

    This class handles:
    - Text extraction with page number tracking
    - Chapter/section detection via heading patterns
    - Metadata extraction (title, author, page count)
    - Export to JSON for inspection and downstream processing

    Example:
        >>> processor = PDFProcessor("/path/to/textbook.pdf")
        >>> processor.extract()
        >>> print(f"Extracted {processor.metadata.total_pages} pages")
        >>> print(f"Found {len(processor.metadata.chapters)} chapters")
        >>> processor.save_json("output/textbook_extracted.json")
    """

    def __init__(self, pdf_path: Union[str, Path]):
        """
        Initialize the PDF processor.

        Args:
            pdf_path: Path to the PDF file to process.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            ValueError: If the path doesn't point to a PDF file.
        """
        self.pdf_path = Path(pdf_path)

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        if self.pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {self.pdf_path}")

        self.pages: List[ExtractedPage] = []
        self.metadata: Optional[PDFMetadata] = None
        self._doc: Optional[fitz.Document] = None

    def extract(self) -> PDFMetadata:
        """
        Extract all text and metadata from the PDF.

        Returns:
            PDFMetadata object containing extraction results.
        """
        self._doc = fitz.open(self.pdf_path)

        try:
            self.pages = list(self._extract_pages())
            chapters = self._detect_chapters()

            for chapter in chapters:
                start = chapter.start_page - 1
                end = chapter.end_page or len(self.pages)
                for i in range(start, end):
                    if i < len(self.pages):
                        self.pages[i].chapter = chapter.chapter_number

            self.metadata = PDFMetadata(
                filename=self.pdf_path.name,
                source_path=str(self.pdf_path.absolute()),
                title=self._doc.metadata.get("title") or None,
                author=self._doc.metadata.get("author") or None,
                total_pages=len(self.pages),
                total_chars=sum(p.char_count for p in self.pages),
                total_words=sum(p.word_count for p in self.pages),
                chapters=chapters,
                extracted_at=datetime.now(),
            )

            return self.metadata

        finally:
            self._doc.close()
            self._doc = None

    def _extract_pages(self) -> Iterator[ExtractedPage]:
        """Extract text from each page, preserving tables and lists."""
        if self._doc is None:
            raise RuntimeError("PDF document not opened")

        for page_num in range(len(self._doc)):
            page = self._doc[page_num]

            tables_text = ""
            try:
                tabs = page.find_tables()
                for i, tab in enumerate(tabs):
                    df = tab.to_pandas()
                    if not df.empty:
                        table_md = f"\n[TABLE {i+1}]\n"
                        table_md += df.to_markdown(index=False)
                        table_md += "\n[/TABLE]\n"
                        tables_text += table_md
            except Exception:
                pass

            blocks = page.get_text("blocks")
            text_parts = []
            for b in blocks:
                if b[6] == 0:
                    text_parts.append(b[4])

            text = "\n".join(text_parts)

            if tables_text:
                text = text + "\n" + tables_text

            text = self._clean_text(text)

            yield ExtractedPage(
                page_number=page_num + 1,
                text=text,
            )

    def _clean_text(self, text: str) -> str:
        """Clean extracted text of common PDF artifacts."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
        text = text.replace("\f", "\n")
        return text.strip()

    def _detect_chapters(self) -> List[ExtractedChapter]:
        """Detect chapter boundaries in the extracted text."""
        toc_chapters = self._detect_chapters_from_toc()
        if toc_chapters:
            return toc_chapters

        chapters: List[ExtractedChapter] = []

        for page in self.pages:
            header_text = page.text[:1000]
            found_on_page = False

            for pattern in CHAPTER_PATTERNS:
                matches = pattern.findall(header_text)
                for match in matches:
                    if isinstance(match, tuple) and len(match) >= 2:
                        num_str, title = match[0], match[1]
                    else:
                        num_str, title = match, ""

                    try:
                        chapter_num = int(num_str)
                        if chapter_num < 1 or chapter_num > 100:
                            continue
                    except ValueError:
                        continue

                    title = (title or "").strip()
                    if len(title) < 3:
                        lines = header_text.split("\n")
                        for i, line in enumerate(lines):
                            if num_str in line and i + 1 < len(lines):
                                title = lines[i + 1].strip()
                                break

                    if len(title) < 3:
                        title = f"Chapter {chapter_num}"

                    existing_nums = {c.chapter_number for c in chapters}
                    if chapter_num in existing_nums:
                        continue

                    chapters.append(
                        ExtractedChapter(
                            title=title,
                            chapter_number=chapter_num,
                            start_page=page.page_number,
                        )
                    )
                    found_on_page = True
                    break
                if found_on_page:
                    break

            if not found_on_page:
                match = STRICT_CHAPTER_PATTERN.search(header_text)
                if match:
                    num_str, title = match.group(1), match.group(2)
                    if any(addr in title.upper() for addr in [" ROAD", " STREET", " AVE", " DRIVE", " WAY"]):
                        continue
                    try:
                        chapter_num = int(num_str)
                        if chapter_num < 1 or chapter_num > 100:
                            continue
                        existing_nums = {c.chapter_number for c in chapters}
                        if chapter_num not in existing_nums:
                            chapters.append(
                                ExtractedChapter(
                                    title=title.strip(),
                                    chapter_number=chapter_num,
                                    start_page=page.page_number,
                                )
                            )
                    except ValueError:
                        pass

        chapters.sort(key=lambda c: c.start_page)

        chapters_with_ends: List[ExtractedChapter] = []
        for i, chapter in enumerate(chapters):
            if i + 1 < len(chapters):
                end_page = chapters[i + 1].start_page - 1
            else:
                end_page = len(self.pages) if self.pages else None

            if end_page and end_page < chapter.start_page:
                end_page = chapter.start_page

            chapters_with_ends.append(
                ExtractedChapter(
                    title=chapter.title,
                    chapter_number=chapter.chapter_number,
                    start_page=chapter.start_page,
                    end_page=end_page,
                )
            )

        if len(chapters_with_ends) < 3:
            return []

        return chapters_with_ends

    def _detect_chapters_from_toc(self) -> List[ExtractedChapter]:
        """Detect chapter boundaries using the PDF's embedded Table of Contents."""
        if self._doc is None:
            return []

        try:
            toc = self._doc.get_toc(simple=True)
        except Exception:
            return []

        if not toc:
            return []

        levels = [row[0] for row in toc if isinstance(row, (list, tuple)) and len(row) >= 3]
        if not levels:
            return []

        preferred_level = 1 if 1 in levels else min(levels)
        entries = [row for row in toc if isinstance(row, (list, tuple)) and len(row) >= 3 and row[0] == preferred_level]

        if len(entries) < 3:
            entries = [row for row in toc if isinstance(row, (list, tuple)) and len(row) >= 3 and row[0] == min(levels)]

        extracted: List[ExtractedChapter] = []

        def parse_chapter_number(title: str) -> Optional[int]:
            if not title:
                return None
            m = re.search(r"\b(?:chapter|section|module)\s+(\d{1,3})\b", title, flags=re.IGNORECASE)
            if m:
                try:
                    return int(m.group(1))
                except ValueError:
                    return None
            m2 = re.match(r"^\s*(\d{1,3})\b", title)
            if m2:
                try:
                    return int(m2.group(1))
                except ValueError:
                    return None
            return None

        total_pages = len(self.pages)
        for level, title, page in entries:
            if not isinstance(page, int):
                continue
            if page < 1 or page > max(1, total_pages):
                continue
            clean_title = (title or "").strip()
            if len(clean_title) < 3:
                continue
            extracted.append(
                ExtractedChapter(
                    title=clean_title,
                    chapter_number=parse_chapter_number(clean_title),
                    start_page=page,
                    end_page=None,
                )
            )

        by_page: dict[int, ExtractedChapter] = {}
        for ch in extracted:
            by_page.setdefault(ch.start_page, ch)
        extracted = sorted(by_page.values(), key=lambda c: c.start_page)

        if len(extracted) < 3:
            return []

        chapters_with_ends: List[ExtractedChapter] = []
        for i, ch in enumerate(extracted):
            if i + 1 < len(extracted):
                end_page = extracted[i + 1].start_page - 1
            else:
                end_page = total_pages if total_pages else None
            if end_page and end_page < ch.start_page:
                end_page = ch.start_page
            chapters_with_ends.append(
                ExtractedChapter(
                    title=ch.title,
                    chapter_number=ch.chapter_number,
                    start_page=ch.start_page,
                    end_page=end_page,
                )
            )

        return chapters_with_ends

    def get_page_text(self, page_number: int) -> str:
        """Get text for a specific page."""
        if page_number < 1 or page_number > len(self.pages):
            raise IndexError(f"Page {page_number} out of range (1-{len(self.pages)})")
        return self.pages[page_number - 1].text

    def get_page_range_text(self, start_page: int, end_page: int) -> str:
        """Get combined text for a range of pages."""
        texts = []
        for page_num in range(start_page, end_page + 1):
            texts.append(f"[Page {page_num}]\n{self.get_page_text(page_num)}")
        return "\n\n".join(texts)

    def save_json(self, output_path: Union[str, Path]) -> Path:
        """Save extracted content to a JSON file for inspection."""
        if self.metadata is None:
            raise RuntimeError("Must call extract() before save_json()")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "metadata": self.metadata.model_dump(mode="json"),
            "pages": [p.model_dump() for p in self.pages],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

        return output_path

    def print_summary(self) -> None:
        """Print a human-readable summary of the extraction."""
        if self.metadata is None:
            print("No extraction performed yet. Call extract() first.")
            return

        print(f"\n{'='*60}")
        print("PDF EXTRACTION SUMMARY")
        print(f"{'='*60}")
        print(f"File:        {self.metadata.filename}")
        print(f"Title:       {self.metadata.title or '(not set)'}")
        print(f"Author:      {self.metadata.author or '(not set)'}")
        print(f"Pages:       {self.metadata.total_pages:,}")
        print(f"Words:       {self.metadata.total_words:,}")
        print(f"Characters:  {self.metadata.total_chars:,}")
        print(f"Extracted:   {self.metadata.extracted_at.strftime('%Y-%m-%d %H:%M:%S')}")

        if self.metadata.chapters:
            print(f"\nDETECTED CHAPTERS ({len(self.metadata.chapters)}):")
            print("-" * 50)
            for ch in self.metadata.chapters:
                num = f"Ch {ch.chapter_number}" if ch.chapter_number else "Section"
                print(f"  {num:8} {ch.title[:40]:40} {ch.page_range}")
        else:
            print("\nNo chapters detected (may need manual mapping)")

        print(f"{'='*60}\n")


def process_pdf(pdf_path: Union[str, Path], output_json: Optional[Union[str, Path]] = None) -> PDFMetadata:
    """Convenience function to process a PDF and optionally save to JSON."""
    processor = PDFProcessor(pdf_path)
    metadata = processor.extract()
    processor.print_summary()

    if output_json:
        saved_path = processor.save_json(output_json)
        print(f"Saved extraction to: {saved_path}")

    return metadata


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m mededucation.ingest.pdf_processor <pdf_path> [output.json]")
        sys.exit(1)

    pdf_file = sys.argv[1]
    json_output = sys.argv[2] if len(sys.argv) > 2 else None

    process_pdf(pdf_file, json_output)
