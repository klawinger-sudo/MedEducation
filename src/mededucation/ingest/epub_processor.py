"""
EPUB text extraction with chapter detection.

This module handles extracting text from EPUB textbooks while
preserving structure for accurate citations.

Usage:
    from mededucation.ingest import EPUBProcessor

    processor = EPUBProcessor("textbook.epub")
    result = processor.extract()
    processor.save_json("output/extracted.json")
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Union

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, NavigableString

from mededucation.models.content import ExtractedChapter, ExtractedPage, PDFMetadata


class EPUBProcessor:
    """
    Extracts text and metadata from EPUB files.

    Note: EPUBs don't have fixed pages like PDFs. Each XHTML document
    in the EPUB spine is treated as a "page" for citation purposes.
    """

    def __init__(self, epub_path: Union[str, Path]):
        self.epub_path = Path(epub_path)

        if not self.epub_path.exists():
            raise FileNotFoundError(f"EPUB not found: {self.epub_path}")

        if self.epub_path.suffix.lower() != ".epub":
            raise ValueError(f"File is not an EPUB: {self.epub_path}")

        self.pages: List[ExtractedPage] = []
        self.metadata: Optional[PDFMetadata] = None
        self._book: Optional[epub.EpubBook] = None

    def extract(self) -> PDFMetadata:
        """Extract all text and metadata from the EPUB."""
        self._book = epub.read_epub(str(self.epub_path))

        try:
            self.pages = list(self._extract_pages())
            chapters = self._detect_chapters()

            for chapter in chapters:
                start = chapter.start_page - 1
                end = chapter.end_page if chapter.end_page else len(self.pages)
                for i in range(start, end):
                    if i < len(self.pages):
                        self.pages[i].chapter = chapter.chapter_number

            title = self._get_metadata("DC", "title")
            author = self._get_metadata("DC", "creator")

            self.metadata = PDFMetadata(
                filename=self.epub_path.name,
                source_path=str(self.epub_path.absolute()),
                title=title,
                author=author,
                total_pages=len(self.pages),
                total_chars=sum(p.char_count for p in self.pages),
                total_words=sum(p.word_count for p in self.pages),
                chapters=chapters,
                extracted_at=datetime.now(),
                extraction_version="2.0-epub",
            )

            return self.metadata

        finally:
            self._book = None

    def _get_metadata(self, namespace: str, name: str) -> Optional[str]:
        """Safely get metadata from EPUB."""
        if self._book is None:
            return None
        try:
            meta = self._book.get_metadata(namespace, name)
            if meta and len(meta) > 0:
                return meta[0][0] if isinstance(meta[0], tuple) else str(meta[0])
        except Exception:
            pass
        return None

    def _extract_pages(self) -> Iterator[ExtractedPage]:
        """Extract text from each document in the EPUB spine."""
        if self._book is None:
            raise RuntimeError("EPUB not opened")

        page_num = 0

        for item in self._book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            try:
                content = item.get_content()
                soup = BeautifulSoup(content, "html.parser")

                body = soup.find("body")
                if not body:
                    body = soup

                tables_text = self._extract_tables(soup)
                text = self._extract_text_structured(body)

                if tables_text:
                    text = text + "\n\n" + tables_text

                text = self._clean_text(text)

                if not text or len(text.strip()) < 50:
                    continue

                page_num += 1
                chapter_title = self._detect_heading(soup)

                yield ExtractedPage(
                    page_number=page_num,
                    text=text,
                    section=chapter_title,
                )

            except Exception as e:
                print(f"Warning: Error processing EPUB item: {e}")
                continue

    def _extract_tables(self, soup: BeautifulSoup) -> str:
        """Extract tables and convert to markdown format."""
        tables_text = []

        for i, table in enumerate(soup.find_all("table")):
            try:
                rows = table.find_all("tr")
                if not rows:
                    continue

                table_data = []
                for row in rows:
                    cells = row.find_all(["th", "td"])
                    row_data = [cell.get_text(strip=True).replace("|", "\\|") for cell in cells]
                    if row_data:
                        table_data.append(row_data)

                if not table_data:
                    continue

                md_lines = [f"\n[TABLE {i+1}]"]

                if table_data:
                    md_lines.append("| " + " | ".join(table_data[0]) + " |")
                    md_lines.append("|" + "|".join(["---"] * len(table_data[0])) + "|")

                    for row in table_data[1:]:
                        while len(row) < len(table_data[0]):
                            row.append("")
                        md_lines.append("| " + " | ".join(row[: len(table_data[0])]) + " |")

                md_lines.append("[/TABLE]\n")
                tables_text.append("\n".join(md_lines))

                table.decompose()

            except Exception:
                continue

        return "\n".join(tables_text)

    def _extract_text_structured(self, element) -> str:
        """Extract text while preserving structure."""
        parts = []

        for child in element.children:
            if isinstance(child, NavigableString):
                text = str(child).strip()
                if text:
                    parts.append(text)
            elif child.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                text = child.get_text(strip=True)
                if text:
                    level = int(child.name[1])
                    prefix = "#" * level + " "
                    parts.append(f"\n{prefix}{text}\n")
            elif child.name == "p":
                text = child.get_text(strip=True)
                if text:
                    parts.append(f"\n{text}\n")
            elif child.name in ["ul", "ol"]:
                items = child.find_all("li", recursive=False)
                for j, li in enumerate(items):
                    text = li.get_text(strip=True)
                    if text:
                        bullet = f"{j+1}." if child.name == "ol" else "•"
                        parts.append(f"  {bullet} {text}")
                parts.append("")
            elif child.name == "blockquote":
                text = child.get_text(strip=True)
                if text:
                    parts.append(f"\n> {text}\n")
            elif child.name in ["div", "section", "article", "aside", "main"]:
                inner = self._extract_text_structured(child)
                if inner.strip():
                    parts.append(inner)
            elif child.name not in ["script", "style", "nav", "table"]:
                text = child.get_text(strip=True)
                if text:
                    parts.append(text)

        return "\n".join(parts)

    def _detect_heading(self, soup: BeautifulSoup) -> Optional[str]:
        """Try to find the main heading/title of a document."""
        for tag in ["h1", "h2", "h3"]:
            heading = soup.find(tag)
            if heading:
                text = heading.get_text(strip=True)
                if text and len(text) > 2:
                    return text
        return None

    def _clean_text(self, text: str) -> str:
        """Clean extracted text of common artifacts."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace("'", "'")
        return text.strip()

    def _detect_chapters(self) -> List[ExtractedChapter]:
        """Detect chapter boundaries using EPUB ToC or heading patterns."""
        toc_chapters = self._detect_chapters_from_toc()
        if toc_chapters:
            return toc_chapters
        return self._detect_chapters_from_headings()

    def _detect_chapters_from_toc(self) -> List[ExtractedChapter]:
        """Extract chapters from EPUB's table of contents."""
        if self._book is None:
            return []

        try:
            toc = self._book.toc
            if not toc:
                return []

            chapters: List[ExtractedChapter] = []

            def process_toc_item(item, depth=0):
                if isinstance(item, tuple):
                    section, children = item
                    if hasattr(section, "title"):
                        title = section.title
                        page_num = self._find_page_for_href(getattr(section, "href", None))
                        if page_num and title:
                            chapters.append(
                                ExtractedChapter(
                                    title=title,
                                    chapter_number=self._parse_chapter_number(title),
                                    start_page=page_num,
                                    end_page=None,
                                )
                            )
                    for child in children:
                        process_toc_item(child, depth + 1)
                elif hasattr(item, "title"):
                    title = item.title
                    page_num = self._find_page_for_href(getattr(item, "href", None))
                    if page_num and title:
                        chapters.append(
                            ExtractedChapter(
                                title=title,
                                chapter_number=self._parse_chapter_number(title),
                                start_page=page_num,
                                end_page=None,
                            )
                        )

            for item in toc:
                process_toc_item(item)

            if len(chapters) < 3:
                return []

            return self._finalize_chapters(chapters)

        except Exception as e:
            print(f"Warning: Error reading EPUB ToC: {e}")
            return []

    def _find_page_for_href(self, href: Optional[str]) -> Optional[int]:
        """Map an EPUB href to a page number."""
        if not href or not self.pages:
            return None
        href = href.split("#")[0]
        return None

    def _detect_chapters_from_headings(self) -> List[ExtractedChapter]:
        """Detect chapters from page headings."""
        chapters: List[ExtractedChapter] = []

        chapter_patterns = [
            re.compile(r"^(?:CHAPTER|Chapter)\s+(\d+)\s*[:\-\.]?\s*(.*)$", re.IGNORECASE),
            re.compile(r"^(?:SECTION|Section)\s+(\d+)\s*[:\-\.]?\s*(.*)$", re.IGNORECASE),
            re.compile(r"^(?:MODULE|Module)\s+(\d+)\s*[:\-\.]?\s*(.*)$", re.IGNORECASE),
            re.compile(r"^(?:UNIT|Unit)\s+(\d+)\s*[:\-\.]?\s*(.*)$", re.IGNORECASE),
            re.compile(r"^(?:PART|Part)\s+(\d+)\s*[:\-\.]?\s*(.*)$", re.IGNORECASE),
        ]

        for page in self.pages:
            if not page.section:
                continue

            title = page.section.strip()

            for pattern in chapter_patterns:
                match = pattern.match(title)
                if match:
                    try:
                        chapter_num = int(match.group(1))
                        chapter_title = match.group(2).strip() if match.group(2) else title

                        existing_nums = {c.chapter_number for c in chapters if c.chapter_number}
                        if chapter_num in existing_nums:
                            continue

                        chapters.append(
                            ExtractedChapter(
                                title=chapter_title or title,
                                chapter_number=chapter_num,
                                start_page=page.page_number,
                                end_page=None,
                            )
                        )
                        break
                    except ValueError:
                        continue

        if len(chapters) < 3:
            return []

        return self._finalize_chapters(chapters)

    def _parse_chapter_number(self, title: str) -> Optional[int]:
        """Extract chapter number from title."""
        if not title:
            return None

        m = re.search(r"\b(?:chapter|section|module|unit|part)\s+(\d+)\b", title, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass

        m2 = re.match(r"^\s*(\d{1,3})[.\s]", title)
        if m2:
            try:
                return int(m2.group(1))
            except ValueError:
                pass

        return None

    def _finalize_chapters(self, chapters: List[ExtractedChapter]) -> List[ExtractedChapter]:
        """Sort chapters and add end pages."""
        chapters.sort(key=lambda c: c.start_page)

        by_page: dict[int, ExtractedChapter] = {}
        for ch in chapters:
            by_page.setdefault(ch.start_page, ch)
        chapters = sorted(by_page.values(), key=lambda c: c.start_page)

        total_pages = len(self.pages)
        chapters_with_ends: List[ExtractedChapter] = []

        for i, ch in enumerate(chapters):
            if i + 1 < len(chapters):
                end_page = chapters[i + 1].start_page - 1
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
        if start_page < 1:
            start_page = 1
        if end_page > len(self.pages):
            end_page = len(self.pages)

        texts = []
        for i in range(start_page - 1, end_page):
            texts.append(self.pages[i].text)

        return "\n\n---\n\n".join(texts)

    def save_json(self, output_path: Union[str, Path]) -> None:
        """Save extracted content to JSON file."""
        if not self.metadata:
            raise ValueError("No content extracted. Call extract() first.")

        data = {
            "metadata": self.metadata.model_dump(mode="json"),
            "pages": [p.model_dump(mode="json") for p in self.pages],
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "EPUBProcessor":
        """Load a previously extracted EPUB from JSON."""
        json_path = Path(json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        processor = object.__new__(cls)
        processor.epub_path = Path(data["metadata"].get("source_path", "unknown.epub"))
        processor.metadata = PDFMetadata(**data["metadata"])
        processor.pages = [ExtractedPage(**p) for p in data["pages"]]
        processor._book = None

        return processor
