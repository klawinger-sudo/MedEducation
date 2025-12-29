"""Content models for PDF extraction and storage."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from pydantic import BaseModel, Field, computed_field


class ExtractedPage(BaseModel):
    """A single page extracted from a PDF."""

    page_number: int = Field(..., description="1-based page number")
    text: str = Field(..., description="Raw text content of the page")
    char_count: int = Field(default=0, description="Character count")
    word_count: int = Field(default=0, description="Word count")
    chapter: Optional[int] = Field(default=None, description="Chapter number this page belongs to")
    section: Optional[str] = Field(default=None, description="Section title if detected")

    def model_post_init(self, __context: Any) -> None:
        """Calculate counts after initialization."""
        if self.char_count == 0:
            object.__setattr__(self, "char_count", len(self.text))
        if self.word_count == 0:
            object.__setattr__(self, "word_count", len(self.text.split()))


class ExtractedChapter(BaseModel):
    """A detected chapter or major section from the PDF."""

    title: str = Field(..., description="Chapter title")
    chapter_number: Optional[int] = Field(default=None, description="Chapter number if detected")
    start_page: int = Field(..., description="Starting page (1-based)")
    end_page: Optional[int] = Field(default=None, description="Ending page (1-based), None if unknown")

    @computed_field
    @property
    def page_range(self) -> str:
        """Human-readable page range."""
        if self.end_page:
            return f"pp. {self.start_page}-{self.end_page}"
        return f"p. {self.start_page}+"


class PDFMetadata(BaseModel):
    """Metadata about an extracted PDF."""

    filename: str = Field(..., description="Original filename")
    source_path: str = Field(..., description="Path to source PDF")
    title: Optional[str] = Field(default=None, description="PDF title from metadata")
    author: Optional[str] = Field(default=None, description="PDF author from metadata")
    total_pages: int = Field(..., description="Total page count")
    total_chars: int = Field(default=0, description="Total character count")
    total_words: int = Field(default=0, description="Total word count")
    chapters: List[ExtractedChapter] = Field(default_factory=list, description="Detected chapters")
    extracted_at: datetime = Field(default_factory=datetime.now, description="Extraction timestamp")
    extraction_version: str = Field(default="1.0", description="Extraction algorithm version")

    @computed_field
    @property
    def short_name(self) -> str:
        """Short reference name for the PDF."""
        name = Path(self.filename).stem
        name = re.sub(r"[-_]?\d+(?:th|rd|nd|st)?[-_]?(?:ed|edition)?", "", name, flags=re.IGNORECASE)
        return name.strip("-_ ")


class ChunkResult(BaseModel):
    """A chunk of text with metadata, used for search results."""

    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    text: str = Field(..., description="Chunk text content")
    start_page: int = Field(..., description="Starting page number (1-based)")
    end_page: int = Field(..., description="Ending page number (1-based)")
    chapter: Optional[int] = Field(default=None, description="Chapter number if known")
    section: Optional[str] = Field(default=None, description="Section title if known")
    source_name: str = Field(..., description="Source document identifier")
    token_estimate: int = Field(default=0, description="Estimated token count")

    # Search-related fields (populated during retrieval)
    relevance_score: float = Field(default=0.0, description="Search relevance score (0-1)")

    @computed_field
    @property
    def citation(self) -> str:
        """Format a citation string for this chunk."""
        if self.start_page == self.end_page:
            return f"({self.source_name}, p. {self.start_page})"
        return f"({self.source_name}, pp. {self.start_page}-{self.end_page})"

    @computed_field
    @property
    def word_count(self) -> int:
        """Word count for the chunk."""
        return len(self.text.split())
