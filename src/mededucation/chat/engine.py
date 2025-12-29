"""RAG Chat Engine for medical education queries.

This module provides the core query functionality:
1. Takes a user question
2. Retrieves relevant chunks from the vector store
3. Formats context with citations
4. Sends to LLM with personalized system prompt
5. Returns answer with proper citations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from mededucation.models.content import ChunkResult
from mededucation.storage.vector_store import VectorStore
from mededucation.llm.client import LocalLLMClient
from mededucation.prompts.system import get_system_prompt, get_context_prompt, PROFILES, DEFAULT_PROFILE


@dataclass
class ChatResponse:
    """Response from the chat engine."""

    answer: str
    chunks_used: List[ChunkResult] = field(default_factory=list)
    query: str = ""
    sources_formatted: str = ""


class ChatEngine:
    """RAG-based chat engine for medical education queries.

    Example:
        >>> engine = ChatEngine(
        ...     vectordb_path="./data/vectordb",
        ...     config_path="./config/sources.yaml",
        ...     profile="flight_critical_care"
        ... )
        >>> response = engine.query("What are the signs of cardiogenic shock?")
        >>> print(response.answer)
    """

    def __init__(
        self,
        vectordb_path: str = "./data/vectordb",
        config_path: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        top_k: int = 12,
        min_relevance: float = 0.25,
        profile: str = DEFAULT_PROFILE,
        custom_instructions: Optional[str] = None,
    ):
        """Initialize the chat engine.

        Args:
            vectordb_path: Path to ChromaDB storage.
            config_path: Path to sources.yaml config file.
            llm_base_url: Override LLM base URL.
            llm_model: Override LLM model name.
            top_k: Number of chunks to retrieve (default 12 for detailed responses).
            min_relevance: Minimum relevance score for chunks.
            profile: User profile for personalized prompts.
            custom_instructions: Additional instructions to append to system prompt.
        """
        self.vectordb_path = Path(vectordb_path)
        self.top_k = top_k
        self.min_relevance = min_relevance
        self.profile = profile
        self.custom_instructions = custom_instructions

        # Get personalized system prompt
        self._system_prompt = get_system_prompt(profile, custom_instructions)

        # Load config if provided
        self.config = {}
        self.sources = {}
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f) or {}

            # Build source lookup
            for source in self.config.get("sources", []):
                if source:
                    self.sources[source.get("id", "")] = source

            # Apply config settings
            settings = self.config.get("settings", {})
            self.top_k = settings.get("top_k", top_k)
            self.min_relevance = settings.get("min_relevance", min_relevance)

            # Profile from config
            if "profile" in settings and not profile:
                self.profile = settings["profile"]
                self._system_prompt = get_system_prompt(self.profile, custom_instructions)

        # Initialize vector store
        self._vector_store: Optional[VectorStore] = None

        # Initialize LLM client
        llm_config = self.config.get("settings", {}).get("llm", {})
        self._llm = LocalLLMClient(
            base_url=llm_base_url or llm_config.get("base_url"),
            model=llm_model or llm_config.get("model"),
            max_tokens=llm_config.get("max_tokens", 8192),  # Larger for detailed responses
            temperature=llm_config.get("temperature", 0.7),
        )

    def _ensure_vector_store(self) -> VectorStore:
        """Lazily initialize the vector store."""
        if self._vector_store is None:
            self._vector_store = VectorStore(
                persist_directory=str(self.vectordb_path),
                collection_name="mededucation_content",
            )
        return self._vector_store

    def set_profile(self, profile: str, custom_instructions: Optional[str] = None) -> None:
        """Change the user profile.

        Args:
            profile: Profile key from PROFILES.
            custom_instructions: Additional instructions.
        """
        if profile in PROFILES:
            self.profile = profile
            self._system_prompt = get_system_prompt(profile, custom_instructions)

    def get_available_profiles(self) -> dict:
        """Get available user profiles.

        Returns:
            Dictionary of profile info.
        """
        return {k: {"name": v["name"], "description": v["description"]} for k, v in PROFILES.items()}

    def query(
        self,
        question: str,
        source_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> ChatResponse:
        """Query the medical textbooks.

        Args:
            question: The user's question.
            source_id: Optional filter to specific source.
            top_k: Override number of chunks to retrieve.

        Returns:
            ChatResponse with answer and citations.
        """
        store = self._ensure_vector_store()

        # Retrieve relevant chunks
        chunks = store.search(
            query=question,
            top_k=top_k or self.top_k,
            source_id=source_id,
            min_score=self.min_relevance,
        )

        if not chunks:
            return ChatResponse(
                answer="I couldn't find relevant information in the textbooks to answer your question. "
                "Please try rephrasing your question or check that the relevant textbooks have been indexed.",
                chunks_used=[],
                query=question,
                sources_formatted="",
            )

        # Format context with citations
        context = self._format_context(chunks)

        # Build the prompt using the prompts module
        prompt = get_context_prompt(question, context)

        # Generate response with personalized system prompt
        answer = self._llm.generate(prompt=prompt, system_prompt=self._system_prompt)

        # Format sources section
        sources_formatted = self._format_sources(chunks)

        return ChatResponse(
            answer=answer,
            chunks_used=chunks,
            query=question,
            sources_formatted=sources_formatted,
        )

    def _format_context(self, chunks: List[ChunkResult]) -> str:
        """Format retrieved chunks as context for the LLM."""
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            # Get full source name if available
            source_name = chunk.source_name
            if chunk.source_name in self.sources:
                source_name = self.sources[chunk.source_name].get("short_name", chunk.source_name)

            # Format citation
            if chunk.start_page == chunk.end_page:
                page_ref = f"p. {chunk.start_page}"
            else:
                page_ref = f"pp. {chunk.start_page}-{chunk.end_page}"

            header = f"[EXCERPT {i} | {source_name}, {page_ref}]"
            if chunk.chapter:
                header = f"[EXCERPT {i} | {source_name}, Chapter {chunk.chapter}, {page_ref}]"

            context_parts.append(f"{header}\n{chunk.text}\n")

        return "\n---\n".join(context_parts)

    def _format_sources(self, chunks: List[ChunkResult]) -> str:
        """Format a sources section for the response."""
        # Group by source
        source_pages: dict[str, set[int]] = {}
        for chunk in chunks:
            source_id = chunk.source_name
            if source_id not in source_pages:
                source_pages[source_id] = set()
            for p in range(chunk.start_page, chunk.end_page + 1):
                source_pages[source_id].add(p)

        lines = ["Sources consulted:"]
        for source_id, pages in source_pages.items():
            # Get full reference if available
            if source_id in self.sources:
                source_info = self.sources[source_id]
                ref = source_info.get("apa_reference", source_info.get("title", source_id))
                short_name = source_info.get("short_name", source_id)
            else:
                ref = source_id
                short_name = source_id

            page_list = sorted(pages)
            if len(page_list) == 1:
                page_str = f"p. {page_list[0]}"
            else:
                page_str = f"pp. {page_list[0]}-{page_list[-1]}"

            lines.append(f"- {short_name} ({page_str}): {ref}")

        return "\n".join(lines)

    def get_sources(self) -> List[dict]:
        """Get list of indexed sources.

        Returns:
            List of source info dictionaries.
        """
        store = self._ensure_vector_store()
        return store.get_sources()

    def test_connection(self) -> dict:
        """Test LLM connection.

        Returns:
            Connection status dictionary.
        """
        return self._llm.test_connection()
