"""ChromaDB vector store for semantic search over medical textbook content.

This module provides storage and retrieval of text chunks using embeddings.
It uses ChromaDB for persistent vector storage with sentence-transformers
for embedding generation.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from mededucation.models.content import ChunkResult


class VectorStore:
    """ChromaDB-backed vector store for semantic search.

    Stores text chunks with embeddings for efficient semantic similarity
    search. Uses sentence-transformers for generating embeddings locally.

    Example:
        >>> store = VectorStore(persist_directory="./data/vectordb")
        >>> store.add_chunks(chunks, source_id="textbook1")
        >>> results = store.search("airway management", top_k=5)
    """

    def __init__(
        self,
        persist_directory: str = "./data/vectordb",
        collection_name: str = "mededucation_content",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """Initialize the vector store.

        Args:
            persist_directory: Path to persist ChromaDB data.
            collection_name: Name of the ChromaDB collection.
            embedding_model: Sentence transformer model for embeddings.
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        self._client = None
        self._collection = None
        self._embedding_function = None

    def _ensure_initialized(self) -> None:
        """Lazily initialize ChromaDB and embeddings."""
        if self._client is not None:
            return

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("ChromaDB is required. Install it with: pip install chromadb")

        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        self._embedding_function = self._get_embedding_function()

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def _get_embedding_function(self):
        """Get the embedding function for ChromaDB."""
        try:
            from chromadb.utils import embedding_functions

            return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.embedding_model)
        except ImportError:
            raise ImportError("sentence-transformers is required. Install it with: pip install sentence-transformers")

    def add_chunks(self, chunks: List[ChunkResult], source_id: str, batch_size: int = 100) -> int:
        """Add chunks to the vector store.

        Args:
            chunks: List of ChunkResult objects to store.
            source_id: Identifier for the source document.
            batch_size: Number of chunks to add per batch.

        Returns:
            Number of chunks added.
        """
        self._ensure_initialized()

        if not chunks:
            return 0

        added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            ids = [f"{source_id}_{chunk.chunk_id}" for chunk in batch]
            documents = [chunk.text for chunk in batch]
            metadatas = [
                {
                    "chunk_id": chunk.chunk_id,
                    "source_id": source_id,
                    "source_name": chunk.source_name,
                    "start_page": chunk.start_page,
                    "end_page": chunk.end_page,
                    "chapter": chunk.chapter if chunk.chapter else 0,
                    "section": chunk.section or "",
                    "token_estimate": chunk.token_estimate,
                    "citation": chunk.citation,
                }
                for chunk in batch
            ]

            self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            added += len(batch)

        return added

    def search(
        self,
        query: str,
        top_k: int = 12,
        source_id: Optional[str] = None,
        chapter: Optional[int] = None,
        min_score: float = 0.0,
    ) -> List[ChunkResult]:
        """Search for relevant chunks.

        Args:
            query: Search query text.
            top_k: Maximum number of results to return.
            source_id: Filter to specific source document.
            chapter: Filter to specific chapter.
            min_score: Minimum relevance score (0-1).

        Returns:
            List of ChunkResult objects sorted by relevance.
        """
        self._ensure_initialized()

        where = None
        where_clauses = []

        if source_id:
            where_clauses.append({"source_id": source_id})
        if chapter:
            where_clauses.append({"chapter": chapter})

        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        chunks: List[ChunkResult] = []

        if results and results["documents"] and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
            distances = results["distances"][0] if results["distances"] else [1.0] * len(documents)

            for doc, meta, dist in zip(documents, metadatas, distances):
                score = 1.0 - dist

                if score < min_score:
                    continue

                chunk = ChunkResult(
                    chunk_id=meta.get("chunk_id", "unknown"),
                    text=doc,
                    start_page=meta.get("start_page", 0),
                    end_page=meta.get("end_page", 0),
                    chapter=meta.get("chapter") if meta.get("chapter", 0) > 0 else None,
                    section=meta.get("section") or None,
                    source_name=meta.get("source_name", meta.get("source_id", "Unknown")),
                    token_estimate=meta.get("token_estimate", 0),
                    relevance_score=score,
                )
                chunks.append(chunk)

        return chunks

    def delete_source(self, source_id: str) -> int:
        """Delete all chunks from a specific source.

        Args:
            source_id: Source document identifier.

        Returns:
            Number of chunks deleted (approximate).
        """
        self._ensure_initialized()

        count_before = self._collection.count()
        self._collection.delete(where={"source_id": source_id})
        count_after = self._collection.count()
        return count_before - count_after

    def get_sources(self) -> List[Dict[str, Any]]:
        """Get list of indexed sources with statistics.

        Returns:
            List of source info dictionaries.
        """
        self._ensure_initialized()

        results = self._collection.get(include=["metadatas"])

        sources: Dict[str, Dict[str, Any]] = {}

        if results and results["metadatas"]:
            for meta in results["metadatas"]:
                source_id = meta.get("source_id", "unknown")
                if source_id not in sources:
                    sources[source_id] = {
                        "source_id": source_id,
                        "source_name": meta.get("source_name", source_id),
                        "chunk_count": 0,
                        "chapters": set(),
                    }
                sources[source_id]["chunk_count"] += 1
                chapter = meta.get("chapter", 0)
                if chapter > 0:
                    sources[source_id]["chapters"].add(chapter)

        for source in sources.values():
            source["chapters"] = sorted(list(source["chapters"]))

        return list(sources.values())

    def get_chunk_count(self) -> int:
        """Get total number of chunks in the store.

        Returns:
            Total chunk count.
        """
        self._ensure_initialized()
        return self._collection.count()

    def clear(self) -> None:
        """Clear all data from the vector store."""
        self._ensure_initialized()
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def save_chunks_json(self, filepath: str, source_id: Optional[str] = None) -> int:
        """Export chunks to JSON file for backup/inspection.

        Args:
            filepath: Path to write JSON file.
            source_id: Optional filter by source.

        Returns:
            Number of chunks exported.
        """
        self._ensure_initialized()

        where = {"source_id": source_id} if source_id else None
        results = self._collection.get(where=where, include=["documents", "metadatas"])

        chunks_data = []
        if results and results["documents"]:
            for doc, meta in zip(results["documents"], results["metadatas"]):
                chunks_data.append({"text": doc, **meta})

        with open(filepath, "w") as f:
            json.dump(chunks_data, f, indent=2)

        return len(chunks_data)
