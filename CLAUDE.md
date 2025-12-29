# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MedEducation** is a RAG (Retrieval-Augmented Generation) assistant for medical education. It allows users to query medical textbooks with natural language and receive detailed answers with proper citations.

**Target User**: Flight Paramedic / Critical Care Nurse needing high-level clinical education with:
- Detailed pathophysiology explanations
- Differential diagnosis tables
- Drug dosages with drip calculations
- Clinical pearls and transport considerations
- Memory aids and mnemonics

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MedEducation                              │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │   PDF    │───▶│ Chunker  │───▶│ ChromaDB │───▶│   LLM    │  │
│  │ Ingester │    │ (HQ)     │    │ (Vector) │    │ (Ollama) │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                       │         │
│                                         ┌─────────────┘         │
│                                         ▼                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                     Interfaces                            │  │
│  │  ┌─────────┐  ┌─────────────┐  ┌──────────────────────┐  │  │
│  │  │   CLI   │  │  Gradio UI  │  │  (Future: REST API)  │  │  │
│  │  └─────────┘  └─────────────┘  └──────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
MedEducation/
├── src/mededucation/
│   ├── __init__.py
│   ├── cli.py                 # Typer CLI - main entry point
│   ├── chat/
│   │   ├── __init__.py
│   │   └── engine.py          # RAG query engine (ChatEngine class)
│   ├── chunking/
│   │   ├── __init__.py
│   │   └── high_quality.py    # Semantic text chunking (HighQualityChunker)
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── pdf_processor.py   # PDF extraction (PDFProcessor)
│   │   └── epub_processor.py  # EPUB extraction (EPUBProcessor)
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py          # OpenAI-compatible LLM client (LocalLLMClient)
│   ├── models/
│   │   ├── __init__.py
│   │   └── content.py         # Pydantic models (ChunkResult, ExtractedPage, etc.)
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── system.py          # Profile-based system prompts (PROFILES dict)
│   ├── storage/
│   │   ├── __init__.py
│   │   └── vector_store.py    # ChromaDB wrapper (VectorStore)
│   └── web/
│       ├── __init__.py
│       └── app.py             # Gradio web UI (MedEducationUI)
├── config/
│   └── sources.yaml           # Source configuration and settings
├── data/
│   ├── textbooks/             # PDF/EPUB files (user-provided)
│   ├── extracted/             # Extracted JSON from PDFs
│   └── vectordb/              # ChromaDB persistent storage
├── Dockerfile                 # Multi-stage Docker build
├── docker-compose.yml         # Docker Compose for deployment
├── pyproject.toml             # Python package configuration
└── README.md                  # User documentation
```

## Key Components

### 1. ChatEngine (`src/mededucation/chat/engine.py`)
The core RAG engine. Handles:
- Vector search via ChromaDB
- Context formatting with citations
- LLM query with personalized system prompts
- Response generation

```python
engine = ChatEngine(
    vectordb_path="./data/vectordb",
    config_path="./config/sources.yaml",
    profile="flight_critical_care"
)
response = engine.query("What are the signs of cardiogenic shock?")
```

### 2. Prompts System (`src/mededucation/prompts/system.py`)
Profile-based system prompts. Current profiles:
- `flight_critical_care` (default) - Detailed, high-level clinical
- `medical_student` - Educational with board focus
- `nursing_student` - Patient care focused
- `ems_provider` - Prehospital protocols

### 3. VectorStore (`src/mededucation/storage/vector_store.py`)
ChromaDB wrapper for semantic search:
- Embedding model: `all-MiniLM-L6-v2` (384-dim)
- Collection: `mededucation_content`
- Stores: chunk text, page numbers, chapter, source metadata

### 4. HighQualityChunker (`src/mededucation/chunking/high_quality.py`)
Semantic-aware text chunking:
- Target: 380 tokens per chunk
- Overlap: 70 tokens
- Removes repeated headers/footers
- Preserves structure (headings, lists, tables)

### 5. PDFProcessor (`src/mededucation/ingest/pdf_processor.py`)
PDF extraction with:
- Page number tracking
- Chapter detection (ToC or heading patterns)
- Table extraction as markdown
- Text cleaning

## CLI Commands

```bash
# Ingest a PDF
mededucation ingest <pdf_path> --source-id <ID> --source-name <NAME>

# Build vector index
mededucation index --source-id <ID>

# List sources
mededucation sources

# Query (single question)
mededucation query "your question" [--source <ID>] [--profile <PROFILE>]

# Interactive chat
mededucation chat [--profile <PROFILE>]

# List profiles
mededucation profiles

# Test LLM connection
mededucation test

# Launch web UI
mededucation web [--host 0.0.0.0] [--port 7860]
```

## Configuration

### Environment Variables
```bash
LOCAL_LLM_BASE_URL=http://localhost:11434/v1  # Ollama URL
LOCAL_LLM_MODEL=qwen3:14b                      # Model name
TOKENIZERS_PARALLELISM=false                   # Suppress warnings
```

### config/sources.yaml
```yaml
sources:
  - id: BOOK_ID
    title: "Full Book Title"
    short_name: "ShortName"
    path: "data/textbooks/book.pdf"
    apa_reference: "Author. (Year). Title. Publisher."

settings:
  profile: flight_critical_care
  top_k: 12
  min_relevance: 0.25
  llm:
    base_url: "http://192.168.68.55:11434/v1"  # 4090 PC with Ollama
    model: "qwen3:14b"
```

## Development Commands

```bash
# Install in development mode
pip install -e ".[all]"

# Run tests (when added)
pytest tests/

# Lint
ruff check src/mededucation/
ruff format src/mededucation/

# Build Docker image
docker build -t mededucation:latest .

# Run with Docker Compose
docker-compose up -d web
```

## Docker Deployment (Unraid)

```bash
# Build
docker build -t mededucation:latest .

# Ingest textbooks
docker run --rm -v $(pwd)/data:/app/data \
  mededucation:latest \
  mededucation ingest /app/data/textbooks/book.pdf -s BOOK_ID -n "Name"

# Index
docker run --rm -v $(pwd)/data:/app/data \
  mededucation:latest \
  mededucation index -s BOOK_ID

# Run web UI
docker run -d --name mededucation \
  -p 7860:7860 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  -e LOCAL_LLM_BASE_URL=http://192.168.68.55:11434/v1 \
  mededucation:latest
```

## Code Patterns

### Lazy Initialization
All heavy resources use lazy init:
```python
def _ensure_initialized(self):
    if self._client is None:
        self._client = create_client()
```

### Pydantic Models
All data structures use Pydantic v2:
```python
class ChunkResult(BaseModel):
    chunk_id: str
    text: str
    start_page: int
    end_page: int
    source_name: str
    relevance_score: float = 0.0
```

### Error Handling
Graceful degradation with user-friendly messages.

## Current Infrastructure

- **This Machine (Unraid)**: 192.168.68.78
  - Runs: Docker containers, ChromaDB storage
  - GPU: RTX 5060 Ti (available for local inference if needed)
  - Storage: 24TB for textbooks and data

- **4090 PC**: 192.168.68.55 (assumed, verify)
  - Runs: Ollama with qwen3:14b or qwen3:32b
  - Used for: LLM inference (faster)

## Common Tasks

### Add a New Textbook
1. Copy PDF to `data/textbooks/`
2. Run `mededucation ingest <path> -s ID -n "Name"`
3. Run `mededucation index -s ID`
4. (Optional) Add to `config/sources.yaml` for APA citations

### Change the LLM Model
1. Pull new model: `ollama pull model:tag`
2. Update `config/sources.yaml` or set `LOCAL_LLM_MODEL` env var
3. Restart web UI

### Add a New Profile
1. Edit `src/mededucation/prompts/system.py`
2. Add new entry to `PROFILES` dict
3. Set as default in config or use `--profile` flag

### Modify Response Format
Edit the profile prompt in `src/mededucation/prompts/system.py`

## Dependencies

Core:
- `pydantic>=2.5.0` - Data validation
- `pymupdf>=1.24.0` - PDF extraction
- `chromadb>=0.4.0` - Vector storage
- `sentence-transformers>=2.2.0` - Embeddings
- `openai>=1.12.0` - LLM client (OpenAI-compatible)
- `typer>=0.9.0` - CLI
- `gradio>=4.0.0` - Web UI

## Troubleshooting

### LLM Connection Failed
```bash
# Check Ollama is running
curl http://192.168.68.55:11434/api/tags

# Check Ollama listening on all interfaces
# On the Ollama machine:
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

### No Sources Found
```bash
# Verify textbooks are ingested
mededucation sources

# Re-index if needed
mededucation index -s SOURCE_ID --rebuild
```

### ChromaDB Issues
```bash
# Delete and rebuild vector store
rm -rf data/vectordb/*
mededucation index -s SOURCE_ID
```
