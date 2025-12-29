# MedEducation

A RAG (Retrieval-Augmented Generation) assistant for medical education. Query your medical textbooks with natural language and get answers with proper citations.

## Features

- **PDF/EPUB Ingestion**: Extract text from medical textbooks with page tracking
- **Semantic Search**: Find relevant passages using vector embeddings
- **Citation Support**: Every answer includes source citations (book, page numbers)
- **Local LLM**: Runs entirely on your machine with Ollama (no API costs)
- **Interactive Chat**: Ask follow-up questions in a chat session

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) (or any OpenAI-compatible LLM server)
- ~8GB disk space for models and vector store
- GPU recommended (but not required) for faster inference

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/YOUR_USERNAME/MedEducation.git
cd MedEducation

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with all dependencies
pip install -e ".[all]"
```

### 2. Set Up Ollama

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the recommended model
ollama pull qwen3:14b

# Start Ollama (if not running)
ollama serve
```

### 3. Ingest Your Textbooks

```bash
# Copy your PDF to the textbooks directory
cp /path/to/your/textbook.pdf data/textbooks/

# Ingest the PDF (extract text)
mededucation ingest data/textbooks/textbook.pdf \
  --source-id TEXTBOOK_ID \
  --source-name "Short Name"

# Build the vector index
mededucation index --source-id TEXTBOOK_ID
```

### 4. Query

```bash
# Single question
mededucation query "What are the signs of hypovolemic shock?"

# Interactive chat
mededucation chat
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `mededucation ingest <file> -s <id> -n <name>` | Extract text from PDF/EPUB |
| `mededucation index -s <id>` | Build vector index for a source |
| `mededucation sources` | List all indexed sources |
| `mededucation query "question"` | Ask a single question |
| `mededucation chat` | Start interactive chat session |
| `mededucation test` | Test LLM connection |

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
# LLM Settings
LOCAL_LLM_BASE_URL=http://localhost:11434/v1
LOCAL_LLM_MODEL=qwen3:14b
```

### Source Configuration

Edit `config/sources.yaml` to add metadata about your textbooks:

```yaml
sources:
  - id: TEXTBOOK_ID
    title: "Full Textbook Title"
    short_name: "ShortName"
    path: "data/textbooks/textbook.pdf"
    apa_reference: "Author. (Year). Title (Edition). Publisher."
```

## Project Structure

```
MedEducation/
├── config/
│   └── sources.yaml          # Source metadata & settings
├── data/
│   ├── textbooks/            # Your PDF/EPUB files
│   ├── extracted/            # Extracted JSON (auto-generated)
│   └── vectordb/             # ChromaDB storage (auto-generated)
├── src/mededucation/
│   ├── chat/                 # RAG query engine
│   ├── chunking/             # Text chunking with semantic awareness
│   ├── ingest/               # PDF/EPUB text extraction
│   ├── llm/                  # LLM client (Ollama-compatible)
│   ├── models/               # Data models
│   ├── storage/              # Vector store (ChromaDB)
│   └── cli.py                # Command-line interface
└── pyproject.toml
```

## Supported Models

Any model available through Ollama or OpenAI-compatible APIs:

| Model | VRAM | Notes |
|-------|------|-------|
| `qwen3:14b` | ~10GB | Recommended - good balance |
| `qwen3:8b` | ~6GB | Faster, slightly less quality |
| `qwen3:32b` | ~20GB | Best quality if you have the VRAM |
| `llama3.1:8b` | ~6GB | Good alternative |

## How It Works

1. **Ingestion**: PDFs are processed page-by-page, preserving page numbers and detecting chapters
2. **Chunking**: Text is split into ~380-token chunks with semantic awareness (respects headings, paragraphs, tables)
3. **Embedding**: Chunks are embedded using `all-MiniLM-L6-v2` and stored in ChromaDB
4. **Query**: Your question is embedded, similar chunks are retrieved, and sent to the LLM with instructions to cite sources
5. **Response**: The LLM generates an answer grounded in the retrieved text with proper citations

## License

MIT
