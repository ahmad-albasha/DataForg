<div align="center">

<br/>

# ⟨ DataForg ⟩

### PDF Intelligence Pipeline · Multilingual Chunking · Local RAG Engine

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-Local_AI-black?style=for-the-badge)](https://ollama.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)]()
[![Language](https://img.shields.io/badge/Multilingual-AR_|_EN-orange?style=for-the-badge)]()

<br/>

> **DataForg** forges raw PDF documents into structured, queryable intelligence —  
> fully offline, bilingual, and powered by a local AI model.

<br/>

</div>

---

## What is DataForg?

**DataForg** is a complete document processing pipeline that transforms PDF books and documents into clean, structured JSON data — then makes them queryable through a fully local RAG (Retrieval-Augmented Generation) engine, with no internet connection required.

The project is built around **two core modules**:

| Module | Purpose |
|--------|---------|
| 🗂️ **PDF → JSON Converter** | Parse, clean, and chunk PDF files into structured JSON |
| 🤖 **RAG Engine** | Answer questions from your indexed data using a local AI model |

---

## Features

### 🗂️ Module 1 — PDF to JSON Converter

- ✅ Full PDF-to-JSON conversion with clean, structured output
- ✅ Intelligent chunking with configurable size and overlap
- ✅ Full support for **Arabic and English** text (and mixed documents)
- ✅ Automatic text direction detection (RTL / LTR)
- ✅ Structure preservation: headings, paragraphs, lists, tables
- ✅ Automatic cleaning of noise, headers, footers, and artifacts
- ✅ Rich metadata per chunk: page number, section title, language, token count

### 🤖 Module 2 — Local RAG Engine

- ✅ **100% offline** — no API keys, no external services
- ✅ Powered by local models via [Ollama](https://ollama.com) (LLaMA, Mistral, Gemma, etc.)
- ✅ Semantic search using local embeddings
- ✅ Context-aware retrieval ranked by relevance
- ✅ Responds in the same language as the question (Arabic / English)
- ✅ Source citations with page numbers in every answer
- ✅ Clean, interactive CLI interface

---

## Project Structure

```
DataForg/
│
├── core/
│   ├── pdf_parser.py          # PDF reading and text extraction
│   ├── text_cleaner.py        # Arabic & English text normalization
│   ├── chunker.py             # Intelligent text chunking
│   ├── json_exporter.py       # Structured JSON output
│   └── language_detector.py   # Per-chunk language detection
│
├── rag/
│   ├── embedder.py            # Local embedding generation
│   ├── vector_store.py        # Vector database (ChromaDB)
│   ├── retriever.py           # Semantic context retrieval
│   ├── generator.py           # Answer generation via Ollama
│   └── rag_pipeline.py        # Main RAG orchestration
│
├── cli/
│   ├── convert_cmd.py         # Conversion commands
│   └── query_cmd.py           # Query commands
│
├── data/
│   ├── input/                 # Place your PDF files here
│   ├── output/                # Generated JSON files
│   └── vector_db/             # Vector database storage
│
├── models/                    # Local model files (optional)
│
├── tests/
│   ├── test_parser.py
│   ├── test_chunker.py
│   └── test_rag.py
│
├── main.py                    # Entry point
├── config.yaml                # Configuration file
├── requirements.txt
└── README.md
```

---

## Requirements

- Python **3.10+**
- [Ollama](https://ollama.com) installed on your machine
- At least one local language model pulled via Ollama

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DataForg.git
cd DataForg
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install a local Ollama model

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (choose one)
ollama pull llama3
# or
ollama pull mistral
# or
ollama pull gemma
```

### 5. Configure the project

```bash
cp config.example.yaml config.yaml
# Edit config.yaml to match your preferences
```

---

## Usage

### 🗂️ Convert PDF to JSON

```bash
# Convert a single file
python main.py convert --input data/input/book.pdf --output data/output/

# Batch convert an entire folder
python main.py convert --input data/input/ --output data/output/ --batch

# With advanced options
python main.py convert \
  --input data/input/book.pdf \
  --output data/output/ \
  --chunk-size 500 \
  --overlap 50 \
  --language auto
```

**Sample JSON output:**

```json
{
  "metadata": {
    "title": "Book Title",
    "total_pages": 320,
    "total_chunks": 850,
    "language": "ar",
    "processed_at": "2025-03-20T10:30:00"
  },
  "chunks": [
    {
      "id": "chunk_001",
      "text": "Paragraph content here...",
      "page": 1,
      "section": "Chapter One",
      "language": "ar",
      "direction": "rtl",
      "token_count": 128
    }
  ]
}
```

---

### 🤖 Query with RAG

```bash
# Index your data first
python main.py index --data data/output/

# Start the interactive query session
python main.py query

# Direct CLI query
python main.py query --question "What are the causes of the Industrial Revolution?"

# Specify a model
python main.py query --model mistral --question "Summarize the main argument of chapter 3."
```

**Sample answer output:**

```
┌──────────────────────────────────────────────┐
│  ⟨ DataForg ⟩  RAG Engine                   │
└──────────────────────────────────────────────┘

❓ Question: What are the causes of the Industrial Revolution?

🤖 Answer:
Based on the indexed content, the Industrial Revolution was driven by...

📍 Sources:
  • Page 45 — Chapter 3: Economic Renaissance
  • Page 67 — Chapter 4: Technological Advancement
```

---

## Configuration

```yaml
# Converter settings
converter:
  chunk_size: 500          # Words per chunk
  chunk_overlap: 50        # Overlap between consecutive chunks
  language: "auto"         # auto | ar | en
  clean_headers: true      # Strip repeated headers and footers
  preserve_structure: true # Keep document hierarchy

# RAG settings
rag:
  model: "llama3"                      # Local Ollama model
  embedding_model: "nomic-embed-text"  # Embedding model
  top_k: 5                             # Number of chunks to retrieve
  temperature: 0.1                     # Response creativity (0 = factual)
  max_tokens: 1024

# Vector store
vector_store:
  type: "chromadb"
  path: "data/vector_db"
```

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `pdfplumber` / `pymupdf` | PDF text extraction |
| `langdetect` | Per-chunk language detection |
| `chromadb` | Vector database |
| `ollama` | Local model communication |
| `sentence-transformers` | Embedding generation |
| `arabic-reshaper` + `python-bidi` | Arabic text processing |
| `rich` | Beautiful CLI output |
| `pyyaml` | Configuration parsing |

---

## Roadmap

- [x] PDF to JSON conversion
- [x] Intelligent bilingual chunking (Arabic & English)
- [x] Local RAG engine via Ollama
- [ ] Web interface (FastAPI + React)
- [ ] DOCX and EPUB support
- [ ] Multi-PDF indexing in a single session
- [ ] Docker image
- [ ] HuggingFace model support

---

## Contributing

Contributions are always welcome.

```bash
# 1. Fork the repository
# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Commit your changes
git add .
git commit -m "feat: describe your change"

# 4. Push and open a Pull Request
git push origin feature/your-feature-name
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⟨ DataForg ⟩ — Built for Arabic content, academic research, and offline AI.**

⭐ If this project helped you, consider leaving a star.

</div>
