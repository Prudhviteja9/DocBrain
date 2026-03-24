# DocBrain - RAG-Powered Document Intelligence Platform

Upload documents. Ask questions. Get cited answers. Powered by RAG (Retrieval-Augmented Generation).

## Live Demo

[Live App](https://your-aws-url-here) | [GitHub Repo](https://github.com/Prudhviteja9/DocBrain)

## Architecture

```
User (Browser)
      |
      v
  Interactive Web UI (HTML/CSS/JS)
      |
      v
  FastAPI Server
      |
      |--- POST /upload    --> Document Processing Pipeline
      |                         |
      |                         |--- PDF Loader (PyMuPDF)
      |                         |--- Chunker (recursive splitting)
      |                         |--- Embedder (OpenAI text-embedding-3-small)
      |                         |--- ChromaDB (vector storage)
      |
      |--- POST /ask       --> RAG Pipeline
      |                         |
      |                         |--- Embed question
      |                         |--- Search ChromaDB (similarity search)
      |                         |--- Send chunks + question to GPT-4o-mini
      |                         |--- Return answer + citations
      |
      |--- GET /documents  --> List uploaded documents
      |--- DELETE /reset   --> Clear all data
```

## Features

- **Interactive Web UI** - Beautiful dark-themed chat interface with drag-and-drop upload
- **RAG Pipeline** - Upload documents, ask questions, get answers from YOUR documents only
- **Smart Chunking** - Recursive text splitting with overlap to preserve context
- **Vector Search** - ChromaDB with cosine similarity for meaning-based retrieval
- **Hybrid Grading** - Answers come with citations (source document, page number, similarity score)
- **No Hallucination** - AI answers ONLY from uploaded documents, says "I don't know" when unsure
- **REST API** - FastAPI with auto-generated docs at /docs
- **Docker Ready** - Containerized for deployment

## Tech Stack

| Category | Technology |
|----------|-----------|
| Backend | Python, FastAPI |
| Frontend | HTML, CSS, JavaScript (no framework) |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Database | ChromaDB (cosine similarity) |
| LLM | GPT-4o-mini |
| PDF Processing | PyMuPDF |
| Chunking | Custom recursive splitter |
| Deployment | AWS EC2, Docker |

## How RAG Works

```
1. UPLOAD    User uploads PDF/TXT documents
2. CHUNK     Documents split into ~500 character pieces
3. EMBED     Each chunk converted to 1536-dimensional vector
4. STORE     Vectors stored in ChromaDB with metadata
5. SEARCH    Question embedded, similar chunks found by cosine similarity
6. GENERATE  GPT reads relevant chunks and answers with citations
```

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Prudhviteja9/DocBrain.git
cd DocBrain
pip install -r requirements.txt
```

### 2. Set up API key

```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

### 3. Run

```bash
uvicorn app.main:app --reload --port 8002
```

Open http://localhost:8002 and start uploading documents!

### 4. Or use Docker

```bash
docker build -t docbrain .
docker run -p 8002:8002 --env-file .env docbrain
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Interactive web interface |
| POST | `/upload` | Upload PDF/TXT document |
| POST | `/ask` | Ask question about documents |
| GET | `/documents` | List uploaded documents + stats |
| DELETE | `/reset` | Clear all documents |
| GET | `/docs` | API documentation |

## Project Structure

```
DocBrain/
├── app/
│   ├── main.py              # FastAPI endpoints + serve frontend
│   ├── engine.py             # (reserved)
│   ├── ingestion/
│   │   ├── loader.py         # PDF/TXT text extraction
│   │   ├── chunker.py        # Recursive text splitting
│   │   └── embedder.py       # OpenAI embedding generation
│   ├── retrieval/
│   │   ├── vector_store.py   # ChromaDB operations
│   │   └── qa_engine.py      # RAG question answering
│   └── schemas/
│       └── models.py         # Pydantic data models
├── static/
│   └── index.html            # Interactive web frontend
├── data/sample_docs/         # Sample documents
├── Dockerfile                # Container config
├── requirements.txt          # Dependencies
└── README.md
```

## Key Technical Decisions

- **Custom chunker over LangChain** - Built recursive splitter from scratch for Python 3.14 compatibility and zero dependencies
- **ChromaDB over Pinecone** - Local-first, no cloud dependency, persistent storage, free
- **Cosine similarity** - Best metric for text embeddings (measures angle, not magnitude)
- **Chunk overlap (50 chars)** - Prevents cutting sentences across chunk boundaries
- **Temperature 0** - Consistent, deterministic answers from GPT (no randomness)
- **No hallucination design** - System prompt instructs GPT to only answer from provided context

## What I Learned

- Designed a complete RAG pipeline from document ingestion to cited answer generation
- Built custom text chunking with recursive splitting and configurable overlap
- Implemented vector similarity search with ChromaDB using cosine distance
- Created a full-stack application with FastAPI backend and vanilla JS frontend
- Deployed on AWS EC2 with Docker containerization

## License

MIT

---

Built by [Prudhvi Teja Yedla](https://github.com/Prudhviteja9)
