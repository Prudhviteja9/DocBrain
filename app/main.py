# ============================================================
# DocBrain — RAG-Powered Document Intelligence Platform
# main.py = All API endpoints
# ============================================================

import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.schemas.models import QuestionRequest, AnswerResponse
from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_document
from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.qa_engine import QAEngine


# --- CREATE APP ---
app = FastAPI(
    title="DocBrain",
    description="RAG-Powered Document Intelligence Platform - Upload documents, ask questions, get cited answers",
    version="1.0.0",
)

# Create folders
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR = Path(__file__).parent.parent / "static"


# --- API KEY MODEL ---
class APIKeyRequest(BaseModel):
    api_key: str


# --- ENDPOINT: SERVE WEBPAGE ---
@app.get("/")
def serve_frontend():
    """Serve the interactive web interface."""
    return FileResponse(str(STATIC_DIR / "index.html"))


# --- ENDPOINT: SET API KEY ---
@app.post("/set-key")
def set_api_key(request: APIKeyRequest):
    """Set the OpenAI API key from the frontend."""
    os.environ["OPENAI_API_KEY"] = request.api_key
    return {"status": "API key set"}


# --- ENDPOINT 2: UPLOAD DOCUMENT ---
# User uploads a PDF -> we process it (chunk + embed + store in ChromaDB)
@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT file. DocBrain will:
    1. Read the document
    2. Cut it into chunks
    3. Create embeddings
    4. Store in ChromaDB for searching
    """

    # Save uploaded file to disk
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Step 1: Load the document
    pages = load_document(str(file_path))

    # Step 2: Chunk it
    chunks = chunk_document(pages, chunk_size=500, chunk_overlap=50)

    # Step 3: Create embeddings
    embedder = Embedder()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed_many(texts)

    # Step 4: Store in ChromaDB
    store = VectorStore()
    count = store.add_chunks(chunks, embeddings, source_name=file.filename)

    return {
        "message": f"Document '{file.filename}' processed successfully!",
        "filename": file.filename,
        "pages": len(pages),
        "chunks_created": count,
        "status": "ready for questions",
    }


# --- ENDPOINT 3: ASK A QUESTION ---
@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    Ask a question about uploaded documents.
    Returns an answer with citations (page number, source).
    """

    qa = QAEngine()
    result = qa.answer_question(
        question=request.question,
        collection_name=request.collection_name,
        n_chunks=request.n_chunks,
    )

    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"],
        chunks_used=result["chunks_used"],
    )


# --- ENDPOINT 4: LIST DOCUMENTS ---
@app.get("/documents")
def list_documents():
    """List all uploaded documents and collection stats."""

    store = VectorStore()
    stats = store.get_collection_stats()

    # List uploaded files
    uploaded_files = []
    if UPLOAD_DIR.exists():
        for f in UPLOAD_DIR.iterdir():
            if f.is_file():
                uploaded_files.append({
                    "filename": f.name,
                    "size_kb": round(f.stat().st_size / 1024, 1),
                })

    return {
        "uploaded_files": uploaded_files,
        "total_files": len(uploaded_files),
        "vector_store": stats,
    }


# --- ENDPOINT 5: RESET (delete all documents) ---
@app.delete("/reset")
def reset_all():
    """Delete all uploaded documents and clear the vector store."""

    # Delete uploaded files
    if UPLOAD_DIR.exists():
        for f in UPLOAD_DIR.iterdir():
            if f.is_file():
                f.unlink()

    # Delete ChromaDB collection
    store = VectorStore()
    store.delete_collection("documents")

    return {"message": "All documents and vectors deleted.", "status": "clean"}
