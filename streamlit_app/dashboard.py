# ============================================================
# DocBrain Dashboard — Upload docs, ask questions, get answers
# ============================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import os
import tempfile

from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_document
from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.qa_engine import QAEngine


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="DocBrain - Document Intelligence",
    page_icon="brain",
    layout="wide",
)

# --- HEADER ---
st.title("DocBrain")
st.subheader("RAG-Powered Document Intelligence Platform")
st.markdown("Upload documents. Ask questions. Get cited answers.")
st.markdown("---")

# --- SIDEBAR ---
st.sidebar.header("Settings")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    placeholder="sk-proj-...",
    help="Enter your OpenAI API key. Get one at platform.openai.com",
)

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.sidebar.success("API key set!")
else:
    st.sidebar.warning("Enter your OpenAI API key to use DocBrain.")

st.sidebar.markdown("---")

# Show collection stats
try:
    store = VectorStore()
    stats = store.get_collection_stats()
    st.sidebar.metric("Documents in Memory", stats["total_chunks"], help="Total chunks stored")
except Exception:
    st.sidebar.metric("Documents in Memory", 0)

# Reset button
if st.sidebar.button("Clear All Documents", type="secondary"):
    try:
        store = VectorStore()
        store.delete_collection("documents")
        # Clear uploaded files
        upload_dir = Path(__file__).parent.parent / "uploads"
        if upload_dir.exists():
            for f in upload_dir.iterdir():
                if f.is_file():
                    f.unlink()
        st.sidebar.success("All documents cleared!")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["Upload Documents", "Ask Questions", "About"])


# ========== TAB 1: UPLOAD ==========
with tab1:
    st.header("Upload Documents")
    st.write("Upload PDF or TXT files. DocBrain will read, chunk, embed, and store them for Q&A.")

    uploaded_files = st.file_uploader(
        "Drop your documents here",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT",
    )

    if uploaded_files and api_key:
        if st.button("Process Documents", type="primary"):
            for uploaded_file in uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        # Save to temp file
                        suffix = "." + uploaded_file.name.split(".")[-1]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name

                        # Also save to uploads folder
                        upload_dir = Path(__file__).parent.parent / "uploads"
                        upload_dir.mkdir(exist_ok=True)
                        save_path = upload_dir / uploaded_file.name
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getvalue())

                        # Step 1: Load
                        pages = load_document(tmp_path)
                        st.write(f"  Loaded {len(pages)} pages")

                        # Step 2: Chunk
                        chunks = chunk_document(pages, chunk_size=500, chunk_overlap=50)
                        st.write(f"  Created {len(chunks)} chunks")

                        # Step 3: Embed
                        embedder = Embedder()
                        texts = [c["text"] for c in chunks]
                        embeddings = embedder.embed_many(texts)
                        st.write(f"  Generated {len(embeddings)} embeddings")

                        # Step 4: Store
                        store = VectorStore()
                        count = store.add_chunks(chunks, embeddings, source_name=uploaded_file.name)
                        st.success(f"{uploaded_file.name} processed! {count} chunks stored.")

                        # Clean up temp file
                        os.unlink(tmp_path)

                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")

    elif uploaded_files and not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar first.")


# ========== TAB 2: ASK QUESTIONS ==========
with tab2:
    st.header("Ask Questions")
    st.write("Ask anything about your uploaded documents. Answers include citations.")

    # Check if documents are loaded
    try:
        store = VectorStore()
        stats = store.get_collection_stats()
        has_docs = stats["total_chunks"] > 0
    except Exception:
        has_docs = False

    if not has_docs:
        st.info("No documents uploaded yet. Go to 'Upload Documents' tab first.")
    elif not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
    else:
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg:
                    with st.expander("View Sources"):
                        for s in msg["sources"]:
                            st.markdown(
                                f"**{s['source']}** (Page {s['page']}) - "
                                f"Similarity: {s['similarity']:.0%}"
                            )
                            st.caption(s["text"][:200] + "...")

        # Chat input
        if question := st.chat_input("Ask a question about your documents..."):
            # Show user message
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            # Get answer
            with st.chat_message("assistant"):
                with st.spinner("Searching documents and generating answer..."):
                    try:
                        qa = QAEngine()
                        result = qa.answer_question(question)

                        st.markdown(result["answer"])

                        # Show sources
                        with st.expander(f"View Sources ({result['chunks_used']} chunks used)"):
                            for s in result["sources"]:
                                st.markdown(
                                    f"**{s['source']}** (Page {s['page']}) - "
                                    f"Similarity: {s['similarity']:.0%}"
                                )
                                st.caption(s["text"][:200] + "...")

                        # Save to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                        })

                    except Exception as e:
                        st.error(f"Error: {e}")


# ========== TAB 3: ABOUT ==========
with tab3:
    st.header("How DocBrain Works")

    st.markdown("""
    ### RAG Pipeline (Retrieval-Augmented Generation)

    **1. Upload** - You upload PDF/TXT documents

    **2. Chunk** - Documents are split into small pieces (~500 characters each)

    **3. Embed** - Each chunk is converted to numbers (embeddings) using OpenAI

    **4. Store** - Embeddings are stored in ChromaDB (vector database)

    **5. Search** - When you ask a question, we find the most relevant chunks by meaning

    **6. Answer** - GPT reads the relevant chunks and answers your question with citations

    ### Tech Stack
    | Component | Technology |
    |-----------|-----------|
    | Backend | Python, FastAPI |
    | Embeddings | OpenAI text-embedding-3-small |
    | Vector DB | ChromaDB |
    | LLM | GPT-4o-mini |
    | Frontend | Streamlit |
    | Chunking | Custom recursive splitter |
    """)

    st.markdown("---")
    st.markdown(
        "Built by **Prudhvi Teja Yedla** | "
        "[GitHub](https://github.com/Prudhviteja9)"
    )
