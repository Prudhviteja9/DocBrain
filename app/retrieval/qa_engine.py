# ============================================================
# QA ENGINE — Answers questions using retrieved chunks
# ============================================================
#
# This is the FINAL piece of RAG. It connects everything:
#
#   1. User asks a question
#   2. We find relevant chunks from ChromaDB (search)
#   3. We send those chunks + the question to GPT
#   4. GPT reads the chunks and answers
#   5. We return the answer + citations
#
# Real life analogy:
#   You ask a LIBRARIAN a question.
#   Librarian finds 3 relevant books (search).
#   Librarian reads those pages (context).
#   Librarian tells you the answer + "I found this on page 12" (citation).
#
# The KEY TRICK:
#   We tell GPT: "Answer ONLY from the context below.
#                 If the answer is not in the context, say I don't know."
#   This PREVENTS hallucination!

import os
from openai import OpenAI
from dotenv import load_dotenv
from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore

load_dotenv()


class QAEngine:
    """
    Answers questions using RAG (Retrieval-Augmented Generation).

    The brain of DocBrain:
      1. Search for relevant chunks
      2. Send chunks + question to GPT
      3. Return answer with citations
    """

    def __init__(self):
        """Set up all the pieces."""

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("OPENAI_API_KEY")
            except Exception:
                pass

        if not api_key:
            raise ValueError("No OpenAI API key found!")

        self.openai_client = OpenAI(api_key=api_key)
        self.embedder = Embedder()
        self.vector_store = VectorStore()

    def _is_greeting(self, text: str) -> bool:
        """Check if the message is a greeting (not a real question)."""
        greetings = ["hi", "hello", "hey", "hola", "good morning", "good evening",
                      "good afternoon", "whats up", "sup", "howdy", "greetings"]
        return text.strip().lower().rstrip("!.,?") in greetings

    def answer_question(
        self,
        question: str,
        collection_name: str = "documents",
        n_chunks: int = 5,
    ) -> dict:
        """
        Answer a question using RAG.

        Parameters:
            question:        The user's question
            collection_name: Which document collection to search
            n_chunks:        How many chunks to retrieve (default 5)

        Returns:
            {
                "answer": "The refund policy allows returns within 30 days...",
                "sources": [
                    {"text": "chunk text...", "page": 3, "source": "policy.pdf"},
                    ...
                ],
                "chunks_used": 3,
            }

        How it works (step by step):
            1. Convert question to numbers (embedding)
            2. Search ChromaDB for similar chunks
            3. Build a prompt: "Here are documents: [chunks]. Answer: [question]"
            4. Send to GPT
            5. Return answer + which chunks were used (citations)
        """

        # Handle greetings
        if self._is_greeting(question):
            return {
                "answer": "Hello! I'm DocBrain, your document assistant. Ask me any question about your uploaded documents and I'll find the answer with citations!",
                "sources": [],
                "chunks_used": 0,
            }

        # STEP 1: Convert question to embedding
        question_embedding = self.embedder.embed_text(question)

        # STEP 2: Search ChromaDB for relevant chunks
        search_results = self.vector_store.search(
            query_embedding=question_embedding,
            n_results=n_chunks,
            collection_name=collection_name,
        )

        # If no results found, we can't answer
        if not search_results:
            return {
                "answer": "I don't have any documents to answer from. Please upload documents first.",
                "sources": [],
                "chunks_used": 0,
            }

        # STEP 3: Build the context from retrieved chunks
        # We combine all chunk texts into one big "context" string
        # Like the librarian stacking all relevant pages on your desk
        context_parts = []
        for i, result in enumerate(search_results):
            source = result["metadata"].get("source", "unknown")
            page = result["metadata"].get("page", "?")
            context_parts.append(
                f"[Source: {source}, Page: {page}]\n{result['text']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # STEP 4: Build the prompt for GPT
        # THIS IS THE KEY PART OF RAG!
        # We tell GPT:
        #   - Here are the documents (context)
        #   - Answer ONLY from these documents
        #   - If answer is not in documents, say "I don't know"
        #   - Include citations (which source, which page)
        system_prompt = """You are a helpful document assistant.
Answer the user's question based ONLY on the provided context below.

RULES:
1. ONLY use information from the context to answer.
2. If the answer is NOT in the context, say "I don't have enough information in the provided documents to answer this question."
3. Always mention which source and page number your answer came from.
4. Be concise and accurate.
5. Do NOT make up information that is not in the context."""

        user_prompt = f"""CONTEXT (from uploaded documents):
{context}

---

QUESTION: {question}

Please answer based ONLY on the context above. Include the source and page number."""

        # STEP 5: Send to GPT
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,  # 0 = consistent answers (no randomness)
        )

        answer = response.choices[0].message.content

        # STEP 6: Package the result with citations
        sources = []
        for result in search_results:
            sources.append({
                "text": result["text"][:200],  # First 200 chars as preview
                "page": result["metadata"].get("page", "?"),
                "source": result["metadata"].get("source", "unknown"),
                "similarity": round(1 - result["distance"], 3),  # Convert distance to similarity
            })

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(search_results),
        }
