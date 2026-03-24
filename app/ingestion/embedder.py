# ============================================================
# EMBEDDER — Converts text into numbers (embeddings)
# ============================================================
#
# What this does:
#   Takes a piece of text -> sends to OpenAI -> gets back a list of numbers.
#   These numbers capture the MEANING of the text.
#
# Real life analogy:
#   GPS turns a city name into coordinates:
#     "Hyderabad" -> (17.38, 78.47)
#     "Secunderabad" -> (17.43, 78.50)  <- nearby = similar coordinates
#     "New York" -> (40.71, -74.00)     <- far = different coordinates
#
#   Embeddings turn TEXT into coordinates in "meaning space":
#     "refund policy" -> [0.31, 0.72, 0.18, ...]
#     "return items"  -> [0.29, 0.74, 0.20, ...]  <- similar meaning = close numbers
#     "pizza delivery" -> [0.85, 0.12, 0.63, ...]  <- different meaning = far numbers
#
# We use OpenAI's text-embedding-3-small model:
#   - Fast and cheap ($0.00002 per 1000 tokens)
#   - Returns 1536 numbers per text
#   - Good enough for most RAG applications

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class Embedder:
    """
    Converts text into numerical vectors (embeddings).

    Think of it as a TRANSLATOR:
      Input:  Human language (text)
      Output: Computer language (list of numbers)
    """

    def __init__(self):
        """Set up the connection to OpenAI."""
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("OPENAI_API_KEY")
            except Exception:
                pass

        if not api_key:
            raise ValueError("No OpenAI API key found!")

        self.client = OpenAI(api_key=api_key)
        self.model = "text-embedding-3-small"  # Fast, cheap, good enough

    def embed_text(self, text: str) -> list[float]:
        """
        Convert ONE piece of text into numbers.

        Parameters:
            text: Any text string (e.g., "refund policy")

        Returns:
            A list of 1536 numbers (the embedding)
            e.g., [0.012, -0.034, 0.078, ..., 0.045]

        Example:
            embedder = Embedder()
            numbers = embedder.embed_text("What is the refund policy?")
            print(len(numbers))  # 1536
            print(numbers[:5])   # [0.012, -0.034, 0.078, 0.091, -0.023]
        """

        # Call OpenAI's embedding API
        # This sends the text to OpenAI's servers
        # OpenAI's model converts it to numbers and sends back
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )

        # Extract the embedding (list of numbers) from the response
        embedding = response.data[0].embedding

        return embedding

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """
        Convert MANY texts into numbers at once (batch processing).
        More efficient than calling embed_text() one by one.

        Parameters:
            texts: List of text strings
                   ["chunk 1 text", "chunk 2 text", "chunk 3 text"]

        Returns:
            List of embeddings (one per text)
            [[0.01, -0.03, ...], [0.05, 0.02, ...], [0.08, -0.01, ...]]

        Why batch?
            Sending 100 texts ONE BY ONE = 100 API calls = slow
            Sending 100 texts AT ONCE = 1 API call = fast!
            Like ordering 10 items at once vs making 10 separate orders.
        """

        # OpenAI allows sending multiple texts in one API call
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        # Extract all embeddings
        embeddings = [item.embedding for item in response.data]

        return embeddings
