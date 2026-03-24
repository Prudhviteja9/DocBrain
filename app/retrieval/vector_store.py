# ============================================================
# VECTOR STORE — ChromaDB wrapper
# ============================================================
#
# What this does:
#   1. STORE chunks + embeddings in ChromaDB
#   2. SEARCH for similar chunks when user asks a question
#
# Real life analogy:
#   This is like a LIBRARIAN:
#     - You give the librarian BOOKS (chunks + embeddings)
#     - Librarian organizes them on shelves (ChromaDB)
#     - Later you ask "I need info about refunds"
#     - Librarian finds the RIGHT books and gives them to you
#
# ChromaDB stores everything locally in a folder on your computer.
# No internet needed for searching (only for creating embeddings).

import chromadb
from pathlib import Path


class VectorStore:
    """
    Stores and searches document chunks using ChromaDB.

    Think of this as a SMART LIBRARY:
      - add_chunks()  = putting books on shelves
      - search()      = asking the librarian to find relevant books
      - delete_collection() = clearing a shelf
    """

    def __init__(self, persist_directory: str = None):
        """
        Set up ChromaDB.

        Parameters:
            persist_directory: Where to save the database files.
                              Like choosing which ROOM to use as the library.
                              If None, uses "chroma_data" in project folder.
        """

        if persist_directory is None:
            # Default: save in project folder / chroma_data
            persist_directory = str(Path(__file__).parent.parent.parent / "chroma_data")

        # Create the ChromaDB client
        # PersistentClient = data saved to disk (survives restart)
        # Like buying a FILING CABINET — documents stay even if you leave
        self.client = chromadb.PersistentClient(path=persist_directory)

    def get_or_create_collection(self, collection_name: str = "documents"):
        """
        Get a collection (or create it if it doesn't exist).

        A collection is like a TABLE in DynamoDB:
          "documents" collection = stores all document chunks

        Parameters:
            collection_name: Name of the collection (like table name)

        Returns:
            ChromaDB collection object
        """

        # get_or_create = if collection exists, get it. If not, create it.
        # Like: "Open the 'documents' folder. If it doesn't exist, create it."
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        return collection

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        collection_name: str = "documents",
        source_name: str = "unknown",
    ) -> int:
        """
        Store chunks + embeddings in ChromaDB.

        Parameters:
            chunks:     List of chunk dicts from chunker.py
                       [{"text": "...", "page": 1, "chunk_index": 0}, ...]
            embeddings: List of embeddings from embedder.py
                       [[0.31, 0.72, ...], [0.85, 0.12, ...], ...]
            collection_name: Which collection to store in
            source_name:     Name of the source document (e.g., "policy.pdf")

        Returns:
            Number of chunks stored

        Real life:
            Like handing the librarian 12 study cards and saying:
            "Please file these under 'company documents'"
        """

        collection = self.get_or_create_collection(collection_name)

        # Prepare the data for ChromaDB
        # ChromaDB needs: ids, documents, embeddings, metadatas
        ids = []          # Unique ID for each chunk
        documents = []    # The actual text
        metadatas = []    # Extra info (page number, source)

        for chunk in chunks:
            # Create a unique ID: source_name + chunk_index
            # Like a library card number: "policy.pdf_chunk_003"
            chunk_id = f"{source_name}_chunk_{chunk['chunk_index']:04d}"

            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append({
                "page": chunk["page"],
                "source": source_name,
                "chunk_index": chunk["chunk_index"],
            })

        # ADD to ChromaDB
        # This stores everything: IDs, text, numbers, metadata
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(ids)

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        collection_name: str = "documents",
    ) -> list[dict]:
        """
        Search for chunks similar to the query.

        Parameters:
            query_embedding: The question converted to numbers
            n_results:       How many results to return (default 5)
            collection_name: Which collection to search

        Returns:
            List of results, each containing:
              - text: the chunk text
              - page: which page it came from
              - source: which document
              - distance: how similar (lower = more similar)

        Real life:
            You ask the librarian: "I need info about refunds"
            Librarian finds the 5 most relevant books/pages.
        """

        collection = self.get_or_create_collection(collection_name)

        # QUERY ChromaDB
        # This compares query_embedding with ALL stored embeddings
        # Returns the n_results closest ones
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        # Package results in a clean format
        search_results = []

        # ChromaDB returns lists inside lists (because you can query multiple at once)
        # We only query one, so we take [0] to get the first (only) result set
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                search_results.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "id": results["ids"][0][i],
                })

        return search_results

    def get_collection_stats(self, collection_name: str = "documents") -> dict:
        """
        Get info about a collection.
        Like asking the librarian: "How many books do we have?"
        """

        collection = self.get_or_create_collection(collection_name)

        return {
            "collection_name": collection_name,
            "total_chunks": collection.count(),
        }

    def delete_collection(self, collection_name: str = "documents"):
        """
        Delete an entire collection.
        Like clearing an entire bookshelf.
        WARNING: This cannot be undone!
        """

        try:
            self.client.delete_collection(name=collection_name)
            return True
        except Exception:
            return False

    def list_collections(self) -> list[str]:
        """List all collection names."""

        collections = self.client.list_collections()
        return [c.name for c in collections]
