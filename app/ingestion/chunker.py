# ============================================================
# CHUNKER — Cuts documents into small, smart pieces
# ============================================================
#
# Why we need chunking:
#   A 100-page document has ~50,000 words.
#   GPT can only handle ~4,000 words at a time.
#   We need to CUT the document into small pieces (chunks).
#   Then we find the RIGHT pieces for each question.
#
# Real life analogy:
#   Full textbook = too big to carry to exam.
#   Study notes = small cards with key info from each chapter.
#   Chunking = creating those study note cards.
#
# We build our OWN recursive splitter (no external library needed):
#   It splits text SMARTLY:
#     First tries to split by paragraphs (\n\n)
#     If still too big, splits by sentences (. )
#     If still too big, splits by words ( )
#   This keeps related sentences TOGETHER!


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Split text into smaller chunks using recursive splitting.

    Parameters:
        text:          The full text to split
        chunk_size:    Max characters per chunk (default 500)
        chunk_overlap: How many characters overlap between chunks (default 50)

    Returns:
        List of text chunks (strings)

    Why chunk_overlap?
        Without overlap:
            Chunk 1: "...refund within 30"
            Chunk 2: "days of purchase..."
            Problem: "30 days" is SPLIT! Search might miss it!

        With overlap:
            Chunk 1: "...refund within 30 days of purchase. Shipping"
            Chunk 2: "30 days of purchase. Shipping takes 3-5 days..."
            Better: "30 days" appears in BOTH chunks!
    """

    # If text is small enough, return as single chunk
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Try splitting by these separators IN ORDER:
    # First by paragraphs, then sentences, then words
    separators = ["\n\n", "\n", ". ", " "]

    chunks = []
    current_chunk = ""

    # Find the best separator (the first one that exists in the text)
    separator = " "  # default: split by words
    for sep in separators:
        if sep in text:
            separator = sep
            break

    # Split text by the separator
    pieces = text.split(separator)

    for piece in pieces:
        # Add separator back (except for spaces)
        piece_with_sep = piece + separator if separator != " " else piece + " "

        # If adding this piece keeps us under the limit, add it
        if len(current_chunk) + len(piece_with_sep) <= chunk_size:
            current_chunk += piece_with_sep
        else:
            # Current chunk is full — save it
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # Start new chunk WITH overlap
            # Take the last 'chunk_overlap' characters from the previous chunk
            if chunk_overlap > 0 and current_chunk:
                overlap_text = current_chunk[-chunk_overlap:]
                current_chunk = overlap_text + piece_with_sep
            else:
                current_chunk = piece_with_sep

    # Don't forget the last chunk!
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def chunk_document(pages: list[dict], chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
    """
    Split a document (list of pages) into chunks WITH metadata.

    Parameters:
        pages: Output from loader.py -> [{"page": 1, "text": "..."}, ...]

    Returns:
        List of chunks with metadata:
        [
            {"text": "chunk text...", "page": 1, "chunk_index": 0},
            {"text": "chunk text...", "page": 1, "chunk_index": 1},
            ...
        ]

    Why metadata?
        When the AI answers a question, we want to say:
        "This answer came from PAGE 3 of your document."
        Metadata = the label that tells us WHERE each chunk came from.
    """

    all_chunks = []
    chunk_index = 0

    for page in pages:
        # Split this page's text into chunks
        text_chunks = chunk_text(page["text"], chunk_size, chunk_overlap)

        for chunk_text_piece in text_chunks:
            all_chunks.append({
                "text": chunk_text_piece,
                "page": page["page"],
                "chunk_index": chunk_index,
            })
            chunk_index += 1

    return all_chunks
