# ============================================================
# DOCUMENT LOADER — Reads text from PDF, DOCX, TXT files
# ============================================================
#
# What this does:
#   You give it a file → it gives you back the text inside.
#
# Real life analogy:
#   You hand someone a BOOK (PDF file)
#   They read every page out loud and write down all the words
#   They give you back a NOTEBOOK with all the text
#
# We use PyMuPDF (fitz) library to read PDFs.
# Why PyMuPDF? It's fast, works on all PDFs, and extracts text accurately.

import fitz  # PyMuPDF — the library that reads PDFs


def load_pdf(file_path: str) -> list[dict]:
    """
    Read a PDF file and extract text from each page.

    Parameters:
        file_path: Path to the PDF file (e.g., "data/policy.pdf")

    Returns:
        A list of dictionaries, one per page:
        [
            {"page": 1, "text": "Chapter 1: Our refund policy..."},
            {"page": 2, "text": "Shipping information..."},
            ...
        ]

    Real life example:
        load_pdf("company_policy.pdf")
        → [
            {page: 1, text: "Welcome to our company..."},
            {page: 2, text: "Refund policy: customers can return..."},
          ]
    """

    # Open the PDF file
    # fitz.open() is like opening a book — you can now flip through pages
    doc = fitz.open(file_path)

    pages = []

    # Loop through each page
    # Like flipping through a book, one page at a time
    for page_num in range(len(doc)):
        # Get this page
        page = doc[page_num]

        # Extract text from this page
        # .get_text() reads all the words on the page
        text = page.get_text()

        # Clean up the text (remove extra whitespace)
        text = text.strip()

        # Only add if the page has actual text (skip blank pages)
        if text:
            pages.append({
                "page": page_num + 1,  # Page numbers start from 1, not 0
                "text": text,
            })

    # Close the document (like closing a book when done)
    doc.close()

    return pages


def load_text_file(file_path: str) -> list[dict]:
    """
    Read a plain text file (.txt).
    Much simpler than PDF — just read the whole file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return [{"page": 1, "text": text}]


def load_document(file_path: str) -> list[dict]:
    """
    Load any supported document type.
    Decides which loader to use based on file extension.

    Like a RECEPTIONIST who directs you:
      - PDF file? → Go to PDF loader
      - TXT file? → Go to text loader
      - Other?    → Sorry, not supported
    """
    file_path_lower = file_path.lower()

    if file_path_lower.endswith(".pdf"):
        return load_pdf(file_path)
    elif file_path_lower.endswith(".txt"):
        return load_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}. Use PDF or TXT files.")
