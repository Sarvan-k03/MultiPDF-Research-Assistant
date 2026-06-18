"""
pdf_processor.py
----------------
Handles loading PDFs and splitting them into chunks with rich metadata.
Each chunk carries: source filename, page number, chunk index.
"""

import os
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Configuration ──────────────────────────────────────────────────────────────
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 100     # overlap between adjacent chunks
# ───────────────────────────────────────────────────────────────────────────────


def clean_extracted_text(text: str) -> str:
    """
    Clean extracted PDF text to handle encoding artifacts, ligatures,
    non-printable control characters, and whitespace anomalies.
    """
    if not text:
        return ""

    # Map common PDF ligatures & special characters to standard ASCII equivalents
    replacements = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "ft",
        "\ufb06": "st",
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\u2215": "/",
        "\u2022": "*", # bullet point
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Filter out non-printable or corrupt unicode control characters
    cleaned = []
    for char in text:
        cp = ord(char)
        # Keep printable ASCII, tab, newline, CR, and typical accented chars
        if (32 <= cp <= 126) or char in "\t\n\r" or (128 <= cp <= 255) or cp in [0x2217, 0x2264, 0x2265, 0x00b1, 0x03b1, 0x03b2]:
            cleaned.append(char)
        else:
            if cp < 32 and char not in "\t\n\r":
                continue
            cleaned.append(char)

    text = "".join(cleaned)

    # Normalize whitespaces: replace multiple spaces with a single space, keep newlines
    lines = []
    for line in text.split("\n"):
        line = " ".join(line.split())
        lines.append(line)

    # Reconstruct text filtering excessive consecutive empty lines
    reconstructed = []
    prev_empty = False
    for line in lines:
        if line == "":
            if not prev_empty:
                reconstructed.append("")
                prev_empty = True
        else:
            reconstructed.append(line)
            prev_empty = False

    return "\n".join(reconstructed).strip()


def extract_text_by_page(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from every page of a PDF, preserving page numbers.

    Returns:
        List of dicts: [{"page": int, "text": str, "source": str}, ...]
    """
    pages = []
    filename = Path(pdf_path).name

    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            raw_text = page.get_text("text")
            
            # Use fallback to block-level extraction if page text is empty
            if not raw_text.strip():
                blocks = page.get_text("blocks")
                raw_text = "\n".join([b[4] for b in blocks if b[4].strip()])

            cleaned_text = clean_extracted_text(raw_text)
            if cleaned_text:  # skip blank pages
                pages.append({
                    "page": page_num + 1,   # 1-indexed (human-readable)
                    "text": cleaned_text,
                    "source": filename
                })
        doc.close()
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF '{pdf_path}': {e}")

    return pages


def chunk_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split each page's text into overlapping chunks.
    Metadata (source, page) is preserved per chunk.

    Returns:
        List of chunk dicts: [{"text": str, "source": str, "page": int, "chunk_index": int}, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks = []
    chunk_index = 0

    for page_data in pages:
        text_chunks = splitter.split_text(page_data["text"])
        for chunk_text in text_chunks:
            chunk_text = chunk_text.strip()
            if len(chunk_text) > 50:  # skip tiny fragments
                all_chunks.append({
                    "text": chunk_text,
                    "source": page_data["source"],
                    "page": page_data["page"],
                    "chunk_index": chunk_index,
                })
                chunk_index += 1

    return all_chunks


def process_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Full pipeline: PDF file → list of text chunks with metadata.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        List of chunk dicts ready for embedding and storage.
    """
    pages = extract_text_by_page(pdf_path)
    chunks = chunk_pages(pages)
    return chunks


def process_multiple_pdfs(pdf_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Process multiple PDFs and return all chunks combined.

    Args:
        pdf_paths: List of absolute paths to PDF files.

    Returns:
        Combined list of chunks from all PDFs.
    """
    all_chunks = []
    for path in pdf_paths:
        try:
            chunks = process_pdf(path)
            all_chunks.extend(chunks)
            print(f"  [OK] Processed '{Path(path).name}': {len(chunks)} chunks")
        except Exception as e:
            print(f"  [ERROR] Error processing '{path}': {e}")
    return all_chunks


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pdf = sys.argv[1]
        chunks = process_pdf(pdf)
        print(f"\nTotal chunks: {len(chunks)}")
        print(f"\nSample chunk:")
        if chunks:
            chunk_repr = str(chunks[0])
            safe_repr = chunk_repr.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
            print(safe_repr)
