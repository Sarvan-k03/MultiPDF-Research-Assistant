"""
retriever.py
------------
Retrieves the most relevant chunks for a query and formats them
as structured context ready for the LLM prompt.
"""

from typing import List, Dict, Any, Optional

from src.vector_store import query_chunks


# ── Config ─────────────────────────────────────────────────────────────────────
DEFAULT_K = 5               # number of chunks to retrieve
MIN_SCORE_THRESHOLD = 20.0  # drop chunks below 20% similarity
# ───────────────────────────────────────────────────────────────────────────────


def retrieve(
    query: str,
    k: int = DEFAULT_K,
    filter_sources: Optional[List[str]] = None,
    min_score: float = MIN_SCORE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Retrieve and filter top-K chunks for a query.

    Args:
        query: The user's question.
        k: Max number of chunks to retrieve.
        filter_sources: Optionally restrict to specific PDF filenames.
        min_score: Drop chunks below this similarity %.

    Returns:
        List of filtered chunk dicts: [{text, source, page, score}]
    """
    results = query_chunks(query, k=k, filter_sources=filter_sources)

    # Filter out low-relevance chunks
    filtered = [r for r in results if r["score"] >= min_score]

    # Sort by score descending
    filtered.sort(key=lambda x: x["score"], reverse=True)

    return filtered


def format_context_for_prompt(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a labeled context block for the LLM prompt.
    Each chunk is labeled with its source and page number.

    Example output:
        [CONTEXT 1 — Source: paper.pdf, Page 3]
        "...chunk text here..."

        [CONTEXT 2 — Source: paper2.pdf, Page 7]
        "...chunk text here..."

    Args:
        chunks: List of chunk dicts from retrieve().

    Returns:
        Formatted multi-line string.
    """
    if not chunks:
        return "No relevant context found in the uploaded documents."

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        label = f"[CONTEXT {i} — Source: {chunk['source']}, Page {chunk['page']}]"
        parts.append(f"{label}\n\"{chunk['text']}\"")

    return "\n\n".join(parts)
