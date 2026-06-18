"""
vector_store.py
---------------
ChromaDB interface: add, query, and manage document chunks.
Stores text + embeddings + metadata (source, page, chunk_index).
"""

import os
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings

from src.embedder import embed_texts, embed_query


# ── Config ─────────────────────────────────────────────────────────────────────
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
COLLECTION_NAME = "pdf_research"
# ───────────────────────────────────────────────────────────────────────────────


def get_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client (creates ./chroma_db if needed)."""
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_collection():
    """Get or create the main ChromaDB collection."""
    client = get_client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )
    return collection


def add_chunks(chunks: List[Dict[str, Any]]) -> int:
    """
    Add text chunks to ChromaDB with their embeddings and metadata.

    Args:
        chunks: List of chunk dicts with keys: text, source, page, chunk_index

    Returns:
        Number of chunks added.
    """
    if not chunks:
        return 0

    collection = get_collection()

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    ids, documents, metadatas, embeds = [], [], [], []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        uid = f"{chunk['source']}_p{chunk['page']}_c{chunk['chunk_index']}"
        ids.append(uid)
        documents.append(chunk["text"])
        metadatas.append({
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_index": chunk["chunk_index"],
        })
        embeds.append(emb)

    # ChromaDB upsert avoids duplicate errors on re-upload
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeds,
    )
    return len(ids)


def query_chunks(
    query: str,
    k: int = 5,
    filter_sources: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-K most relevant chunks for a query.

    Args:
        query: The user's question.
        k: Number of results to return.
        filter_sources: Optional list of filenames to restrict search to.

    Returns:
        List of result dicts: [{text, source, page, score}, ...]
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    query_emb = embed_query(query)

    where_filter = None
    if filter_sources and len(filter_sources) == 1:
        where_filter = {"source": filter_sources[0]}
    elif filter_sources and len(filter_sources) > 1:
        where_filter = {"source": {"$in": filter_sources}}

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=min(k, collection.count()),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Convert cosine distance → similarity score (0-100%)
            similarity = round((1 - dist) * 100, 1)
            output.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", 0),
                "score": similarity,
            })

    return output


def delete_source(filename: str) -> int:
    """
    Delete all chunks belonging to a specific PDF file.

    Args:
        filename: The PDF filename (e.g. "paper.pdf")

    Returns:
        Number of deleted chunks.
    """
    collection = get_collection()
    results = collection.get(where={"source": filename})
    if results["ids"]:
        collection.delete(ids=results["ids"])
        return len(results["ids"])
    return 0


def list_sources() -> List[str]:
    """Return a sorted list of unique source filenames in the collection."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    sources = sorted(set(m["source"] for m in all_meta if "source" in m))
    return sources


def get_collection_stats() -> Dict[str, Any]:
    """Return basic stats about what's in ChromaDB."""
    collection = get_collection()
    total = collection.count()
    sources = list_sources()
    return {
        "total_chunks": total,
        "total_documents": len(sources),
        "sources": sources,
    }


def clear_all() -> None:
    """Delete the entire collection (wipe all documents)."""
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  [OK] All documents cleared from ChromaDB.")
    except Exception:
        pass
