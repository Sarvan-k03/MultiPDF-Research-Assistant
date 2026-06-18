"""
embedder.py
-----------
Wraps sentence-transformers all-MiniLM-L6-v2 for text embedding.
Model is loaded once and cached for performance.
"""

from typing import List
from sentence_transformers import SentenceTransformer


# ── Model Config ───────────────────────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"   # 80MB, 384-dim, fast & accurate
_model_instance = None              # module-level cache
# ───────────────────────────────────────────────────────────────────────────────


def get_model() -> SentenceTransformer:
    """
    Return the embedding model (lazy-loaded singleton).
    Loads from HuggingFace on first call; cached afterwards.
    """
    global _model_instance
    if _model_instance is None:
        print(f"  Loading embedding model '{MODEL_NAME}'...")
        _model_instance = SentenceTransformer(MODEL_NAME)
        print(f"  [OK] Model loaded (dim={_model_instance.get_sentence_embedding_dimension()})")
    return _model_instance


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of strings into 384-dimensional vectors.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of float vectors (one per input text).
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=len(texts) > 50,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity ready
    )
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string.

    Args:
        query: The user's question.

    Returns:
        384-dim float vector.
    """
    model = get_model()
    embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding.tolist()
