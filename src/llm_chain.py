"""
llm_chain.py
------------
Wraps the Groq API with a citation-forcing prompt.
The prompt instructs the model to cite [Source: file.pdf, Page X] for
every fact it uses, and to refuse to answer outside the provided context.
"""

import os
from typing import List, Dict, Any, Tuple

from groq import Groq
from dotenv import load_dotenv

from src.retriever import retrieve, format_context_for_prompt


# ── Setup ──────────────────────────────────────────────────────────────────────
load_dotenv()
_GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "openai/gpt-oss-120b"
# ───────────────────────────────────────────────────────────────────────────────


SYSTEM_PROMPT = """You are a precise research assistant. Your job is to answer questions 
using ONLY the context provided below. 

RULES you must follow:
1. Cite every piece of information using the format: [Source: filename.pdf, Page X]
2. Place citations immediately after the sentence they support.
3. If the answer spans multiple sources, cite each one separately.
4. If the answer is NOT found in the provided context, respond EXACTLY with:
   "I don't have enough information in the uploaded documents to answer this question."
5. Do NOT use any external knowledge or make things up.
6. Be concise but complete. Use bullet points for multi-part answers.
7. Format your response with proper markdown:
   - Use **bold** for important terms
   - Use `code` for technical terms or formulas
   - Use $$formula$$ for mathematical equations (display math)
   - Use $formula$ for inline math
   - Use ### for subheadings
   - Use - or * for bullet points
   - Use numbered lists (1. 2. 3.) when appropriate

---
CONTEXT FROM UPLOADED DOCUMENTS:
{context}
---

QUESTION: {question}

ANSWER (with citations and proper markdown formatting):"""


def ask(
    question: str,
    k: int = 5,
    filter_sources: List[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Full RAG pipeline: retrieve chunks → build prompt → call Gemini → return answer + sources.

    Args:
        question: The user's question.
        k: Number of context chunks to retrieve.
        filter_sources: Optional list of PDF filenames to restrict search to.

    Returns:
        Tuple of (answer_text, source_chunks_list)
    """
    if not _GROQ_API_KEY or _GROQ_API_KEY == "your_groq_api_key_here":
        return (
            "⚠️ Groq API key not configured. Please add your GROQ_API_KEY to the .env file.",
            []
        )

    # Step 1: Retrieve relevant chunks
    chunks = retrieve(question, k=k, filter_sources=filter_sources)

    if not chunks:
        return (
            "I don't have enough information in the uploaded documents to answer this question. "
            "Please make sure you have uploaded relevant PDF files first.",
            []
        )

    # Step 2: Format context
    context = format_context_for_prompt(chunks)

    # Step 3: Build prompt
    prompt = SYSTEM_PROMPT.format(context=context, question=question)

    # Step 4: Call Groq
    try:
        client = Groq(api_key=_GROQ_API_KEY)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,        # low temperature = more factual
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"⚠️ Error calling Groq API: {str(e)}"

    return answer, chunks


def check_api_key() -> bool:
    """Return True if a real API key is configured."""
    key = os.getenv("GROQ_API_KEY", "")
    return bool(key) and key != "your_groq_api_key_here"
