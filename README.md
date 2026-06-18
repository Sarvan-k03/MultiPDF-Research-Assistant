# 🔬 Multi-PDF Research Assistant

> A production-quality RAG (Retrieval-Augmented Generation) application that lets you query multiple research papers simultaneously and receive answers with precise source citations.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat-square&logo=streamlit)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-green?style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini_1.5_Flash-free_tier-orange?style=flat-square&logo=google)

---

## ✨ What Makes This Different

Most "chat with PDF" projects stop at a single document and return answers with no source grounding. This project adds three key features:

| Feature | What it does |
|---------|-------------|
| 📚 **Multi-document** | Upload 3–5 PDFs and query across all simultaneously |
| 📎 **Source citations** | Every answer shows `[Source: paper.pdf, Page X]` with matching text previews |
| 📊 **Evaluation layer** | Automated pipeline measuring source precision & keyword hit rate |

---

## 🏗️ Architecture

```
PDF Files (3-5)
     │
     ▼
[PyMuPDF] → Extract text page-by-page (preserving page numbers)
     │
     ▼
[LangChain RecursiveCharacterTextSplitter] → 800-char chunks, 100 overlap
     │
     ▼
[all-MiniLM-L6-v2] → 384-dim embeddings (local, free)
     │
     ▼
[ChromaDB] → Persist vectors + metadata {source, page, chunk_index}
     │
     ▼  ← User query
[Cosine Similarity Search] → Top-5 chunks
     │
     ▼
[Gemini 1.5 Flash] + Citation-forcing prompt
     │
     ▼
Answer with [Source: file.pdf, Page X] citations
```

---

## 🚀 Quick Start

### 1. Clone & set up environment

```bash
git clone <your-repo-url>
cd "Multi pdf chat assistant"

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Add your Gemini API key

```bash
# Copy the template
copy .env.example .env

# Edit .env and add your key:
# GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Get a free key at → **[aistudio.google.com](https://aistudio.google.com)** (no credit card needed)

### 3. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` 🎉

---

## 📁 Project Structure

```
multi-pdf-chat-assistant/
│
├── app.py                    # Streamlit UI (dark theme, chat, citations)
├── requirements.txt
├── .env                      # Your API key (not in git)
├── .env.example              # Template
│
├── src/
│   ├── pdf_processor.py      # PDF loading + chunking with page metadata
│   ├── embedder.py           # all-MiniLM-L6-v2 embeddings (singleton)
│   ├── vector_store.py       # ChromaDB: add, query, delete, stats
│   ├── retriever.py          # Top-K retrieval + context formatting
│   ├── llm_chain.py          # Gemini API + citation-forcing prompt
│   └── evaluator.py          # Automated evaluation pipeline
│
├── chroma_db/                # Auto-created: persisted vector database
│
├── evaluation/
│   ├── test_questions.json   # 10 test Q&A pairs
│   └── eval_results.md       # Auto-generated evaluation report
│
└── data/uploads/             # Temp storage for uploaded PDFs
```

---

## 💬 Usage

1. **Upload PDFs** — Use the sidebar uploader to select 2–5 related PDF files
2. **Click "Process & Index PDFs"** — Extracts text, embeds chunks, stores in ChromaDB
3. **Ask questions** — Type in the chat box; the assistant retrieves relevant chunks and answers with citations
4. **View sources** — Click "📎 Sources Used" below any answer to see matching excerpts with page numbers and similarity scores

---

## 📊 Evaluation

Run the automated evaluation pipeline:

```bash
python -m src.evaluator
```

This runs 10 test questions through the pipeline and generates `evaluation/eval_results.md` with:
- Source precision per question (did we retrieve from the right doc?)
- Keyword hit rate (does the answer contain expected terms?)
- Summary table + full detailed results

### Sample Evaluation Results

| Metric | Score |
|--------|-------|
| Avg Source Precision | ~85% |
| Avg Keyword Hit Rate | ~80% |
| Test Questions | 10 |

*(Results vary based on your uploaded PDFs)*

---

## 🛠️ Tech Stack

| Component | Library | Notes |
|-----------|---------|-------|
| PDF Parsing | PyMuPDF (fitz) | Fast, preserves page structure |
| Chunking | LangChain | RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers | all-MiniLM-L6-v2 (384-dim, local) |
| Vector DB | ChromaDB | Local, persistent, cosine similarity |
| LLM | Gemini 1.5 Flash | Free tier, citation-instructed prompt |
| UI | Streamlit | Dark theme, glassmorphism design |

---

## ⚙️ Configuration

| Parameter | Default | Location |
|-----------|---------|----------|
| Chunk size | 800 chars | `src/pdf_processor.py` |
| Chunk overlap | 100 chars | `src/pdf_processor.py` |
| Top-K retrieval | 5 | UI slider (3–10) |
| Min similarity | 20% | `src/retriever.py` |
| LLM temperature | 0.1 | `src/llm_chain.py` |

---

## 🔑 Key Design Decisions

**Why `all-MiniLM-L6-v2`?**
Runs locally (no API cost), 80MB model, excellent for English text, normalized embeddings = direct cosine similarity.

**Why chunk overlap?**
Without 100-char overlap, an answer that spans a chunk boundary would be split and neither chunk would retrieve well.

**Why low LLM temperature (0.1)?**
Research Q&A needs factual, consistent answers. Low temperature reduces hallucination and keeps citations accurate.

**Why citation-forcing prompt design?**
The prompt pre-labels each context chunk with its source before the LLM sees it. This makes citation trivial for the model — it just has to copy the label it already saw.

---

## 📝 License

MIT License — free to use, fork, and build on.

---

*Built as a portfolio project demonstrating production RAG patterns: multi-document retrieval, metadata-aware chunking, and grounded generation with citations.*
