"""
app.py
------
Streamlit UI for the Multi-PDF Research Assistant.
Features: PDF upload, chat interface, source citations, document management.
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

# ── Add project root to path ───────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pdf_processor import process_multiple_pdfs
from src.vector_store import add_chunks, list_sources, delete_source, get_collection_stats, clear_all
from src.llm_chain import ask, check_api_key

load_dotenv()

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-PDF Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }

/* ── App background ── */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #a0aec0;
}

/* ── Page title ── */
.app-title {
    text-align: center;
    padding: 0.5rem 0 1.5rem 0;
}
.app-title h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.app-title p {
    color: #718096;
    font-size: 0.95rem;
    margin-top: 0.3rem;
}

/* ── Chat messages ── */
.chat-message {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    margin-bottom: 1.2rem;
    animation: fadeIn 0.3s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.chat-avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.user-avatar {
    background: linear-gradient(135deg, #667eea, #764ba2);
}
.bot-avatar {
    background: linear-gradient(135deg, #11998e, #38ef7d);
}
.chat-bubble {
    border-radius: 16px;
    padding: 0.8rem 1.1rem;
    max-width: 90%;
    line-height: 1.6;
    font-size: 0.92rem;
}
.user-bubble {
    background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
    border: 1px solid rgba(102,126,234,0.3);
    color: #e2e8f0;
    margin-left: auto;
}
.bot-bubble {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    color: #e2e8f0;
}

/* ── Source card ── */
.source-card {
    background: rgba(102,126,234,0.08);
    border: 1px solid rgba(102,126,234,0.25);
    border-left: 3px solid #667eea;
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    margin: 0.4rem 0;
    font-size: 0.82rem;
    color: #a0aec0;
    transition: border-color 0.2s;
}
.source-card:hover {
    border-color: rgba(102,126,234,0.5);
}
.source-header {
    color: #667eea;
    font-weight: 600;
    font-size: 0.8rem;
    margin-bottom: 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.score-badge {
    background: rgba(102,126,234,0.2);
    color: #a78bfa;
    padding: 0.1rem 0.5rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
}

/* ── Document pill ── */
.doc-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: rgba(17,153,142,0.15);
    border: 1px solid rgba(17,153,142,0.3);
    color: #38ef7d;
    border-radius: 20px;
    padding: 0.25rem 0.7rem;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 0.2rem;
}

/* ── Stats card ── */
.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 0.8rem;
    text-align: center;
}
.stat-number {
    font-size: 1.8rem;
    font-weight: 700;
    color: #667eea;
}
.stat-label {
    font-size: 0.75rem;
    color: #718096;
    margin-top: 0.1rem;
}

/* ── Input box ── */
.stTextInput input, .stChatInput textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(102,126,234,0.3) !important;
}

/* ── Warning / info banners ── */
.api-warning {
    background: rgba(245,101,101,0.1);
    border: 1px solid rgba(245,101,101,0.3);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: #fc8181;
    font-size: 0.85rem;
    margin-bottom: 1rem;
}
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #4a5568;
}
.empty-state .icon { font-size: 3rem; margin-bottom: 0.5rem; }
.empty-state h3 { color: #718096; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "is_thinking" not in st.session_state:
    st.session_state.is_thinking = False


# ── Helpers ────────────────────────────────────────────────────────────────────
def save_uploaded_files(uploaded_files) -> List[str]:
    """Save Streamlit UploadedFile objects to a temp directory. Returns list of paths."""
    paths = []
    upload_dir = os.path.join(os.path.dirname(__file__), "data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    for uf in uploaded_files:
        dest = os.path.join(upload_dir, uf.name)
        with open(dest, "wb") as f:
            f.write(uf.read())
        paths.append(dest)
    return paths


def ingest_pdfs(pdf_paths: List[str]) -> dict:
    """Run the full ingestion pipeline and return stats."""
    new_paths = [p for p in pdf_paths if Path(p).name not in st.session_state.processed_files]
    if not new_paths:
        return {"skipped": True, "new_files": 0, "total_chunks": 0}

    chunks = process_multiple_pdfs(new_paths)
    n_added = add_chunks(chunks)

    for path in new_paths:
        st.session_state.processed_files.add(Path(path).name)

    return {
        "skipped": False,
        "new_files": len(new_paths),
        "total_chunks": n_added,
    }


def render_chat_message(role: str, content: str, sources: list = None):
    """Render a chat message bubble with optional source citations."""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message" style="flex-direction: row-reverse;">
            <div class="chat-avatar user-avatar">👤</div>
            <div class="chat-bubble user-bubble">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Create two columns: avatar + content
        col1, col2 = st.columns([0.08, 0.92])
        with col1:
            st.markdown("🤖", unsafe_allow_html=True)
        with col2:
            # Render the bot response as markdown for proper formatting (math, code, etc.)
            st.markdown(content)

            if sources:
                with st.expander(f"📎 {len(sources)} Source(s) Used", expanded=False):
                    for src in sources:
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-header">
                                📄 {src['source']} · Page {src['page']}
                                <span class="score-badge">{src['score']}% match</span>
                            </div>
                            <div style="color:#cbd5e0; font-style: italic; line-height:1.5;">
                                "{src['text'][:280]}{'...' if len(src['text']) > 280 else ''}"
                            </div>
                        </div>
                        """, unsafe_allow_html=True)


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Research Assistant")
    st.markdown("---")

    # API key status
    if not check_api_key():
        st.markdown("""
        <div class="api-warning">
        ⚠️ <strong>Gemini API key missing</strong><br>
        Add <code>GEMINI_API_KEY=...</code> to your <code>.env</code> file.
        </div>
        """, unsafe_allow_html=True)

    # ── Upload section ──
    st.markdown("### 📂 Upload PDFs")
    uploaded_files = st.file_uploader(
        label="Select PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload 2–5 related research papers or documents",
        key="pdf_uploader",
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("⚡ Process & Index PDFs", use_container_width=True, type="primary"):
            with st.spinner("Extracting text and building vector index..."):
                paths = save_uploaded_files(uploaded_files)
                result = ingest_pdfs(paths)

            if result["skipped"]:
                st.info("All selected files are already indexed.")
            else:
                st.success(
                    f"✅ Indexed **{result['new_files']}** new file(s)  \n"
                    f"🧩 **{result['total_chunks']}** chunks stored in ChromaDB"
                )

    st.markdown("---")

    # ── Documents in DB ──
    st.markdown("### 📋 Loaded Documents")
    stats = get_collection_stats()
    sources = stats["sources"]

    if sources:
        # Stats row
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{stats['total_documents']}</div>
                <div class="stat-label">Documents</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{stats['total_chunks']}</div>
                <div class="stat-label">Chunks</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Document list with remove buttons
        for src in sources:
            col_name, col_del = st.columns([5, 1])
            with col_name:
                st.markdown(f'<div class="doc-pill">📄 {src}</div>', unsafe_allow_html=True)
            with col_del:
                if st.button("✕", key=f"del_{src}", help=f"Remove {src}"):
                    n = delete_source(src)
                    st.session_state.processed_files.discard(src)
                    st.success(f"Removed {src} ({n} chunks)")
                    st.rerun()

        st.markdown("---")
        if st.button("🗑️ Clear All Documents", use_container_width=True):
            clear_all()
            st.session_state.processed_files.clear()
            st.session_state.messages.clear()
            st.rerun()
    else:
        st.markdown('<div style="color:#4a5568;font-size:0.85rem;">No documents loaded yet.</div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # ── Settings ──
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Chunks to retrieve (K)", min_value=3, max_value=10, value=5,
                      help="More chunks = more context but slower")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages.clear()
        st.rerun()


# ── MAIN AREA ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-title">
    <h1>🔬 Multi-PDF Research Assistant</h1>
    <p>Upload research papers and ask questions — get cited answers from your documents</p>
</div>
""", unsafe_allow_html=True)

# Empty state
if not sources:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">📚</div>
        <h3>No documents loaded</h3>
        <p style="color:#4a5568;font-size:0.9rem;">Upload your PDFs in the sidebar and click "Process & Index PDFs" to get started.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Show active documents as pills
    pills_html = "".join(
        f'<span class="doc-pill">📄 {s}</span>' for s in sources
    )
    st.markdown(
        f'<div style="margin-bottom:1rem;">Searching across: {pills_html}</div>',
        unsafe_allow_html=True
    )

    # ── Chat history ──
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center;padding:2rem;color:#4a5568;">
                <div style="font-size:2rem;">💬</div>
                <p>Ask anything about your uploaded documents.<br>
                <span style="font-size:0.85rem;">Try: "What is the main contribution of the Transformer?"</span></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                render_chat_message(
                    role=msg["role"],
                    content=msg["content"],
                    sources=msg.get("sources"),
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chat input ──
    user_input = st.chat_input(
        placeholder="Ask a question about your documents...",
        disabled=not check_api_key(),
        key="chat_input",
    )

    if user_input:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Get answer
        with st.spinner("🔍 Searching documents and generating answer..."):
            answer, source_chunks = ask(user_input, k=top_k)

        # Append assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": source_chunks,
        })

        st.rerun()
