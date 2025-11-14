import os
import tempfile
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from pdf2image import convert_from_path
import pytesseract


# =========================
# 1. CONFIG
# =========================

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please set it in your environment variables.")

client = OpenAI(api_key=api_key)


@st.cache_resource
def load_embedder():
    # Multilingual model (Arabic + English)
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# =========================
# 2. FILE LOADING & CHUNKING
# =========================

def load_file(uploaded_file) -> str:
    """Read PDF/TXT/CSV into one big text string. Uses OCR as a fallback for scanned PDFs. LLL"""
    suffix = uploaded_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    text = ""
    try:
        if suffix == "pdf":
            # 1) Try regular text extraction
            reader = PdfReader(tmp_path)
            pages_text = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                pages_text.append(page_text)
            text = "\n\n".join(pages_text)

            # 2) If no text → try OCR
            if not text.strip():
                try:
                    images = convert_from_path(tmp_path)
                    ocr_pages = []
                    for img in images:
                        page_txt = pytesseract.image_to_string(img, lang="eng+ara")
                        ocr_pages.append(page_txt)
                    text = "\n\n".join(ocr_pages)
                except Exception:
                    # OCR failed (e.g. no Poppler/Tesseract on server)
                    text = ""

        elif suffix in ("txt", "md"):
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        elif suffix == "csv":
            df = pd.read_csv(tmp_path)
            text = df.to_csv(index=False)

        else:
            raise ValueError("Unsupported file type. Use PDF, TXT, or CSV.")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Very simple word-based chunking.
    chunk_size = number of words per chunk
    overlap = how many words to overlap between consecutive chunks
    """
    words = text.split()
    chunks = []
    if not words:
        return chunks

    step = max(chunk_size - overlap, 1)
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

    return chunks


# =========================
# 3. VECTOR STORE (IN-MEMORY)
# =========================

def build_vector_store(chunks: List[str]):
    """Return (embeddings, chunks) where embeddings is a numpy array."""
    embedder = load_embedder()
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    return embeddings, chunks


def search_chunks(
    query: str,
    embeddings: np.ndarray,
    chunks: List[str],
    top_k: int = 8,
) -> List[Tuple[str, float]]:
    """Return top_k (chunk, similarity) for the query."""
    embedder = load_embedder()
    q_emb = embedder.encode([query], convert_to_numpy=True)[0]

    # Cosine similarity
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb)
    sims = embeddings @ q_emb / (norms + 1e-10)

    top_idx = np.argsort(sims)[::-1][:top_k]
    results = [(chunks[i], float(sims[i])) for i in top_idx]
    return results


# =========================
# 4. LLM CALL
# =========================

def generate_answer(question: str, retrieved_chunks: List[Tuple[str, float]]) -> str:
    """Call OpenAI ChatCompletion with context from retrieved chunks."""
    context_text = "\n\n---\n\n".join([c for c, _ in retrieved_chunks]) or "No context available."

    # STRICT: answer ONLY from context, but try to explain concepts if they appear there.
    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the provided context, "
        "which comes from an uploaded document.\n\n"
        "Rules:\n"
        "1. Treat the context as your only source of knowledge.\n"
        "2. If the context clearly discusses the topic of the question, summarize and explain "
        "   the concept in your own words, even if the exact wording isn't present.\n"
        "3. If the context does NOT contain any relevant information about the question, "
        "   answer exactly:\n"
        "   \"I am not sure, this does not appear in the uploaded document.\"\n"
        "4. Do NOT use any outside knowledge beyond the context.\n"
        "5. Keep the answer short, clear, and in the same language as the question."
    )

    user_prompt = f"Context:\n{context_text}\n\nQuestion:\n{question}\n\nAnswer:"

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or gpt-4o / any deployed model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


# =========================
# 5. STREAMLIT APP
# =========================

def init_session_state():
    if "chunks" not in st.session_state:
        st.session_state.chunks = None
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "file_name" not in st.session_state:
        st.session_state.file_name = None
    if "history" not in st.session_state:
        st.session_state.history = []  # list of dicts: {question, answer, sources}


def main():
    st.set_page_config(page_title="RAG Doc Assistant", page_icon="📄", layout="wide")
    init_session_state()

    st.title("📄 RAG – Personal Knowledge Assistant")
    st.write(
        "Upload a small PDF / TXT / CSV file, ask questions about it, "
        "and the app will answer using Retrieval-Augmented Generation (RAG)."
    )

    # ---------- Sidebar ----------
    st.sidebar.header("1️⃣ Upload Document")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a PDF / TXT / CSV",
        type=["pdf", "txt", "csv"],
    )

    chunk_size = st.sidebar.slider("Chunk size (words)", 200, 1000, 500, step=50)
    overlap = st.sidebar.slider("Chunk overlap (words)", 0, 400, 100, step=50)

    # ✅ Work with the file only inside this block
    if uploaded_file is not None:
        # If new file (different name) → rebuild everything
        if uploaded_file.name != st.session_state.file_name:
            with st.spinner("Reading & chunking file..."):
                try:
                    text = load_file(uploaded_file)
                except Exception:
                    st.error(
                        "Error while reading the file. "
                        "If this is a scanned PDF, OCR may not be supported on this server."
                    )
                    st.stop()

                if not text or not text.strip():
                    st.error(
                        "Could not extract any text from this file.\n\n"
                        "Tip: Use a PDF with selectable text or upload a TXT/CSV file. "
                        "Scanned PDFs may not be supported in this demo."
                    )
                    st.stop()

                chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

                if not chunks:
                    st.error("No chunks could be created from this file.")
                    st.stop()

                embeddings, chunks = build_vector_store(chunks)
                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.session_state.file_name = uploaded_file.name
                st.session_state.history = []  # reset history for new file

        # Use what’s in session_state even if sliders change
        if st.session_state.chunks is not None:
            st.sidebar.success(f"Loaded file: {st.session_state.file_name}")
            st.sidebar.write(f"Chunks created: **{len(st.session_state.chunks)}**")
    else:
        st.session_state.file_name = None
        st.session_state.chunks = None
        st.session_state.embeddings = None
        st.session_state.history = []

    # ---------- Main layout ----------
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("2️⃣ Ask a Question")
        if st.session_state.chunks is None:
            st.info("Upload a document first to start asking questions.")
        else:
            question = st.text_input(
                "Your question:",
                placeholder="e.g., What is the main idea in this document?",
            )

            if st.button("Get Answer", type="primary") and question.strip():
                with st.spinner("Thinking..."):
                    results = search_chunks(
                        query=question,
                        embeddings=st.session_state.embeddings,
                        chunks=st.session_state.chunks,
                        top_k=8,   # more chunks for big books
                    )
                    answer = generate_answer(question, results)
                    # results = search_chunks(
                    #     query=question,
                    #     embeddings=st.session_state.embeddings,
                    #     chunks=st.session_state.chunks,
                    #     top_k=8,   # more chunks for big books
                    # )

                    # # 🔒 HARD GATE: if similarity is too low, treat it as unrelated to the document
                    # max_score = max((score for _, score in results), default=0.0)
                    # SIM_THRESHOLD = 0.3  # much lower; ML questions will pass

                    # if max_score < SIM_THRESHOLD:
                    #     answer = (
                    #         "This question does not seem to be answered in the uploaded document, "
                    #         "so I cannot answer it from this file."
                    #     )
                    # else:
                    #     answer = generate_answer(question, results)

                    st.session_state.history.append(
                        {
                            "question": question,
                            "answer": answer,
                            "sources": results,
                        }
                    )

            if st.session_state.history:
                last = st.session_state.history[-1]
                st.markdown("### ✅ Answer")
                st.write(last["answer"])

                st.markdown("### 📚 Retrieved Chunks")
                for i, (chunk_content, score) in enumerate(last["sources"], start=1):
                    with st.expander(f"Chunk {i} (similarity: {score:.3f})"):
                        st.write(chunk_content)

    with col_right:
        st.subheader("3️⃣ Q&A History")
        if not st.session_state.history:
            st.write("No questions yet. Ask something about your document!")
        else:
            for i, item in enumerate(reversed(st.session_state.history), start=1):
                st.markdown(f"**Q{i}:** {item['question']}")
                st.markdown(f"**A{i}:** {item['answer']}")
                st.markdown("---")


if __name__ == "__main__":
    main()
