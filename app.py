
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load environment variables BEFORE importing rag_engine.
load_dotenv(override=True)

from rag_engine import (
    build_pdf_index,
    answer_question,
    GeminiConfigurationError,
    PDFProcessingError,
    GeminiRequestError,
)


load_dotenv()

st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1200px; }
      .hero { padding: 1.35rem 1.5rem; border-radius: 18px;
              background: linear-gradient(120deg, rgba(79,70,229,.13), rgba(14,165,233,.10));
              border: 1px solid rgba(128,128,128,.20); margin-bottom: 1rem; }
      .hero h1 { margin: 0 0 .25rem 0; font-size: 2rem; }
      .hero p { margin: 0; opacity: .82; }
      [data-testid="stChatMessage"] { border-radius: 14px; }
      .small-muted { opacity: .7; font-size: .9rem; }
      @media (max-width: 700px) {
        .hero h1 { font-size: 1.55rem; }
        .block-container { padding-left: 1rem; padding-right: 1rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_index" not in st.session_state:
    st.session_state.pdf_index = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None
if "document_fingerprint" not in st.session_state:
    st.session_state.document_fingerprint = None

with st.sidebar:
    st.title("📚 PDF Assistant")
    st.caption("Ask questions grounded in your documents.")
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Text-based PDFs work best. Scanned image PDFs need OCR, which this demo does not include.",
    )
    st.divider()

    st.subheader("Settings")
    top_k = st.slider("Source passages to retrieve", min_value=1, max_value=6, value=4)
    show_sources = st.toggle("Show source passages", value=True)
    st.caption("The Gemini API key is read from your local `.env` file.")
    st.divider()

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("♻️ Remove current PDF", use_container_width=True):
        st.session_state.pdf_index = None
        st.session_state.document_name = None
        st.session_state.document_fingerprint = None
        st.session_state.messages = []
        st.rerun()

st.markdown(
    """
    <div class="hero">
      <h1>Chat with your PDF</h1>
      <p>Upload a document, ask a question, and inspect the passages used to answer it.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

api_key = os.getenv("GEMINI_API_KEY", "").strip()
if not api_key or api_key == "paste_your_gemini_api_key_here":
    st.warning(
        "Gemini API key not configured. Add GEMINI_API_KEY to your `.env` file, "
        "then restart Streamlit. See README.md for setup steps."
    )

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    # The digest avoids rebuilding the vector index on every Streamlit rerun.
    import hashlib
    fingerprint = hashlib.sha256(file_bytes).hexdigest()

    if fingerprint != st.session_state.document_fingerprint:
        with st.spinner("Reading PDF, splitting text, and building the search index…"):
            try:
                index_data = build_pdf_index(
                    file_bytes=file_bytes,
                    filename=uploaded_file.name,
                    api_key=api_key,
                )
                st.session_state.pdf_index = index_data
                st.session_state.document_name = uploaded_file.name
                st.session_state.document_fingerprint = fingerprint
                st.session_state.messages = []
                st.success(
                    f"Indexed **{index_data['page_count']}** pages and "
                    f"**{len(index_data['chunks'])}** text chunks."
                )
            except GeminiConfigurationError as exc:
                st.session_state.pdf_index = None
                st.error(str(exc))
            except PDFProcessingError as exc:
                st.session_state.pdf_index = None
                st.error(str(exc))
            except GeminiRequestError as exc:
                st.session_state.pdf_index = None
                st.error(f"Gemini could not create embeddings: {exc}")
            except Exception as exc:
                st.session_state.pdf_index = None
                st.error(f"Unexpected indexing error: {exc}")
                st.caption("Check the terminal for details. Do not share API keys in screenshots.")
else:
    if st.session_state.pdf_index is not None:
        st.info(f"Current document: **{st.session_state.document_name}**")
    else:
        st.info("Upload a PDF from the sidebar to begin.")

if st.session_state.pdf_index is not None:
    index_data = st.session_state.pdf_index
    st.caption(
        f"**Document:** {st.session_state.document_name} · "
        f"**Pages with text:** {index_data['pages_with_text']} · "
        f"**Chunks:** {len(index_data['chunks'])}"
    )

    # Render conversation history first so it remains visible after every rerun.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and show_sources and message.get("sources"):
                with st.expander("View retrieved source passages"):
                    for i, source in enumerate(message["sources"], start=1):
                        st.markdown(f"**Source {i} · page {source['page']} · similarity {source['score']:.3f}**")
                        st.write(source["text"])
                        if i < len(message["sources"]):
                            st.divider()

    question = st.chat_input("Ask something about this PDF…")
    if question:
        question = question.strip()
        if not question:
            st.warning("Please enter a question.")
        elif not api_key:
            st.error("Configure GEMINI_API_KEY in `.env` before asking questions.")
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching the PDF and asking Gemini…"):
                    try:
                        result = answer_question(
                            question=question,
                            index_data=index_data,
                            api_key=api_key,
                            top_k=top_k,
                        )
                        st.markdown(result["answer"])
                        if show_sources and result["sources"]:
                            with st.expander("View retrieved source passages", expanded=True):
                                for i, source in enumerate(result["sources"], start=1):
                                    st.markdown(
                                        f"**Source {i} · page {source['page']} · "
                                        f"similarity {source['score']:.3f}**"
                                    )
                                    st.write(source["text"])
                                    if i < len(result["sources"]):
                                        st.divider()
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": result["answer"],
                                "sources": result["sources"],
                            }
                        )
                    except GeminiConfigurationError as exc:
                        st.error(str(exc))
                    except GeminiRequestError as exc:
                        st.error(f"Gemini request failed: {exc}")
                        st.caption("Check the API key, model availability, internet connection, and API quota.")
                    except Exception as exc:
                        st.error(f"Unexpected error while answering: {exc}")
                        st.caption("Check the VS Code terminal for diagnostic details.")
else:
    st.markdown("### How it works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Extract**")
        st.write("PyPDF reads text and page numbers from the uploaded PDF.")
    with c2:
        st.markdown("**2. Retrieve**")
        st.write("Text chunks become Gemini embeddings, stored in a FAISS similarity index.")
    with c3:
        st.markdown("**3. Answer**")
        st.write("The most relevant passages are sent to Gemini to generate a grounded response.")

st.markdown("---")
st.markdown(
    '<p class="small-muted">Demo note: answers can still be wrong. Verify important details against the cited page passages.</p>',
    unsafe_allow_html=True,
)
