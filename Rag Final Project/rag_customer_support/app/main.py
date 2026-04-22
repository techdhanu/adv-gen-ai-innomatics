from __future__ import annotations

from pathlib import Path
import shutil
import sys

import streamlit as st

# Ensure absolute imports like `from app...` work when Streamlit runs from app/.
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.embeddings.embedder import get_embeddings
from app.ingestion.chunking import chunk_documents
from app.ingestion.loader import load_pdfs
from app.llm.generator import get_llm
from app.retrieval.retriever import build_retriever
from app.utils.helpers import unique_sources
from app.vectorstore.chroma_store import build_store, load_store
from app.workflow.langgraph_flow import build_graph


st.set_page_config(page_title="RAG Customer Support", page_icon="🤖", layout="wide")


def index_documents(force_reindex: bool = False):
    docs = load_pdfs(settings.data_dir)
    chunked = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    embeddings = get_embeddings(settings.embedding_model)
    marker = settings.chroma_dir / "chroma.sqlite3"

    if force_reindex:
        # Rebuild index from scratch to avoid incompatible persisted Chroma metadata.
        if settings.chroma_dir.exists():
            shutil.rmtree(settings.chroma_dir, ignore_errors=True)
        vectorstore = build_store(chunked, embeddings, settings.chroma_dir)
    else:
        if marker.exists():
            try:
                vectorstore = load_store(embeddings, settings.chroma_dir)
            except Exception:
                # Fall back to a clean rebuild if existing on-disk index is incompatible.
                shutil.rmtree(settings.chroma_dir, ignore_errors=True)
                vectorstore = build_store(chunked, embeddings, settings.chroma_dir)
        else:
            vectorstore = build_store(chunked, embeddings, settings.chroma_dir)

    retriever = build_retriever(vectorstore, top_k=settings.top_k)
    llm = get_llm(settings.provider, settings.openai_model, settings.huggingface_repo_id)
    graph = build_graph(retriever, llm)
    return graph


def init_state():
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []


def main():
    init_state()

    st.title("RAG-Based Customer Support Assistant")
    st.caption("LangChain + ChromaDB + LangGraph + HITL")

    with st.sidebar:
        st.subheader("Pipeline Settings")
        st.write(f"Provider: `{settings.provider}`")
        st.write(f"Embedding model: `{settings.embedding_model}`")
        st.write(f"Top-K retrieval: `{settings.top_k}`")

        if st.button("Build / Refresh Index", type="primary"):
            try:
                st.session_state.graph = index_documents(force_reindex=True)
                st.success("Vector index refreshed.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Indexing failed: {exc}")

    data_dir = Path(settings.data_dir)
    pdf_count = len(list(data_dir.glob("*.pdf")))
    st.info(f"PDF files found in data/: {pdf_count}")

    if st.session_state.graph is None:
        try:
            st.session_state.graph = index_documents(force_reindex=False)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Initialize pipeline first. Reason: {exc}")

    query = st.text_input("Ask your question")
    if st.button("Get Answer"):
        if not query.strip():
            st.warning("Please enter a query.")
            return
        if st.session_state.graph is None:
            st.error("Pipeline not initialized. Add PDF(s) in data/ and build index.")
            return

        with st.spinner("Thinking..."):
            try:
                result = st.session_state.graph.invoke({"query": query})
            except Exception as exc:  # noqa: BLE001
                st.error(f"Query failed: {type(exc).__name__}: {exc}")
                st.info("Try clicking 'Build / Refresh Index' and then ask again.")
                return

            docs = result.get("docs", [])
            st.session_state.last_sources = unique_sources(docs)

        st.subheader("Response")
        st.write(result.get("final_answer", "No answer generated."))

        if result.get("generation_warning"):
            st.caption(f"Generation fallback used: {result['generation_warning']}")

        if result.get("needs_human"):
            st.error("HITL Triggered: Escalation recommended.")

        if st.session_state.last_sources:
            st.subheader("Sources")
            for src in st.session_state.last_sources:
                st.write(f"- {src}")


if __name__ == "__main__":
    main()

