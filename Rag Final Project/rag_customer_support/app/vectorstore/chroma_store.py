from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document


def build_store(
    docs: list[Document], embeddings, persist_directory: Path, collection_name: str = "support_docs"
) -> Chroma:
    persist_directory.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_directory),
    )


def load_store(embeddings, persist_directory: Path, collection_name: str = "support_docs") -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )

