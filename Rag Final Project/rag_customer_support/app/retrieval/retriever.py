from __future__ import annotations

from langchain_chroma import Chroma


def build_retriever(vectorstore: Chroma, top_k: int = 4):
    return vectorstore.as_retriever(search_kwargs={"k": top_k})

