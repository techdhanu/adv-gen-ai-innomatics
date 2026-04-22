from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)

