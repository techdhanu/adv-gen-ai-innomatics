from __future__ import annotations

import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI


def get_llm(provider: str, openai_model: str, huggingface_repo_id: str, temperature: float = 0.2):
    provider = provider.lower()

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is missing. Set it in .env or environment variables.")
        return ChatOpenAI(model=openai_model, temperature=temperature)

    if provider == "huggingface":
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
        if not token:
            raise ValueError(
                "HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN) is missing. Set it in .env or environment variables."
            )
        return HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            huggingfacehub_api_token=token,
            task="text-generation",
            streaming=False,
            temperature=temperature,
            max_new_tokens=512,
        )

    raise ValueError("Unsupported provider. Use 'openai' or 'huggingface'.")


