from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

# Reuse one token variable for Hugging Face Hub + sentence-transformers downloads.
if os.getenv("HUGGINGFACEHUB_API_TOKEN") and not os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = project_root / "data"
    chroma_dir: Path = project_root / "chroma_db"

    provider: str = os.getenv("LLM_PROVIDER", "huggingface").lower()
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    huggingface_repo_id: str = os.getenv("HF_REPO_ID", "mistralai/Mistral-7B-Instruct-v0.1")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    top_k: int = int(os.getenv("TOP_K", "4"))


settings = Settings()

