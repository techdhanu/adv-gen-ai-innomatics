from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdfs(data_dir: Path) -> list[Document]:
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {data_dir}. Add at least one PDF for ingestion."
        )

    docs: list[Document] = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for idx, page in enumerate(pages, start=1):
            page.metadata.setdefault("source", pdf_path.name)
            page.metadata.setdefault("page", idx)
        docs.extend(pages)
    return docs

