from __future__ import annotations

import re
from typing import Iterable

from langchain_core.documents import Document


def format_docs(docs: Iterable[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def unique_sources(docs: Iterable[Document]) -> list[str]:
    sources = []
    seen = set()
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        if source not in seen:
            seen.add(source)
            sources.append(source)
    return sources


def build_fallback_answer(query: str, context: str, max_points: int = 3) -> str:
    cleaned_context = context.strip()
    if not cleaned_context:
        return "I do not know based on provided documents."

    # Keep meaningful unique lines to avoid repeated chunk output.
    raw_lines = [re.sub(r"\s+", " ", line).strip() for line in cleaned_context.splitlines()]
    lines = [line for line in raw_lines if len(line) > 20]

    unique_lines: list[str] = []
    seen = set()
    for line in lines:
        key = line.casefold()
        if key not in seen:
            seen.add(key)
            unique_lines.append(line)

    # Prioritize lines matching the user query keywords.
    query_terms = [w for w in re.findall(r"[a-zA-Z]+", query.lower()) if len(w) > 3]
    ranked: list[str] = []
    for line in unique_lines:
        line_low = line.lower()
        if any(term in line_low for term in query_terms):
            ranked.append(line)

    if not ranked:
        ranked = unique_lines

    selected = ranked[:max_points]
    if not selected:
        return "I do not know based on provided documents."

    bullets = "\n".join(f"- {line}" for line in selected)
    return f"Based on the documents:\n{bullets}"


