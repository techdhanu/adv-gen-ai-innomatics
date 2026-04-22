from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph

from app.hitl.human_loop import should_escalate
from app.utils.helpers import build_fallback_answer, format_docs


class GraphState(TypedDict, total=False):
    query: str
    docs: list[Any]
    context: str
    answer: str
    generation_error: str
    generation_warning: str
    needs_human: bool
    final_answer: str


def build_graph(retriever, llm):
    prompt = PromptTemplate.from_template(
        """You are a customer support assistant. Use only the given context.
If the answer is not in context, say exactly: I do not know based on provided documents.

Context:
{context}

Question:
{query}

Answer in a concise and helpful way."""
    )
    generation_chain = prompt | llm | StrOutputParser()

    def _normalize_answer(raw: Any) -> str:
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw.strip()
        content = getattr(raw, "content", None)
        if isinstance(content, str):
            return content.strip()
        return str(raw).strip()

    def _context_fallback(query: str, context: str) -> str:
        return build_fallback_answer(query=query, context=context, max_points=3)

    def retrieve_node(state: GraphState) -> GraphState:
        docs = retriever.invoke(state["query"])
        context = format_docs(docs)
        return {"docs": docs, "context": context}

    def generate_node(state: GraphState) -> GraphState:
        try:
            raw_answer = generation_chain.invoke(
                {"query": state["query"], "context": state.get("context", "")}
            )
            answer = _normalize_answer(raw_answer)
            if not answer:
                fallback = _context_fallback(state["query"], state.get("context", ""))
                return {
                    "answer": fallback,
                    "generation_warning": "empty_generation",
                }
            return {"answer": answer}
        except Exception as exc:  # noqa: BLE001
            # Keep app responsive even when provider runtime errors occur.
            fallback = _context_fallback(state["query"], state.get("context", ""))
            return {
                "answer": fallback,
                "generation_warning": f"{type(exc).__name__}: {exc}",
            }

    def decide_node(state: GraphState) -> GraphState:
        no_answer = not state.get("answer", "").strip()
        needs_human = no_answer or bool(state.get("generation_error")) or should_escalate(
            state.get("answer", ""), state.get("context", ""), state["query"]
        )
        return {"needs_human": needs_human}

    def human_node(state: GraphState) -> GraphState:
        return {
            "final_answer": (
                "Escalated to human support: I could not confidently answer from the available documents."
            )
        }

    def finalize_node(state: GraphState) -> GraphState:
        return {"final_answer": state.get("answer", "")}

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("decide", decide_node)
    graph.add_node("human", human_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "decide")

    def route_after_decide(state: GraphState) -> str:
        return "human" if state.get("needs_human") else "finalize"

    graph.add_conditional_edges(
        "decide",
        route_after_decide,
        {"human": "human", "finalize": "finalize"},
    )
    graph.add_edge("human", END)
    graph.add_edge("finalize", END)

    return graph.compile()


