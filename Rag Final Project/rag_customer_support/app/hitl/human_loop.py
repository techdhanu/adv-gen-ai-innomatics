from __future__ import annotations


def should_escalate(answer: str, context: str, query: str) -> bool:
    low_context = len(context.strip()) < 80
    uncertain = any(
        token in answer.lower()
        for token in ["i do not know", "not sure", "insufficient", "cannot find", "unclear"]
    )
    manual_request = any(
        token in query.lower() for token in ["human", "agent", "escalate", "supervisor"]
    )
    return low_context or uncertain or manual_request

