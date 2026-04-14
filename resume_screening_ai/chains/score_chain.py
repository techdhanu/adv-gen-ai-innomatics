import json

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, RunnableLambda

from chains.schemas import ScoreResult
from prompts.score_prompt import get_score_prompt


def build_score_chain(llm: Runnable) -> Runnable:
    parser = PydanticOutputParser(pydantic_object=ScoreResult)
    prompt = get_score_prompt().partial(format_instructions=parser.get_format_instructions())

    def _serialize_input(payload: dict) -> dict:
        return {
            "extracted_resume": json.dumps(payload["extracted_resume"], indent=2),
            "match_result": json.dumps(payload["match_result"], indent=2),
        }

    return (RunnableLambda(_serialize_input) | prompt | llm | parser).with_config(tags=["score"])


