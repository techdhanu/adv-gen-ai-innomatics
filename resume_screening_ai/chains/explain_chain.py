import json

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, RunnableLambda

from chains.schemas import ExplainResult
from prompts.explain_prompt import get_explain_prompt


def build_explain_chain(llm: Runnable) -> Runnable:
    parser = PydanticOutputParser(pydantic_object=ExplainResult)
    prompt = get_explain_prompt().partial(format_instructions=parser.get_format_instructions())

    def _serialize_input(payload: dict) -> dict:
        return {
            "job_description": payload["job_description"],
            "extracted_resume": json.dumps(payload["extracted_resume"], indent=2),
            "match_result": json.dumps(payload["match_result"], indent=2),
            "score_result": json.dumps(payload["score_result"], indent=2),
        }

    return (RunnableLambda(_serialize_input) | prompt | llm | parser).with_config(tags=["explain"])

