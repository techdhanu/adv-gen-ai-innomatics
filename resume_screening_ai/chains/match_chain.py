import json

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, RunnableLambda

from chains.schemas import MatchResult
from prompts.match_prompt import get_match_prompt


def build_match_chain(llm: Runnable) -> Runnable:
    parser = PydanticOutputParser(pydantic_object=MatchResult)
    prompt = get_match_prompt().partial(format_instructions=parser.get_format_instructions())

    def _serialize_input(payload: dict) -> dict:
        return {
            "job_description": payload["job_description"],
            "extracted_resume": json.dumps(payload["extracted_resume"], indent=2),
        }

    return (RunnableLambda(_serialize_input) | prompt | llm | parser).with_config(tags=["match"])

