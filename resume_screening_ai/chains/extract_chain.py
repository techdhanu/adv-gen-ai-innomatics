from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable

from chains.schemas import ExtractedResume
from prompts.extract_prompt import get_extract_prompt


def build_extract_chain(llm: Runnable) -> Runnable:
    parser = PydanticOutputParser(pydantic_object=ExtractedResume)
    prompt = get_extract_prompt().partial(format_instructions=parser.get_format_instructions())
    return (prompt | llm | parser).with_config(tags=["extract"])

