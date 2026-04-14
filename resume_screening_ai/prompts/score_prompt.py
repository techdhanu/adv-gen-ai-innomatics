from langchain_core.prompts import PromptTemplate


def get_score_prompt() -> PromptTemplate:
    template = """
You are a strict resume screening scorer.
Use the matching analysis and extracted data to assign a final fit score from 0 to 100.

Scoring guidance:
- Skills match: 40%
- Tools match: 20%
- Experience relevance/years: 25%
- Domain alignment: 15%

Return valid JSON with this exact schema:
{{
  "fit_score": number,
  "skills_score": number,
  "tools_score": number,
  "experience_score": number,
  "domain_score": number,
  "score_rationale": "string"
}}

Formatting instructions:
{format_instructions}

Rules:
- Scores must be integers.
- fit_score must be consistent with component scores.
- Do NOT award points for missing qualifications.

Extracted Resume Data:
{extracted_resume}

Match Analysis:
{match_result}
""".strip()
    return PromptTemplate.from_template(template)



