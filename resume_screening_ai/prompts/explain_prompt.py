from langchain_core.prompts import PromptTemplate


def get_explain_prompt() -> PromptTemplate:
    template = """
You are an explainable AI assistant for recruiters.
Write a concise explanation for the assigned resume fit score.

Return valid JSON with this exact schema:
{{
  "decision": "Strong Fit|Moderate Fit|Low Fit",
  "strengths": ["string"],
  "gaps": ["string"],
  "recommendation": "string",
  "explanation": "string"
}}

Formatting instructions:
{format_instructions}

Rules:
- Keep explanation under 120 words.
- Mention evidence only from provided inputs.
- Do NOT hallucinate skills or tools.

Job Description:
{job_description}

Extracted Resume Data:
{extracted_resume}

Match Analysis:
{match_result}

Scoring Result:
{score_result}
""".strip()
    return PromptTemplate.from_template(template)

