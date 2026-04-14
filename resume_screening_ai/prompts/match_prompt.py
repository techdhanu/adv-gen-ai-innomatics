from langchain_core.prompts import PromptTemplate


def get_match_prompt() -> PromptTemplate:
    template = """
You are evaluating resume-job alignment.
Use only provided resume extraction and job description text.
Do NOT invent qualifications.

Return valid JSON with this exact schema:
{{
  "matched_skills": ["string"],
  "missing_skills": ["string"],
  "matched_tools": ["string"],
  "missing_tools": ["string"],
  "experience_alignment": "strong|moderate|weak",
  "domain_alignment": "strong|moderate|weak",
  "match_summary": "string"
}}

Formatting instructions:
{format_instructions}

Job Description:
{job_description}

Extracted Resume Data:
{extracted_resume}
""".strip()
    return PromptTemplate.from_template(template)

