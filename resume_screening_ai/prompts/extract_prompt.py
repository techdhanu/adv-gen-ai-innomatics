from langchain_core.prompts import PromptTemplate


def get_extract_prompt() -> PromptTemplate:
    template = """
You are an expert resume parser.
Extract only information explicitly present in the resume.
Do NOT assume missing details.

Return valid JSON with this exact schema:
{{
  "candidate_name": "string",
  "skills": ["string"],
  "tools": ["string"],
  "experience_years": number,
  "education": "string",
  "domain_experience": ["string"]
}}

Formatting instructions:
{format_instructions}

Rules:
- If candidate name is missing, use "Unknown".
- If experience is unclear, estimate conservatively from explicit dates and mention only a numeric value.
- Keep skills/tools concise and deduplicated.

Resume:
{resume_text}
""".strip()
    return PromptTemplate.from_template(template)



