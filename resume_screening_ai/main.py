import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from chains.explain_chain import build_explain_chain
from chains.extract_chain import build_extract_chain
from chains.match_chain import build_match_chain
from chains.score_chain import build_score_chain

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"


def _read_text(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8").strip()


def _load_inputs() -> Dict[str, Any]:
    return {
        "job_description": _read_text(DATA_DIR / "job_description.txt"),
        "resumes": {
            "strong": _read_text(DATA_DIR / "strong_resume.txt"),
            "average": _read_text(DATA_DIR / "average_resume.txt"),
            "weak": _read_text(DATA_DIR / "weak_resume.txt"),
        },
    }


def _mock_llm_router(prompt_value: Any) -> str:
    text = prompt_value.to_string() if hasattr(prompt_value, "to_string") else str(prompt_value)
    lowered = text.lower()

    if "expert resume parser" in lowered:
        name = "Unknown"
        for candidate in ["Aarav Mehta", "Neha Rao", "Rohan Sharma"]:
            if candidate.lower() in lowered:
                name = candidate
                break

        skill_bank = [
            "python",
            "sql",
            "statistics",
            "machine learning",
            "feature engineering",
            "data cleaning",
            "data visualization",
        ]
        tool_bank = ["pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "jupyter notebook", "git"]
        skills = [s.title() if s != "sql" else "SQL" for s in skill_bank if s in lowered]
        tools = [t.title() if t not in {"numpy", "git"} else t.upper() if t == "git" else "NumPy" for t in tool_bank if t in lowered]

        experience_years = 0.0
        if "6 months" in lowered and "8 months" in lowered:
            experience_years = 1.2
        elif "3 months" in lowered:
            experience_years = 0.3
        elif "2 months" in lowered:
            experience_years = 0.2

        education = ""
        if "b.tech" in lowered:
            education = "B.Tech Computer Science"
        elif "b.sc" in lowered:
            education = "B.Sc Data Science"
        elif "b.com" in lowered:
            education = "B.Com"

        domain = []
        if "data science" in lowered or "analytics" in lowered:
            domain.append("data science")
        if "ml" in lowered or "machine learning" in lowered:
            domain.append("machine learning")

        return json.dumps(
            {
                "candidate_name": name,
                "skills": skills,
                "tools": tools,
                "experience_years": experience_years,
                "education": education,
                "domain_experience": sorted(set(domain)),
            }
        )

    if "evaluating resume-job alignment" in lowered:
        extracted_blob = lowered.split("extracted resume data:", 1)[-1]
        required_skills = ["python", "sql", "statistics", "machine learning", "data cleaning", "feature engineering", "data visualization"]
        required_tools = ["pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "jupyter notebook", "git"]

        matched_skills = [s.title() if s != "sql" else "SQL" for s in required_skills if s in extracted_blob]
        missing_skills = [s.title() if s != "sql" else "SQL" for s in required_skills if s not in extracted_blob]
        matched_tools = [t.title() if t not in {"numpy", "git"} else t.upper() if t == "git" else "NumPy" for t in required_tools if t in extracted_blob]
        missing_tools = [t.title() if t not in {"numpy", "git"} else t.upper() if t == "git" else "NumPy" for t in required_tools if t not in extracted_blob]

        if "1.2" in extracted_blob:
            experience_alignment = "strong"
        elif "0.3" in extracted_blob:
            experience_alignment = "moderate"
        else:
            experience_alignment = "weak"

        domain_alignment = "strong" if "data science" in extracted_blob else "weak"
        if domain_alignment == "weak" and "machine learning" in extracted_blob:
            domain_alignment = "moderate"

        return json.dumps(
            {
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "matched_tools": matched_tools,
                "missing_tools": missing_tools,
                "experience_alignment": experience_alignment,
                "domain_alignment": domain_alignment,
                "match_summary": "Heuristic match completed for local mock mode.",
            }
        )

    if "strict resume screening scorer" in lowered:
        has_many_skills = lowered.count("missing_skills") and lowered.count("matched_skills")
        fit_score = 55
        skills_score = 20
        tools_score = 10
        experience_score = 15
        domain_score = 10

        if "aarav mehta" in lowered:
            fit_score, skills_score, tools_score, experience_score, domain_score = (89, 36, 18, 22, 13)
        elif "neha rao" in lowered:
            fit_score, skills_score, tools_score, experience_score, domain_score = (66, 27, 13, 14, 12)
        elif "rohan sharma" in lowered:
            fit_score, skills_score, tools_score, experience_score, domain_score = (24, 8, 2, 6, 8)
        elif not has_many_skills:
            fit_score, skills_score, tools_score, experience_score, domain_score = (35, 10, 5, 10, 10)

        return json.dumps(
            {
                "fit_score": fit_score,
                "skills_score": skills_score,
                "tools_score": tools_score,
                "experience_score": experience_score,
                "domain_score": domain_score,
                "score_rationale": "Weighted score based on matched skills/tools, experience, and domain.",
            }
        )

    if "explainable ai assistant for recruiters" in lowered:
        score = 0
        if '"fit_score": 89' in lowered:
            score = 89
        elif '"fit_score": 66' in lowered:
            score = 66
        elif '"fit_score": 24' in lowered:
            score = 24

        if score >= 80:
            decision = "Strong Fit"
            rec = "Shortlist for technical interview."
        elif score >= 50:
            decision = "Moderate Fit"
            rec = "Consider for screening round with targeted upskilling discussion."
        else:
            decision = "Low Fit"
            rec = "Not a strong match for current role requirements."

        return json.dumps(
            {
                "decision": decision,
                "strengths": ["Clear evidence of matched qualifications."],
                "gaps": ["Some required qualifications are missing or weak."],
                "recommendation": rec,
                "explanation": "Decision is based on explicit skills, tools, experience, and domain alignment against the job description.",
            }
        )

    return "{}"


def _build_llm(use_mock: bool, provider: str, model: str, hf_repo_id: str):
    if use_mock:
        return RunnableLambda(_mock_llm_router)

    if provider == "huggingface":
        return HuggingFaceEndpoint(
            repo_id=hf_repo_id,
            temperature=0,
            max_new_tokens=512,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_INFERENCE_API_KEY"),
        )

    return ChatOpenAI(model=model, temperature=0)


def _is_quota_or_rate_limit_error(error: Exception) -> bool:
    message = str(error).lower()
    markers = ["insufficient_quota", "rate limit", "ratelimit", "error code: 429", "quota"]
    return any(marker in message for marker in markers)


def run_pipeline(job_description: str, resume_text: str, label: str, llm) -> Dict[str, Any]:
    extract_chain = build_extract_chain(llm)
    match_chain = build_match_chain(llm)
    score_chain = build_score_chain(llm)
    explain_chain = build_explain_chain(llm)

    extracted = extract_chain.with_config(tags=[label]).invoke({"resume_text": resume_text})
    match_result = match_chain.with_config(tags=[label]).invoke(
        {"job_description": job_description, "extracted_resume": extracted.model_dump()},
    )
    score_result = score_chain.with_config(tags=[label]).invoke(
        {"extracted_resume": extracted.model_dump(), "match_result": match_result.model_dump()},
    )
    explain_result = explain_chain.with_config(tags=[label]).invoke(
        {
            "job_description": job_description,
            "extracted_resume": extracted.model_dump(),
            "match_result": match_result.model_dump(),
            "score_result": score_result.model_dump(),
        },
    )

    return {
        "label": label,
        "candidate_name": extracted.candidate_name,
        "extracted": extracted.model_dump(),
        "match": match_result.model_dump(),
        "score": score_result.model_dump(),
        "explanation": explain_result.model_dump(),
    }


def run_debug_case(llm, strong_resume: str) -> Dict[str, Any]:
    wrong_job_description = (
        "Frontend React Developer role requiring React.js, TypeScript, HTML, CSS, and UI animation."
    )
    return run_pipeline(wrong_job_description, strong_resume, "debug_incorrect_output", llm)


def _print_ranked(results: List[Dict[str, Any]]) -> None:
    ranked = sorted(results, key=lambda item: item["score"]["fit_score"], reverse=True)
    print("\n=== Final Ranking ===")
    for idx, item in enumerate(ranked, start=1):
        print(
            f"{idx}. {item['candidate_name']} ({item['label']}): "
            f"{item['score']['fit_score']}/100 -> {item['explanation']['decision']}"
        )


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="AI Resume Screening System with LangChain + LangSmith tracing")
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "huggingface"],
        help="LLM provider for non-mock runs",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name for real LLM runs")
    parser.add_argument(
        "--hf-repo-id",
        default="mistralai/Mistral-7B-Instruct-v0.1",
        help="Hugging Face model repo id for HuggingFaceEndpoint",
    )
    parser.add_argument("--mock", action="store_true", help="Run with local mock LLM (no API key needed)")
    parser.add_argument("--debug", action="store_true", help="Run one intentionally wrong-case input for tracing")
    parser.add_argument("--no-tracing", action="store_true", help="Disable tracing env defaults")
    args = parser.parse_args()

    if not args.no_tracing and os.getenv("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "resume-screening-ai")
    elif not args.no_tracing:
        print("LANGSMITH_API_KEY not found, running without hosted tracing.")

    inputs = _load_inputs()
    llm = _build_llm(
        use_mock=args.mock,
        provider=args.provider,
        model=args.model,
        hf_repo_id=args.hf_repo_id,
    )
    effective_mode = "mock" if args.mock else args.provider

    results = []
    for label, resume_text in inputs["resumes"].items():
        try:
            result = run_pipeline(inputs["job_description"], resume_text, label, llm)
        except Exception as error:
            if not args.mock and args.provider == "openai" and _is_quota_or_rate_limit_error(error):
                print("OpenAI quota/rate limit issue detected. Falling back to local mock mode.")
                llm = _build_llm(
                    use_mock=True,
                    provider=args.provider,
                    model=args.model,
                    hf_repo_id=args.hf_repo_id,
                )
                effective_mode = "mock_fallback"
                result = run_pipeline(inputs["job_description"], resume_text, label, llm)
            else:
                raise

        result["run_mode"] = effective_mode
        results.append(result)

    _print_ranked(results)

    if args.debug:
        debug_result = run_debug_case(llm, inputs["resumes"]["strong"])
        debug_result["run_mode"] = effective_mode
        print("\n=== Debug Run (Intentionally Wrong JD) ===")
        print(json.dumps(debug_result["score"], indent=2))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "latest_report.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nDetailed report saved to: {output_path}")


if __name__ == "__main__":
    main()



