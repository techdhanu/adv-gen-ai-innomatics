from typing import List, Literal

from pydantic import BaseModel, Field


class ExtractedResume(BaseModel):
    candidate_name: str = Field(default="Unknown")
    skills: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    experience_years: float = Field(default=0)
    education: str = Field(default="")
    domain_experience: List[str] = Field(default_factory=list)


class MatchResult(BaseModel):
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    matched_tools: List[str] = Field(default_factory=list)
    missing_tools: List[str] = Field(default_factory=list)
    experience_alignment: Literal["strong", "moderate", "weak"] = "weak"
    domain_alignment: Literal["strong", "moderate", "weak"] = "weak"
    match_summary: str = Field(default="")


class ScoreResult(BaseModel):
    fit_score: int = Field(ge=0, le=100)
    skills_score: int = Field(ge=0, le=40)
    tools_score: int = Field(ge=0, le=20)
    experience_score: int = Field(ge=0, le=25)
    domain_score: int = Field(ge=0, le=15)
    score_rationale: str = Field(default="")


class ExplainResult(BaseModel):
    decision: Literal["Strong Fit", "Moderate Fit", "Low Fit"]
    strengths: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    recommendation: str = Field(default="")
    explanation: str = Field(default="")

