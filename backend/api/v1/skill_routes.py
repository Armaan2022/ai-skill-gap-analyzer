from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(tags=["ML"])


class TextPayload(BaseModel):
    text: str


class SkillGapPayload(BaseModel):
    resume_text: str
    job_description: str


@router.post("/extract-skills")
def extract_skills(payload: TextPayload, request: Request):
    skills = request.app.state.extractor.extract_skills(payload.text)
    return {"skills": skills}


@router.post("/skill-gap")
def skill_gap(payload: SkillGapPayload, request: Request):
    return request.app.state.analyzer.analyze(
        payload.resume_text, payload.job_description
    )
