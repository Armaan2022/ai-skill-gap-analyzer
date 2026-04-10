from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["ML"])


def _require_models(request: Request):
    if not getattr(request.app.state, "models_ready", False):
        raise HTTPException(status_code=503, detail="Models are still loading, please retry in a moment.")


class TextPayload(BaseModel):
    text: str


class SkillGapPayload(BaseModel):
    resume_text: str
    job_description: str


@router.post("/extract-skills")
def extract_skills(payload: TextPayload, request: Request):
    _require_models(request)
    skills = request.app.state.extractor.extract_skills(payload.text)
    return {"skills": skills}


@router.post("/skill-gap")
def skill_gap(payload: SkillGapPayload, request: Request):
    _require_models(request)
    return request.app.state.analyzer.analyze(
        payload.resume_text, payload.job_description
    )
