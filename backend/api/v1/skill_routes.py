from fastapi import APIRouter
from pydantic import BaseModel
from backend.services.skill_service import SkillService

router = APIRouter()
service = SkillService()

class SkillRequest(BaseModel):
    text: str

class SkillGapRequest(BaseModel):
    resume_text: str
    job_text: str

@router.post("/extract-skills")
def extract_skills(req: SkillRequest):
    return service.extract_skills(req.text)

@router.post("/skill-gap")
def skill_gap(req: SkillGapRequest):
    return service.compute_skill_gap(req.resume_text, req.job_text)

