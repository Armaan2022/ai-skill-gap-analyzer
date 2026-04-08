from pydantic import BaseModel, ConfigDict
from typing import Optional, Any
from datetime import datetime


class ResumeCreate(BaseModel):
    user_id: Optional[int] = None
    filename: Optional[str] = None
    text: Optional[str] = None


class ResumeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: Optional[int] = None
    filename: Optional[str] = None
    text: Optional[str] = None
    skills: Optional[Any] = None


class JobCreate(BaseModel):
    title: str
    description: Optional[str] = None


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    description: Optional[str] = None
    skills: Optional[Any] = None


class SkillGapFromIdsRequest(BaseModel):
    resume_id: int
    job_id: int


class SkillMappingCreate(BaseModel):
    raw_phrase: str
    canonical_skill: str


class AnalysisSave(BaseModel):
    resume_text: str
    job_title: str
    job_description: str
    result: Any
    label: Optional[str] = None   # custom name; auto-generated if omitted


class AnalysisOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    label: Optional[str] = None
    resume_id: Optional[int] = None
    job_id: Optional[int] = None
    result: Any
    created_at: datetime
