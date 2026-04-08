from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from datetime import datetime
from backend.database.connection import get_db
from backend.api.v1.schemas import (
    ResumeCreate, ResumeOut,
    JobCreate, JobOut,
    SkillGapFromIdsRequest,
    AnalysisSave, AnalysisOut,
)
from backend.services.db_service import DBService

router = APIRouter(tags=["Database"])


# ── Resumes ───────────────────────────────────────────────────────────────────

@router.get("/resumes", response_model=list[ResumeOut])
def list_resumes(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    return DBService.list_resumes(db, skip, limit)


@router.post("/resumes", response_model=ResumeOut, status_code=201)
def create_resume(payload: ResumeCreate, request: Request, db: Session = Depends(get_db)):
    data = payload.model_dump(exclude_none=True)
    if data.get("text"):
        data["skills"] = request.app.state.skill_service.extract_skills(data["text"])
    return DBService.create_resume(db, data)


@router.get("/resumes/{resume_id}", response_model=ResumeOut)
def get_resume(resume_id: int, db: Session = Depends(get_db)):
    r = DBService.get_resume(db, resume_id)
    if not r:
        raise HTTPException(status_code=404, detail="Resume not found")
    return r


@router.delete("/resumes/{resume_id}", status_code=204)
def delete_resume(resume_id: int, db: Session = Depends(get_db)):
    if not DBService.delete_resume(db, resume_id):
        raise HTTPException(status_code=404, detail="Resume not found")


# ── Jobs ──────────────────────────────────────────────────────────────────────

@router.get("/jobs", response_model=list[JobOut])
def list_jobs(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    return DBService.list_jobs(db, skip, limit)


@router.post("/jobs", response_model=JobOut, status_code=201)
def create_job(payload: JobCreate, request: Request, db: Session = Depends(get_db)):
    data = payload.model_dump(exclude_none=True)
    if data.get("description"):
        data["skills"] = request.app.state.skill_service.extract_skills(data["description"])
    return DBService.create_job(db, data)


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: int, db: Session = Depends(get_db)):
    j = DBService.get_job(db, job_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")
    return j


@router.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: int, db: Session = Depends(get_db)):
    if not DBService.delete_job(db, job_id):
        raise HTTPException(status_code=404, detail="Job not found")


# ── Skill Gap from stored records ─────────────────────────────────────────────

@router.post("/skill-gap/from-ids")
def skill_gap_from_ids(
    payload: SkillGapFromIdsRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    r = DBService.get_resume(db, payload.resume_id)
    j = DBService.get_job(db, payload.job_id)
    if not r:
        raise HTTPException(status_code=404, detail="Resume not found")
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")
    if not r.text:
        raise HTTPException(status_code=422, detail="Resume has no text to analyze")
    if not j.description:
        raise HTTPException(status_code=422, detail="Job has no description to analyze")
    return request.app.state.skill_service.compute_skill_gap(r.text, j.description)


# ── Saved Analyses ────────────────────────────────────────────────────────────

@router.get("/analyses", response_model=list[AnalysisOut])
def list_analyses(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    return DBService.list_analyses(db, skip, limit)


@router.post("/analyses", response_model=AnalysisOut, status_code=201)
def save_analysis(payload: AnalysisSave, request: Request, db: Session = Depends(get_db)):
    # Persist resume
    resume = DBService.create_resume(db, {"text": payload.resume_text})

    # Persist job
    job = DBService.create_job(db, {
        "title": payload.job_title,
        "description": payload.job_description,
    })

    # Auto-generate label if not provided
    label = payload.label or f"{payload.job_title} — {datetime.utcnow().strftime('%b %d, %Y')}"

    analysis = DBService.save_analysis(db, {
        "label": label,
        "resume_id": resume.id,
        "job_id": job.id,
        "result": payload.result,
    })
    return analysis


@router.get("/analyses/{analysis_id}", response_model=AnalysisOut)
def get_analysis(analysis_id: int, db: Session = Depends(get_db)):
    a = DBService.get_analysis(db, analysis_id)
    if not a:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return a


@router.delete("/analyses/{analysis_id}", status_code=204)
def delete_analysis(analysis_id: int, db: Session = Depends(get_db)):
    if not DBService.delete_analysis(db, analysis_id):
        raise HTTPException(status_code=404, detail="Analysis not found")
