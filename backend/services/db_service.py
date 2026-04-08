from sqlalchemy.orm import Session
from backend.database.models.resume import Resume
from backend.database.models.job import Job
from backend.database.models.analysis import Analysis
from typing import Optional, Dict, Any, List


class DBService:
    # ── Resumes ───────────────────────────────────────────────────────────────

    @staticmethod
    def create_resume(db: Session, data: Dict[str, Any]) -> Resume:
        r = Resume(**data)
        db.add(r)
        db.commit()
        db.refresh(r)
        return r

    @staticmethod
    def get_resume(db: Session, resume_id: int) -> Optional[Resume]:
        return db.query(Resume).filter(Resume.id == resume_id).first()

    @staticmethod
    def list_resumes(db: Session, skip: int = 0, limit: int = 20) -> List[Resume]:
        return db.query(Resume).offset(skip).limit(limit).all()

    @staticmethod
    def update_resume_skills(db: Session, resume_id: int, skills: Any) -> Optional[Resume]:
        r = db.query(Resume).filter(Resume.id == resume_id).first()
        if not r:
            return None
        r.skills = skills
        db.commit()
        db.refresh(r)
        return r

    @staticmethod
    def delete_resume(db: Session, resume_id: int) -> bool:
        r = db.query(Resume).filter(Resume.id == resume_id).first()
        if not r:
            return False
        db.delete(r)
        db.commit()
        return True

    # ── Jobs ──────────────────────────────────────────────────────────────────

    @staticmethod
    def create_job(db: Session, data: Dict[str, Any]) -> Job:
        j = Job(**data)
        db.add(j)
        db.commit()
        db.refresh(j)
        return j

    @staticmethod
    def get_job(db: Session, job_id: int) -> Optional[Job]:
        return db.query(Job).filter(Job.id == job_id).first()

    @staticmethod
    def list_jobs(db: Session, skip: int = 0, limit: int = 20) -> List[Job]:
        return db.query(Job).offset(skip).limit(limit).all()

    @staticmethod
    def update_job_skills(db: Session, job_id: int, skills: Any) -> Optional[Job]:
        j = db.query(Job).filter(Job.id == job_id).first()
        if not j:
            return None
        j.skills = skills
        db.commit()
        db.refresh(j)
        return j

    @staticmethod
    def delete_job(db: Session, job_id: int) -> bool:
        j = db.query(Job).filter(Job.id == job_id).first()
        if not j:
            return False
        db.delete(j)
        db.commit()
        return True

    # ── Analyses ──────────────────────────────────────────────────────────────

    @staticmethod
    def save_analysis(db: Session, data: Dict[str, Any]) -> Analysis:
        a = Analysis(**data)
        db.add(a)
        db.commit()
        db.refresh(a)
        return a

    @staticmethod
    def list_analyses(db: Session, skip: int = 0, limit: int = 50) -> List[Analysis]:
        return (
            db.query(Analysis)
            .order_by(Analysis.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    @staticmethod
    def get_analysis(db: Session, analysis_id: int) -> Optional[Analysis]:
        return db.query(Analysis).filter(Analysis.id == analysis_id).first()

    @staticmethod
    def delete_analysis(db: Session, analysis_id: int) -> bool:
        a = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if not a:
            return False
        db.delete(a)
        db.commit()
        return True
