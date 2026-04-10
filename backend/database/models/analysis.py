from sqlalchemy import Column, Integer, String, JSON, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.database.connection import Base


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    label = Column(String(300), nullable=True)       # user-defined name
    resume_id = Column(Integer, ForeignKey("resumes.id"), nullable=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=True)
    result = Column(JSON, nullable=False)             # full SkillGapResult JSON
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    resume = relationship("Resume")
    job = relationship("Job")

    def __repr__(self):
        return f"<Analysis id={self.id} label={self.label}>"
