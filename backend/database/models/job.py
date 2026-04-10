from sqlalchemy import Column, Integer, String, JSON
from backend.database.connection import Base

class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(300), nullable=False)
    description = Column(String, nullable=True)
    skills = Column(JSON, nullable=True)   # canonical job skills list

    def __repr__(self):
        return f"<Job id={self.id} title={self.title}>"