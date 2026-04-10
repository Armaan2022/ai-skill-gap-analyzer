from sqlalchemy import Column, Integer, String, ForeignKey, JSON
from sqlalchemy.orm import relationship
from backend.database.connection import Base

class Resume(Base):
    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    filename = Column(String(300), nullable=True)
    text = Column(String, nullable=True)            # extracted resume text
    skills = Column(JSON, nullable=True)     # list/dict of extracted skills

    user = relationship("User")

    def __repr__(self):
        return f"<Resume id={self.id} user_id={self.user_id}>"