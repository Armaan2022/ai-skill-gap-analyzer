from sqlalchemy import Column, Integer, String
from backend.database.connection import Base

class SkillMapping(Base):
    """
    Optional: map raw extracted phrases -> canonical skill label,
    or store manual overrides / synonyms added by an admin.
    """
    __tablename__ = "skill_mappings"

    id = Column(Integer, primary_key=True, index=True)
    raw_phrase = Column(String(300), nullable=False, index=True)
    canonical_skill = Column(String(300), nullable=False, index=True)

    def __repr__(self):
        return f"<SkillMapping {self.raw_phrase} -> {self.canonical_skill}>"