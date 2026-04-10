from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from backend.database.connection import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    email = Column(String(200), unique=True, index=True, nullable=False)
    hashed_password = Column(String(300), nullable=True)  # if you add auth later

    # Relationships
    # resumes = relationship("Resume", back_populates="user")

    def __repr__(self):
        return f"<User id={self.id} email={self.email}>"