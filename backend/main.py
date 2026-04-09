from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import DATABASE_URL, CORS_ORIGINS
from backend.database.connection import Base, engine
from backend.api.v1.skill_routes import router as skill_router
from backend.api.v1.db_routes import router as db_router


class SkillService:
    """Thin wrapper around SkillExtractor + SkillGapAnalyzer for app.state."""

    def __init__(self):
        from ml.skills.skill_extractor import SkillExtractor
        from ml.gap.skill_gap_analyzer import SkillGapAnalyzer

        self._extractor = SkillExtractor()
        self._analyzer = SkillGapAnalyzer(extractor=self._extractor)

    def extract_skills(self, text: str) -> list:
        result = self._extractor.extract_skills(text)
        return result.get("skills", [])

    def compute_skill_gap(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        return self._analyzer.analyze(resume_text, jd_text)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create DB tables
    Base.metadata.create_all(bind=engine)

    # Load ML models once
    skill_service = SkillService()
    app.state.skill_service = skill_service
    app.state.extractor = skill_service._extractor
    app.state.analyzer = skill_service._analyzer

    yield


app = FastAPI(title="AI Skill Gap Analyzer", version="1.0.0", lifespan=lifespan)

origins = CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(skill_router, prefix="/api/v1")
app.include_router(db_router, prefix="/api/v1/db")


@app.get("/health")
def health():
    return {"status": "ok"}
