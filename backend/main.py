import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.core.config import CORS_ORIGINS
from backend.database.connection import Base, engine
from backend.api.v1.skill_routes import router as skill_router
from backend.api.v1.db_routes import router as db_router

logger = logging.getLogger(__name__)


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


def _load_models(app: FastAPI) -> None:
    """Blocking model load — runs in a thread so the server starts immediately."""
    try:
        logger.info("Loading ML models...")
        svc = SkillService()
        app.state.skill_service = svc
        app.state.extractor = svc._extractor
        app.state.analyzer = svc._analyzer
        app.state.models_ready = True
        logger.info("ML models loaded.")
    except Exception:
        logger.exception("Failed to load ML models")
        app.state.models_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create DB tables synchronously — fast, no blocking concern
    Base.metadata.create_all(bind=engine)

    # Mark models as not ready yet
    app.state.models_ready = False

    # Load models in a background thread so the port opens immediately
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _load_models, app)

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
def health(request: Request):
    ready = getattr(request.app.state, "models_ready", False)
    return {"status": "ok", "models_ready": ready}
