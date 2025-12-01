from fastapi import FastAPI
from backend.api.v1.skill_routes import router as skill_router

app = FastAPI(
    title="AI Skill Gap Analyzer API",
    version="1.0.0",
    description="Backend for skill extraction + skill gap analysis"
)

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(skill_router, prefix="/api/v1")