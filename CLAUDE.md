# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered tool that analyzes skill gaps between a candidate's resume and a job description using NLP + sentence embeddings.

## Commands

### Setup
```bash
# Install dependencies (use a virtual environment)
pip install -r requirements.txt

# Download spaCy model (required)
python -m spacy download en_core_web_sm

# Run database migrations
cd backend && alembic upgrade head
```

### Run the Backend
```bash
uvicorn backend.main:app --reload
# API docs available at http://localhost:8000/docs
```

### Run Tests
```bash
# ML skill extractor tests
python -m pytest ml/skills/test_skill_extractor.py -v

# Gap analyzer tests
python -m pytest ml/gap/test_gap_analyzer.py -v
```

### Data Pipeline
```bash
# Process all resume and JD text files in data/
python -m pipeline.process_all
```

## Architecture

The project has three main layers:

### ML Layer (`ml/`)
- **`ml/skills/skill_extractor.py`** — `SkillExtractor` class. Hybrid pipeline: spaCy noun-chunk/entity extraction → candidate phrases → sentence-transformer embeddings (BAAI/bge-small-en) → cosine similarity match against `skills_master.json` canonical skill list. Embeddings are cached in `.npy` files.
- **`ml/gap/skill_gap_analyzer.py`** — `SkillGapAnalyzer` class. Takes resume text + JD text, extracts skills from both, builds a similarity matrix, and classifies each job skill as `matched` / `related` / `missing`. Category weights from `skill_meta.json` adjust the final `weighted_score` (0–100). Cross-category matches are rejected.

### Backend Layer (`backend/`)
- **FastAPI** app in `backend/main.py` with two route groups mounted at `/api/v1`:
  - `skill_routes` — stateless ML endpoints (`POST /extract-skills`, `POST /skill-gap`)
  - `db_routes` — CRUD endpoints for persisting resumes/jobs and running gap analysis from stored IDs
- **`backend/services/skill_service.py`** — thin wrapper around `SkillExtractor` + `SkillGapAnalyzer`
- **`backend/services/db_service.py`** — static SQLAlchemy CRUD methods for Resume and Job models
- **Database**: SQLite by default (`./skillgap.db`). Switch to PostgreSQL by setting `DATABASE_URL` env var. Models: `User`, `Resume`, `Job`, `SkillMapping`. Tables are auto-created on startup via `Base.metadata.create_all`; Alembic is also available for schema migrations.
- **Migrations**: Alembic configured in `backend/alembic.ini`, versions in `backend/migrations/versions/`.

### Pipeline Layer (`pipeline/`)
- Batch processing scripts for `.txt` files in `data/resumes/` and `data/job_descriptions/`. Outputs parsed JSON to `data/processed/json/`. Used for offline data preparation, not the API flow.

## Key Data Files

- **`ml/skills/skills_master.json`** — canonical skill vocabulary; add new skills here
- **`ml/skills/skill_meta.json`** — category mappings and per-category weights used by `SkillGapAnalyzer`
- **`ml/skills/skills_embeddings.npy`** — cached embeddings for `skills_master.json`; delete to force regeneration

## Skill Gap Analysis Output Shape

```python
{
  "resume_skills": [...],
  "job_skills": [...],
  "matched": [...],
  "missing": [...],
  "extra": [...],
  "weighted_score": float,  # 0–100
  "explanation": str,
  "details": [
    {
      "job_skill": str,
      "job_weight": float,
      "category": str,
      "status": "matched|related|missing",
      "best_match": str,
      "similarity": float
    }
  ]
}
```

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./skillgap.db` | Database connection string |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
