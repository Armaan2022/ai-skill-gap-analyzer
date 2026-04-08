# 🚀 AI Skill Gap Analyzer

An end-to-end machine learning system that analyzes the gap between a candidate’s resume and a job description, identifies missing skills, and provides actionable insights to improve employability.

---

## 📌 Overview

The **AI Skill Gap Analyzer** is a full-stack application that combines:

* Natural Language Processing (NLP)
* Embedding-based semantic similarity
* Backend APIs (FastAPI)
* Database persistence (SQLAlchemy + Alembic)
* (Planned) Frontend dashboard (React)

The system extracts skills from resumes and job descriptions, maps them to canonical skills, and computes a **skill gap score** with detailed explanations.

---

## 🎯 Key Features

### ✅ Implemented

* 🔍 **Skill Extraction (NLP + Embeddings)**

  * Uses spaCy + regex + heuristics
  * Maps phrases to canonical skills using embeddings

* 🧠 **Skill Gap Analysis**

  * Compares resume vs job description
  * Categorizes skills into:

    * Matched
    * Missing
    * Extra
  * Computes weighted score

* ⚡ **FastAPI Backend**

  * `/extract-skills` → extract skills from text
  * `/skill-gap` → compute gap between resume & job

* 🗄️ **Database Layer**

  * SQLAlchemy ORM models
  * JSON storage for skills
  * Alembic migrations configured

* 🧩 **Modular Architecture**

  * `ml/` → ML logic
  * `backend/` → API + DB
  * `pipeline/` → preprocessing scripts

---

## 🏗️ Project Structure

```
ai-skill-gap-analyzer/
│
├── backend/
│   ├── api/v1/
│   │   ├── skill_routes.py
│   │
│   ├── database/
│   │   └── connection.py
│   │
│   ├── models/
│   │   ├── user.py
│   │   ├── resume.py
│   │   ├── job.py
│   │   └── skill_mapping.py
│   │
│   ├── services/
│   │   └── skill_service.py
│   │
│   ├── migrations/   # Alembic
│   └── main.py
│
├── ml/
│   ├── skills/
│   │   ├── skill_extractor.py
│   │
│   ├── gap/
│   │   └── skill_gap_analyzer.py
│
├── pipeline/
│   ├── resume_parser.py
│   ├── jd_parser.py
│   └── process_all.py
│
├── data/
│   ├── resumes/
│   ├── job_descriptions/
│   └── processed/json/
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

### Backend

* **FastAPI**
* **SQLAlchemy**
* **Alembic**
* **Pydantic**

### Machine Learning / NLP

* **spaCy**
* **Sentence Transformers**
* **NumPy / Scikit-learn**

### Database

* **SQLite (development)**
* **PostgreSQL (planned for production)**

### Frontend (Planned)

* **React + TypeScript**

---

## 🧠 How It Works

### 1️⃣ Skill Extraction

* Extracts candidate phrases using:

  * spaCy noun chunks
  * Named entities
  * Regex patterns
* Maps phrases → canonical skills using embeddings

---

### 2️⃣ Skill Gap Analysis

Given:

* Resume text
* Job description text

The system:

1. Extracts skills from both
2. Computes similarity using embeddings
3. Categorizes:

   * ✅ Matched
   * ❌ Missing
   * ➕ Extra
4. Computes weighted score

---

## 🔌 API Endpoints

### 🟢 Extract Skills

```
POST /api/v1/extract-skills
```

**Request**

```json
{
  "text": "Python, React, FastAPI developer"
}
```

---

### 🟢 Skill Gap Analysis

```
POST /api/v1/skill-gap
```

**Request**

```json
{
  "resume_text": "I know Python and React",
  "job_text": "We need Python, React, AWS"
}
```

---

## 🗄️ Database Models

* **User**
* **Resume**
* **Job**
* **SkillMapping**

Skills are stored as:

```
Column(JSON)
```

---

## 🚧 Work in Progress / TODO

### 🔴 Backend

* [ ] CRUD APIs for:

  * Resumes
  * Jobs
  * Skill gap results
* [ ] Store ML outputs in DB
* [ ] Add authentication (JWT)

---

### 🔴 Machine Learning

* [ ] Improve skill extraction accuracy
* [ ] Add custom training dataset
* [ ] Expand skill taxonomy (O*NET integration)
* [ ] Fine-tune embedding models
* [ ] Improve category-aware scoring

---

### 🔴 Database

* [ ] Move to PostgreSQL
* [ ] Add indexing for JSON fields
* [ ] Normalize skills into separate table (optional)

---

### 🔴 Frontend

* [ ] React dashboard
* [ ] Resume upload UI
* [ ] Job input UI
* [ ] Visualization of skill gaps
* [ ] Charts (matched vs missing skills)

---

### 🔴 Deployment

* [ ] Dockerize backend
* [ ] Deploy API (Render / AWS / Railway)
* [ ] CI/CD pipeline
* [ ] Production DB setup

---

## 🧪 Running the Project

### 1️⃣ Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

### 2️⃣ Run Backend

```bash
uvicorn backend.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

### 3️⃣ Run Data Pipeline

```bash
python pipeline/process_all.py
```

---
