from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.preprocessing import normalize
from pathlib import Path
import json
from ml.skills.skill_extractor import SkillExtractor

SKILL_META_PATH = Path(__file__).resolve().parents[1] / "skills" / "skill_meta.json"

DEFAULT_CATEGORY_WEIGHTS = {
    "programming_languages": 1.1,
    "machine_learning": 1.2,
    "deep_learning": 1.3,
    "cloud": 1.4,
    "databases": 1.1,
    "devops": 1.3,
    "frontend": 1.0,
    "backend": 1.2,
    "data_engineering": 1.2,
    "testing": 1.0,
    "security": 1.2,
    "architecture": 1.1,
    "analytics": 1.0,
    "mobile": 1.0,
    "tools": 0.9,
    "soft_skills": 0.6,
    "default": 1.0,
}

HEURISTIC_CATEGORY_MAP = [
    (["python", "java", "javascript", "typescript", "c++", "go", "rust", "kotlin", "swift", "ruby", "php", "scala"], "programming_languages"),
    (["tensorflow", "pytorch", "keras", "neural", "deep learning", "transformer", "llm", "onnx", "hugging face"], "deep_learning"),
    (["machine learning", "ml", "scikit", "xgboost", "lightgbm", "catboost", "feature engineering"], "machine_learning"),
    (["aws", "azure", "gcp", "cloud", "lambda", "ec2", "s3", "vertex", "cloud run"], "cloud"),
    (["docker", "kubernetes", "helm", "ci/cd", "terraform", "ansible", "argocd", "jenkins"], "devops"),
    (["sql", "postgres", "mysql", "mongodb", "redis", "dynamo", "cassandra", "snowflake", "bigquery"], "databases"),
    (["react", "angular", "vue", "svelte", "next.js", "frontend", "html", "css", "tailwind", "webpack", "vite"], "frontend"),
    (["api", "rest", "graphql", "grpc", "backend", "express", "fastapi", "django", "flask", "spring"], "backend"),
    (["spark", "hadoop", "airflow", "kafka", "etl", "data pipeline", "flink", "dbt", "dagster"], "data_engineering"),
    (["pytest", "jest", "cypress", "selenium", "playwright", "unit test", "tdd", "bdd"], "testing"),
    (["security", "owasp", "penetration", "oauth", "jwt", "encryption", "zero trust", "gdpr"], "security"),
    (["system design", "distributed", "microservices", "cqrs", "event-driven", "domain-driven", "design pattern"], "architecture"),
    (["tableau", "power bi", "looker", "metabase", "business intelligence", "statistics"], "analytics"),
    (["react native", "flutter", "ios", "android", "mobile"], "mobile"),
    (["jira", "confluence", "leadership", "communication", "teamwork", "agile", "scrum"], "soft_skills"),
]


class SkillGapAnalyzer:
    """
    Category-aware Skill Gap Analyzer.
    - Loads skill_meta.json at init for O(1) category/weight/related lookups
    - Cross-category matches are penalized
    - Related skills give bonus partial credit
    - Returns weighted score 0-100 + full per-skill breakdown
    """

    def __init__(
        self,
        extractor: Optional[SkillExtractor] = None,
        category_weights: Optional[Dict[str, float]] = None,
        match_threshold: float = 0.70,
        related_threshold: float = 0.50,
    ):
        self.extractor = extractor or SkillExtractor()
        self.match_threshold = match_threshold
        self.related_threshold = related_threshold
        self.category_weights = {**DEFAULT_CATEGORY_WEIGHTS, **(category_weights or {})}

        self._skill_category: Dict[str, str] = {}
        self._skill_weight: Dict[str, float] = {}
        self._skill_related: Dict[str, List[str]] = {}
        self._load_skill_meta()

    def _load_skill_meta(self):
        if not SKILL_META_PATH.exists():
            return
        try:
            meta = json.loads(SKILL_META_PATH.read_text(encoding="utf-8"))
        except Exception:
            return

        for canonical, info in meta.items():
            cat = info.get("category", "default")
            w = float(info.get("weight", self.category_weights.get(cat, 1.0)))
            self._skill_category[canonical] = cat
            self._skill_weight[canonical] = w
            self._skill_related[canonical] = info.get("related", [])

    def _skill_category_and_weight(self, skill: str) -> Tuple[str, float]:
        if skill in self._skill_category:
            return self._skill_category[skill], self._skill_weight[skill]

        low = skill.lower()
        for tokens, category in HEURISTIC_CATEGORY_MAP:
            if any(tok in low for tok in tokens):
                return category, float(self.category_weights.get(category, self.category_weights["default"]))

        return "default", float(self.category_weights["default"])

    def _embed_list(self, texts: List[str]) -> np.ndarray:
        if not texts:
            dim = self.extractor.skill_embs.shape[1] if hasattr(self.extractor, "skill_embs") else 384
            return np.zeros((0, dim), dtype=np.float32)
        arr = self.extractor.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        arr = arr.astype(np.float32)
        return normalize(arr, axis=1)

    def _related_bonus(self, job_skill: str, resume_skill: str) -> float:
        related_of_job = self._skill_related.get(job_skill, [])
        related_of_resume = self._skill_related.get(resume_skill, [])
        if resume_skill in related_of_job or job_skill in related_of_resume:
            return 0.10
        return 0.0

    def analyze(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """Analyze skill gap between a resume and job description."""
        resume_res = self.extractor.extract_skills(resume_text)
        jd_res = self.extractor.extract_skills(jd_text)

        resume_skills: List[str] = resume_res.get("skills", [])
        jd_skills: List[str] = jd_res.get("skills", [])

        jd_embs = self._embed_list(jd_skills)
        res_embs = self._embed_list(resume_skills)

        if jd_embs.shape[0] > 0 and res_embs.shape[0] > 0:
            sims = np.matmul(jd_embs, res_embs.T)
        else:
            sims = np.zeros((len(jd_skills), len(resume_skills)), dtype=np.float32)

        details = []
        matched: List[str] = []
        missing: List[str] = []
        total_weight = 0.0
        score_obtained = 0.0

        for i, job_skill in enumerate(jd_skills):
            category, job_weight = self._skill_category_and_weight(job_skill)
            total_weight += job_weight

            best_match: Optional[str] = None
            best_sim = 0.0

            if resume_skills and sims.shape[1] > 0:
                jrow = sims[i]
                idx = int(np.argmax(jrow))
                raw_sim = float(jrow[idx])
                candidate = resume_skills[idx]

                job_cat, _ = self._skill_category_and_weight(job_skill)
                res_cat, _ = self._skill_category_and_weight(candidate)

                if job_cat == res_cat or job_cat == "default" or res_cat == "default":
                    bonus = self._related_bonus(job_skill, candidate)
                    best_sim = min(raw_sim + bonus, 1.0)
                    best_match = candidate
                else:
                    best_sim = 0.0
                    best_match = None

            if best_sim >= self.match_threshold:
                status = "matched"
                score_obtained += job_weight * best_sim
                matched.append(job_skill)
            elif best_sim >= self.related_threshold:
                status = "related"
                score_obtained += job_weight * (0.5 * best_sim)
                missing.append(job_skill)
            else:
                status = "missing"
                missing.append(job_skill)

            details.append({
                "job_skill": job_skill,
                "job_weight": job_weight,
                "category": category,
                "status": status,
                "best_match": best_match,
                "similarity": round(best_sim, 4),
            })

        matched_resume_skills = {
            d["best_match"] for d in details
            if d["best_match"] is not None and d["similarity"] >= self.related_threshold
        }
        extra = [r for r in resume_skills if r not in matched_resume_skills]

        weighted_score = round(float(score_obtained / total_weight * 100), 2) if total_weight > 0 else 0.0

        top_missing = sorted(
            [d for d in details if d["status"] in ("missing", "related")],
            key=lambda d: d["job_weight"],
            reverse=True,
        )[:5]
        top_missing_names = [d["job_skill"] for d in top_missing]

        raw_pct = (len(matched) / len(jd_skills) * 100) if jd_skills else 0.0
        explanation = (
            f"Matched {len(matched)} of {len(jd_skills)} required skills ({raw_pct:.1f}% raw). "
            f"Weighted score: {weighted_score}/100. "
            + (f"Top missing skills: {', '.join(top_missing_names)}." if top_missing_names else "All required skills matched!")
        )

        return {
            "resume_skills": resume_skills,
            "job_skills": jd_skills,
            "details": details,
            "matched": sorted(matched),
            "missing": sorted(missing),
            "extra": sorted(extra),
            "weighted_score": weighted_score,
            "explanation": explanation,
        }
