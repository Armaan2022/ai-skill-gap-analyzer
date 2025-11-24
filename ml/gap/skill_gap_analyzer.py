from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.preprocessing import normalize
from pathlib import Path
import json
from ml.skills.skill_extractor import SkillExtractor

META_PATH = Path(__file__).resolve().parents[1] / "skills" / "skill_meta.json"

# Default category weights (tweakable)
DEFAULT_CATEGORY_WEIGHTS = {
    "programming_languages": 1.0,
    "machine_learning": 1.2,
    "deep_learning": 1.3,
    "cloud": 1.4,
    "databases": 1.1,
    "devops": 1.3,
    "frontend": 1.0,
    "backend": 1.2,
    "data_engineering": 1.2,
    "tools": 1.0,
    "soft_skills": 0.6,
    "default": 1.0,
}

class SkillGapAnalyzer:
    """
    Advanced Skill Gap Analyzer (Option B)

    - Uses your existing ml.skills.skill_extractor.SkillExtractor
    - Produces category-aware weighted match score
    - Identifies matched / related / missing skills
    - Returns detailed per-skill breakdown + overall score (0-100)
    """
    def __init__(
        self,
        extractor: Optional[SkillExtractor] = None,
        category_weights: Optional[Dict[str, float]] = None,
        match_threshold: float = 0.70,
        related_threshold: float = 0.50,
    ):
        """
        Args:
            extractor: existing SkillExtractor instance (reused to avoid re-loading models)
            category_weights: mapping category -> weight (overrides defaults)
            match_threshold: cosine similarity threshold considered a direct match
            related_threshold: lower threshold for related-but-not-exact matches
        """
        self.extractor = extractor or SkillExtractor()
        self.match_threshold = match_threshold
        self.related_threshold = related_threshold
        self.category_weights = {**DEFAULT_CATEGORY_WEIGHTS, **(category_weights or {})}

        # try to load optional skill->category meta if user provided it
        self.skill_meta = self._load_skill_meta()
    
    def _load_skill_meta(self) -> Dict[str, Dict[str, Any]]:
        """
        If ml/skills/skill_meta.json exists, load mapping:
            { "Python": {"category": "programming_languages", "default_weight": 1.0}, ... }
        Otherwise return empty dict.
        """
        meta_path = Path(self.extractor.skill_file).parent / "skill_meta.json"
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _skill_category_and_weight(self, skill: str) -> Tuple[str, float]:
        """
        Determine category and weight for a canonical skill.
        1) If skill present in skill_meta (user-provided) use that.
        2) Otherwise heuristically infer category by keywords in skill text.
        3) Use DEFAULT weights.
        """
        # exact meta lookup
        meta = self.skill_meta.get(skill)
        if meta:
            return meta.get("category", "default"), float(meta.get("default_weight", 1.0))

        low = skill.lower()
        # heuristic rules
        heur = [
            (["python", "java", "c++", "typescript", "javascript", "go", "rust", "kotlin", "swift"], "programming_languages"),
            (["tensorflow", "pytorch", "keras", "neural", "deep learning", "transformer", "llm", "onnx"], "deep_learning"),
            (["machine learning", "ml", "scikit", "xgboost", "lightgbm"], "machine_learning"),
            (["aws", "azure", "gcp", "cloud", "lambda", "ec2", "s3"], "cloud"),
            (["docker", "kubernetes", "ci/cd", "terraform", "helm"], "devops"),
            (["sql", "postgres", "mysql", "mongodb", "redis", "dynamo"], "databases"),
            (["react", "angular", "vue", "frontend", "html", "css"], "frontend"),
            (["api", "rest", "graphql", "backend", "server"], "backend"),
            (["spark", "hadoop", "airflow", "etl", "data pipeline"], "data_engineering"),
            (["jira", "confluence", "leadership", "communication", "teamwork"], "soft_skills"),
        ]
        for tokens, category in heur:
            if any(tok in low for tok in tokens):
                return category, float(self.category_weights.get(category, self.category_weights["default"]))

        # fallback
        return "default", float(self.category_weights.get("default", 1.0))

    def _embed_list(self, texts: List[str]) -> np.ndarray:
        """
        Return normalized embeddings for a list of texts using the extractor's embedder.
        """
        if not texts:
            return np.zeros((0, self.extractor.skill_embs.shape[1]), dtype=np.float32) if hasattr(self.extractor, "skill_embs") else np.zeros((0, 384), dtype=np.float32)
        arr = self.extractor.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        arr = arr.astype(np.float32)
        arr = normalize(arr, axis=1)
        return arr
    
    def analyze(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """
        Main analyzer.

        Returns:
            {
                "resume_skills": [...],
                "job_skills": [...],
                "details": [ { "job_skill": ..., "job_weight":..., "status":"matched|related|missing", "best_match":..., "similarity":..., "category":... }, ... ],
                "matched": [...], "missing": [...], "extra": [...],
                "weighted_score": float (0-100),
                "explain": str
            }
        """
        # 1) extract canonical skills
        resume_res = self.extractor.extract_skills(resume_text)
        jd_res = self.extractor.extract_skills(jd_text)

        resume_skills = resume_res.get("skills", [])  # canonical skill names
        jd_skills = jd_res.get("skills", [])

        # 2) if resume or jd skills are empty, also attempt to use phrase candidates (fallback) to increase coverage
        # Map phrases to canonical if possible (extractor already does mapping inside extract_skills)
        # 3) compute embeddings for canonical skill labels (these are short strings)
        jd_embs = self._embed_list(jd_skills)
        res_embs = self._embed_list(resume_skills)

        # Prepare lookup
        details = []
        matched = []
        missing = []
        extra = []

        # Build similarity matrix (job x resume)
        if jd_embs.shape[0] > 0 and res_embs.shape[0] > 0:
            sims = np.matmul(jd_embs, res_embs.T)  # cosine because both normalized
        else:
            sims = np.zeros((len(jd_skills), len(resume_skills)), dtype=np.float32)

        # compute per-job skill best match
        total_weight_possible = 0.0
        weighted_score_obtained = 0.0

        for i, job_skill in enumerate(jd_skills):
            category, cat_weight = self._skill_category_and_weight(job_skill)
            # weight contribution for this skill (you can adjust formula to be more nuanced)
            job_weight = cat_weight
            total_weight_possible += job_weight

            best_match = None
            best_sim = 0.0
            if resume_skills and sims.shape[1] > 0:
                jrow = sims[i]
                idx = int(np.argmax(jrow))
                best_sim = float(jrow[idx])
                best_match = resume_skills[idx]

            # Determine status
            if best_sim >= self.match_threshold:
                status = "matched"
                weighted_score_obtained += job_weight * best_sim
                matched.append(job_skill)
            elif best_sim >= self.related_threshold:
                status = "related"
                # give partial credit (half the sim*weight)
                weighted_score_obtained += job_weight * (0.5 * best_sim)
                missing.append(job_skill)  # still considered missing, but related
            else:
                status = "missing"
                missing.append(job_skill)

            details.append(
                {
                    "job_skill": job_skill,
                    "job_weight": job_weight,
                    "category": category,
                    "status": status,
                    "best_match": best_match,
                    "similarity": round(best_sim, 4),
                }
            )

        # extras: resume skills not matched to any job skill above related threshold
        for r in resume_skills:
            # check if r matched any job skill at >= related_threshold
            r_in_good = False
            for d in details:
                if d["best_match"] == r and d["similarity"] >= self.related_threshold:
                    r_in_good = True
                    break
            if not r_in_good:
                extra.append(r)

        # Avoid division by zero
        if total_weight_possible > 0:
            weighted_score = round(float(weighted_score_obtained / total_weight_possible * 100), 2)
        else:
            weighted_score = 0.0

        # Compose explanation text
        explain_lines = [
            f"Matched {len(matched)} of {len(jd_skills)} required skills ({len(matched)/len(jd_skills)*100:.1f}% raw).",
            f"Weighted score (category-aware): {weighted_score} / 100.",
            f"Thresholds: direct match>={self.match_threshold}, related>={self.related_threshold}.",
        ]
        explanation = " ".join(explain_lines)

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