"""
Tests for ml.gap.skill_gap_analyzer.SkillGapAnalyzer

Run:
    python -m pytest ml/gap/test_gap_analyzer.py -v
"""
import pytest
from ml.skills.skill_extractor import SkillExtractor
from ml.gap.skill_gap_analyzer import SkillGapAnalyzer


@pytest.fixture(scope="module")
def analyzer():
    extractor = SkillExtractor()
    return SkillGapAnalyzer(extractor=extractor)


RESUME_BACKEND = """
Skills: Python, PostgreSQL, Docker, AWS, Flask, REST APIs, git
Experience with CI/CD pipelines and Linux.
"""

JD_BACKEND = """
Requirements:
- Python programming
- Docker and Kubernetes
- AWS or GCP cloud experience
- PostgreSQL or MySQL database
- REST API development
- CI/CD pipelines
"""

RESUME_ML = """
Skills: Python, TensorFlow, PyTorch, pandas, NumPy, scikit-learn
Worked on NLP and computer vision projects.
"""

JD_ML = """
Looking for a Machine Learning Engineer with:
- Python and PyTorch or TensorFlow
- Experience in NLP and Computer Vision
- Data preprocessing and feature engineering
- Model deployment and MLOps
"""


class TestOutputShape:
    def test_all_keys_present(self, analyzer):
        result = analyzer.analyze(RESUME_BACKEND, JD_BACKEND)
        for key in ("resume_skills", "job_skills", "matched", "missing", "extra",
                    "weighted_score", "explanation", "details"):
            assert key in result

    def test_details_item_shape(self, analyzer):
        result = analyzer.analyze(RESUME_BACKEND, JD_BACKEND)
        for d in result["details"]:
            for key in ("job_skill", "job_weight", "category", "status", "best_match", "similarity"):
                assert key in d

    def test_status_values_are_valid(self, analyzer):
        result = analyzer.analyze(RESUME_BACKEND, JD_BACKEND)
        valid = {"matched", "related", "missing"}
        for d in result["details"]:
            assert d["status"] in valid

    def test_score_is_between_0_and_100(self, analyzer):
        result = analyzer.analyze(RESUME_BACKEND, JD_BACKEND)
        assert 0.0 <= result["weighted_score"] <= 100.0

    def test_similarity_is_between_0_and_1(self, analyzer):
        result = analyzer.analyze(RESUME_BACKEND, JD_BACKEND)
        for d in result["details"]:
            assert 0.0 <= d["similarity"] <= 1.0

    def test_job_skills_all_in_details(self, analyzer):
        result = analyzer.analyze(RESUME_BACKEND, JD_BACKEND)
        detail_skills = {d["job_skill"] for d in result["details"]}
        assert set(result["job_skills"]) == detail_skills


class TestMatchingLogic:
    def test_perfect_match_scores_high(self, analyzer):
        resume = "Python, Docker, AWS, PostgreSQL, REST APIs, CI/CD"
        jd = "Python, Docker, AWS, PostgreSQL"
        result = analyzer.analyze(resume, jd)
        assert result["weighted_score"] >= 70.0

    def test_no_overlap_scores_low(self, analyzer):
        resume = "Figma, Photoshop, Sketch, Illustrator"
        jd = "Python, Docker, Kubernetes, TensorFlow"
        result = analyzer.analyze(resume, jd)
        assert result["weighted_score"] < 30.0

    def test_matched_subset_of_job_skills(self, analyzer):
        result = analyzer.analyze(RESUME_BACKEND, JD_BACKEND)
        job_set = set(result["job_skills"])
        assert set(result["matched"]).issubset(job_set)

    def test_missing_subset_of_job_skills(self, analyzer):
        result = analyzer.analyze(RESUME_BACKEND, JD_BACKEND)
        job_set = set(result["job_skills"])
        assert set(result["missing"]).issubset(job_set)

    def test_matched_and_missing_cover_all_job_skills(self, analyzer):
        result = analyzer.analyze(RESUME_BACKEND, JD_BACKEND)
        covered = set(result["matched"]) | set(result["missing"])
        assert covered == set(result["job_skills"])

    def test_no_duplicates_in_matched(self, analyzer):
        result = analyzer.analyze(RESUME_BACKEND, JD_BACKEND)
        assert len(result["matched"]) == len(set(result["matched"]))

    def test_no_duplicates_in_missing(self, analyzer):
        result = analyzer.analyze(RESUME_BACKEND, JD_BACKEND)
        assert len(result["missing"]) == len(set(result["missing"]))


class TestMLScenario:
    def test_ml_resume_matches_ml_jd(self, analyzer):
        result = analyzer.analyze(RESUME_ML, JD_ML)
        assert result["weighted_score"] >= 50.0

    def test_pytorch_or_tensorflow_matched(self, analyzer):
        result = analyzer.analyze(RESUME_ML, JD_ML)
        assert "PyTorch" in result["matched"] or "TensorFlow" in result["matched"]

    def test_mlops_is_missing_from_ml_resume(self, analyzer):
        result = analyzer.analyze(RESUME_ML, JD_ML)
        detail = next((d for d in result["details"] if d["job_skill"] == "MLOps"), None)
        if detail:
            assert detail["status"] in ("missing", "related")


class TestEdgeCases:
    def test_empty_resume(self, analyzer):
        result = analyzer.analyze("", JD_BACKEND)
        assert result["weighted_score"] == 0.0
        assert result["matched"] == []

    def test_empty_jd(self, analyzer):
        result = analyzer.analyze(RESUME_BACKEND, "")
        assert result["job_skills"] == []
        assert result["weighted_score"] == 0.0

    def test_both_empty(self, analyzer):
        result = analyzer.analyze("", "")
        assert result["weighted_score"] == 0.0

    def test_explanation_is_nonempty_string(self, analyzer):
        result = analyzer.analyze(RESUME_BACKEND, JD_BACKEND)
        assert isinstance(result["explanation"], str)
        assert len(result["explanation"]) > 0
