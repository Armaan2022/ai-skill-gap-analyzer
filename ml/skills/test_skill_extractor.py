"""
Tests for ml.skills.skill_extractor.SkillExtractor

Run:
    python -m pytest ml/skills/test_skill_extractor.py -v
"""
import pytest
from ml.skills.skill_extractor import SkillExtractor


@pytest.fixture(scope="module")
def extractor():
    return SkillExtractor()


class TestAliasLookup:
    def test_golang_maps_to_go(self, extractor):
        result = extractor.map_phrase_to_skill("golang")
        assert result[0] == ("Go", 1.0)

    def test_py_maps_to_python(self, extractor):
        result = extractor.map_phrase_to_skill("py")
        assert result[0] == ("Python", 1.0)

    def test_k8s_maps_to_kubernetes(self, extractor):
        result = extractor.map_phrase_to_skill("k8s")
        assert result[0][0] == "Kubernetes"

    def test_ts_maps_to_typescript(self, extractor):
        result = extractor.map_phrase_to_skill("ts")
        assert result[0][0] == "TypeScript"

    def test_exact_canonical_returns_score_1(self, extractor):
        result = extractor.map_phrase_to_skill("Python")
        assert result[0] == ("Python", 1.0)

    def test_canonical_case_insensitive(self, extractor):
        result = extractor.map_phrase_to_skill("python")
        assert result[0][0] == "Python"


class TestPhraseExtraction:
    def test_comma_list_fast_path(self, extractor):
        phrases = extractor.extract_skill_phrases("Python, React, AWS, Docker")
        assert "python" in phrases
        assert "react" in phrases
        assert "aws" in phrases
        assert "docker" in phrases

    def test_blacklist_filters_noise(self, extractor):
        phrases = extractor.extract_skill_phrases(
            "5 years of experience with strong understanding of systems"
        )
        assert "years" not in phrases
        assert "experience" not in phrases
        assert "strong" not in phrases

    def test_single_word_skills_included(self, extractor):
        result = extractor.extract_skills(
            "We need Python and Go expertise for this backend role."
        )
        assert "Python" in result["skills"] or "Go" in result["skills"]

    def test_max_phrase_words(self, extractor):
        phrases = extractor.extract_skill_phrases(
            "Proficient in Python machine learning deep learning neural networks"
        )
        assert all(len(p.split()) <= 6 for p in phrases)

    def test_empty_text_returns_empty(self, extractor):
        assert extractor.extract_skill_phrases("") == []


class TestExtractSkills:
    def test_basic_skill_list(self, extractor):
        result = extractor.extract_skills(
            "Experience with Python, React, AWS Lambda, Docker, and PostgreSQL"
        )
        skills = result["skills"]
        assert "Python" in skills
        assert "React" in skills
        assert "Docker" in skills
        assert "PostgreSQL" in skills

    def test_returns_canonical_names(self, extractor):
        result = extractor.extract_skills("Skills: golang, py, k8s")
        skills = result["skills"]
        assert "Go" in skills or "Python" in skills or "Kubernetes" in skills

    def test_result_has_required_keys(self, extractor):
        result = extractor.extract_skills("Python developer with Django experience")
        assert "skills" in result
        assert "phrases" in result
        assert "mappings" in result

    def test_no_duplicate_skills(self, extractor):
        result = extractor.extract_skills(
            "Python, Python programming, py — all refer to the same language"
        )
        skills = result["skills"]
        assert len(skills) == len(set(skills))

    def test_empty_text_returns_empty_skills(self, extractor):
        result = extractor.extract_skills("")
        assert result["skills"] == []

    def test_jd_style_text(self, extractor):
        jd = """
        Requirements:
        - 3+ years of Python or Java experience
        - Proficient in Docker and Kubernetes
        - Familiarity with AWS or GCP
        - Strong SQL skills (PostgreSQL preferred)
        """
        result = extractor.extract_skills(jd)
        skills = result["skills"]
        assert len(skills) >= 3
        assert any(s in skills for s in ["Python", "Java"])
        assert any(s in skills for s in ["Docker", "Kubernetes"])

    def test_resume_style_text(self, extractor):
        resume = """
        Skills: Python, React, PostgreSQL, Docker, AWS, scikit-learn, pandas
        Experience with REST APIs and Flask.
        """
        result = extractor.extract_skills(resume)
        skills = result["skills"]
        assert "Python" in skills
        assert "React" in skills
        assert len(skills) >= 4


class TestCustomSkills:
    def test_add_custom_skill(self, extractor):
        extractor.add_custom_skills(["QuantumML"], recompute_embeddings=False)
        assert "QuantumML" in extractor.canonical_skills
        assert "quantumml" in extractor.alias_lookup or "quantumml" in extractor._skill_lower_to_idx

    def test_no_duplicate_on_readd(self, extractor):
        before = len(extractor.canonical_skills)
        extractor.add_custom_skills(["Python"], recompute_embeddings=False)
        assert len(extractor.canonical_skills) == before
