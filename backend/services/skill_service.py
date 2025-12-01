from ml.skills.skill_extractor import SkillExtractor
from ml.gap.skill_gap_analyzer import SkillGapAnalyzer

class SkillService:
    def __init__(self):
        self.extractor = SkillExtractor()
        self.gap = SkillGapAnalyzer()

    def extract_skills(self, text: str):
        return self.extractor.extract_skills(text)

    def compute_skill_gap(self, resume_text: str, job_text: str):
        return self.gap.analyze(resume_text, job_text)