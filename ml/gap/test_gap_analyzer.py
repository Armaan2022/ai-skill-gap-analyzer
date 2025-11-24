from ml.gap.skill_gap_analyzer import SkillGapAnalyzer

resume = """
Experienced backend developer with Python, SQL, APIs, Docker, cloud, 
TensorFlow, and machine learning experience.
"""

jd = """
Looking for a backend engineer skilled in Python, Docker, AWS, 
SQL, API Development, and Cloud Architecture.
"""

analyzer = SkillGapAnalyzer(match_threshold=0.72, related_threshold=0.50)
result = analyzer.analyze(resume, jd)

print("\n=== Advanced Skill Gap Result ===")
print("Weighted Score:", result["weighted_score"])
print("Matched:", result["matched"])
print("Missing:", result["missing"])
print("Extra:", result["extra"])
print("\nDetails:")
for d in result["details"]:
    print(f"- {d['job_skill']}: status={d['status']} sim={d['similarity']} weight={d['job_weight']} best_match={d['best_match']}")
print("\nExplanation:", result["explanation"])