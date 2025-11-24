from skill_extractor import SkillExtractor

# Initialize extractor
extractor = SkillExtractor()

# Sample resume text
sample_text = """
Experienced software developer skilled in Python, machine learning,
TensorFlow, APIs, Docker, and cloud deployment. Familiar with SQL,
FastAPI, and large language models. Looking to work on backend systems.
"""

result = extractor.extract_skills(sample_text)

print("\n---- Extracted Skills ----")
print("Skill-like phrases:")
for p in result["phrases"]:
    print(" -", p)

print("\nMapped skills:")
for m in result["mappings"]:
    print(" -", m)

print("\nFinal chosen skills:")
for s in result["skills"]:
    print(" -", s)