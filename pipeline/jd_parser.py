import os
import json
from pathlib import Path
from typing import Dict, Any
from preprocess import clean_text, split_sections
from ml.skills.skill_extractor import SkillExtractor


SKILL_EXTRACTOR = SkillExtractor()

def parse_jd_text(text: str) -> Dict[str, Any]:
    clean = clean_text(text)
    sections = split_sections(clean)

    # extract skills from the whole JD
    found_skills = SKILL_EXTRACTOR.extract_skills(clean)

    # try to capture title from first non-empty line
    lines = [l for l in clean.splitlines() if l.strip()]
    title = lines[0].strip() if lines else None

    parsed = {
        "title": title,
        "sections": sections,
        "required_skills": found_skills,
        "full_text_snippet": clean[:1000]
    }
    return parsed

def parse_and_save_jd(input_path: str, output_dir: str):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    parsed = parse_jd_text(text)
    out_name = Path(input_path).stem + ".json"
    output_path = Path(output_dir) / out_name
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(parsed, out, indent=2)
    print(f"Saved parsed JD -> {output_path}")
