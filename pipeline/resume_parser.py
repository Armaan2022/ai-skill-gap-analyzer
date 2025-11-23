import os
import json
from pathlib import Path
from typing import Dict, Any
from preprocess import clean_text, extract_contact_info, split_sections
from ml.skills.skill_extractor import SkillExtractor


SKILL_EXTRACTOR = SkillExtractor()

def parse_resume_text(resume_text: str) -> Dict[str, Any]:
    clean = clean_text(resume_text)
    contacts = extract_contact_info(clean)
    sections = split_sections(clean)
    found_skills = SKILL_EXTRACTOR.extract_skills(clean)

    # basic heuristics for name: first non-empty line before contact info
    lines = [l for l in clean.splitlines() if l.strip()]
    name = lines[0].strip() if lines else None

    parsed = {
        "name": name,
        "contacts": contacts,
        "sections": sections,
        "extracted_skills": found_skills,
        "full_text_snippet": clean[:1000]
    }

    return parsed

def parse_and_save_file(input_path: str, output_dir: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    parsed = parse_resume_text(text)
    out_name = Path(input_path).stem + ".json"
    output_path = Path(output_dir) / out_name
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(parsed, out, indent=2)
    print(f"Saved parsed resume -> {output_path}")

