import json
import re

class SkillExtractor:
    def __init__(self, dict_path="ml/skills/skill_dictionary.json"):
        with open(dict_path, 'r') as file:
            self.skill_dict = json.load(file)

        self.all_skills = []
        for category, skills in self.skill_dict.items():
            for skill in skills:
                self.all_skills.append(skill.lower())
    
    def extract_skills(self, text):
        text = text.lower()
        found = set()

        for skill in self.all_skills:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text):
                found.add(skill)
        
        return list(found)