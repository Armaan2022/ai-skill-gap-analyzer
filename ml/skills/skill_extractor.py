from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# Constants
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
FALLBACK_MODEL = "all-MiniLM-L6-v2"
BASE_DIR = Path(__file__).resolve().parent
SKILLS_MASTER = BASE_DIR / "skills_master.json"
SKILL_META = BASE_DIR / "skill_meta.json"
EMBED_CACHE_FILE = BASE_DIR / "skills_embeddings.npy"
EMBED_CACHE_META = BASE_DIR / "skills_embeddings_meta.json"

# Phrases that should never be treated as skills
PHRASE_BLACKLIST = {
    "years", "year", "experience", "strong", "understanding", "knowledge",
    "ability", "work", "team", "management", "development", "background",
    "basis", "use", "using", "used", "good", "great", "excellent",
    "hands", "hands-on", "ability to", "working knowledge", "solid",
    "well", "strong understanding", "good understanding", "familiarity",
    "proficiency", "expertise", "skills", "skill", "requirement",
    "requirements", "responsibilities", "responsibility", "qualification",
    "qualifications", "preferred", "required", "plus", "bonus",
    "nice to have", "must have", "minimum", "at least", "years of",
}


class SkillExtractor:
    """
    Hybrid SkillExtractor
    - Rule-based phrase extraction using spaCy (noun chunks + simple patterns)
    - Alias lookup from skill_meta.json (O(1) exact match for known aliases)
    - Embedding-based normalization (semantic matching) against canonical skill list

    Usage:
        from ml.skills.skill_extractor import SkillExtractor
        se = SkillExtractor()
        result = se.extract_skills(text)  # -> {"skills": [...], "phrases": [...], "mappings": [...]}
    """

    def __init__(
        self,
        model_name: Optional[str] = DEFAULT_MODEL,
        skill_file: Optional[Path] = None,
        device: Optional[str] = None,
        reuse_cache: bool = True,
        threshold: float = 0.72,
    ):
        self.skill_file = Path(skill_file) if skill_file else SKILLS_MASTER
        if not self.skill_file.exists():
            raise FileNotFoundError(f"Skill master file not found: {self.skill_file}")

        self.threshold = threshold
        self.load_skill_master()
        self._load_alias_lookup()

        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm"
            ) from e

        # Load sentence-transformers model
        self.model_name = model_name or DEFAULT_MODEL
        try:
            self.embedder = SentenceTransformer(self.model_name, device=device)
        except Exception:
            self.model_name = FALLBACK_MODEL
            self.embedder = SentenceTransformer(self.model_name, device=device)

        self.reuse_cache = reuse_cache
        self._prepare_skill_embeddings()

    def load_skill_master(self):
        """Load canonical skills from skills_master.json."""
        with open(self.skill_file, "r", encoding="utf-8") as f:
            js = json.load(f)
        raw_skills = js.get("skills") if isinstance(js, dict) else js
        if raw_skills is None:
            raise ValueError("skills_master.json should contain a top-level 'skills' array")

        self.canonical_skills: List[str] = [s.strip() for s in raw_skills if isinstance(s, str)]
        self.canonical_skills_lower = [s.lower() for s in self.canonical_skills]
        # O(1) lookup: lowercase -> index
        self._skill_lower_to_idx: Dict[str, int] = {
            s: i for i, s in enumerate(self.canonical_skills_lower)
        }

    def _load_alias_lookup(self):
        """
        Build alias_lookup dict from skill_meta.json:
            { "py": "Python", "golang": "Go", ... }
        Also covers the canonical names themselves (lowercased).
        """
        self.alias_lookup: Dict[str, str] = {}

        # canonical names -> themselves
        for skill in self.canonical_skills:
            self.alias_lookup[skill.lower()] = skill

        if not SKILL_META.exists():
            return

        try:
            meta = json.loads(SKILL_META.read_text(encoding="utf-8"))
        except Exception:
            return

        for canonical, info in meta.items():
            self.alias_lookup[canonical.lower()] = canonical
            for alias in info.get("aliases", []):
                if alias:
                    self.alias_lookup[alias.lower().strip()] = canonical

    def _prepare_skill_embeddings(self):
        """Load or compute embeddings for the canonical skill list."""
        if self.reuse_cache and EMBED_CACHE_FILE.exists() and EMBED_CACHE_META.exists():
            try:
                meta = json.loads(EMBED_CACHE_META.read_text(encoding="utf-8"))
                if (
                    meta.get("model_name") == self.model_name
                    and meta.get("n_skills") == len(self.canonical_skills)
                ):
                    arr = np.load(EMBED_CACHE_FILE)
                    self.skill_embs = normalize(arr, axis=1)
                    return
            except Exception:
                pass

        batch_size = 64
        embs = []
        for i in range(0, len(self.canonical_skills), batch_size):
            batch = self.canonical_skills[i : i + batch_size]
            enc = self.embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embs.append(enc)
        embs_arr = np.vstack(embs).astype(np.float32)
        embs_arr = normalize(embs_arr, axis=1)
        self.skill_embs = embs_arr

        try:
            np.save(EMBED_CACHE_FILE, embs_arr)
            EMBED_CACHE_META.write_text(
                json.dumps({"model_name": self.model_name, "n_skills": len(self.canonical_skills)}),
                encoding="utf-8",
            )
        except Exception:
            pass

    def extract_skill_phrases(self, text: str, max_phrases: int = 80) -> List[str]:
        """Extract candidate skill phrases from text."""
        if not text:
            return []

        # Fast path: short comma-separated skill lists
        simple_separators = r",|;|/|\||\sand\s"
        if len(text.split()) <= 20:
            parts = [p.strip().lower() for p in re.split(simple_separators, text) if p.strip()]
            if len(parts) > 1:
                return [p for p in parts if self._is_valid_phrase(p)]

        doc = self.nlp(text)
        candidates = []

        # 1) Noun chunks (1-6 words)
        for nc in doc.noun_chunks:
            phrase = nc.text.strip()
            if 1 <= len(phrase.split()) <= 6:
                candidates.append(phrase)

        # 2) Named entities
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART", "NORP", "TECH"):
                candidates.append(ent.text.strip())

        # 3) Regex patterns
        patterns = [
            r"(?:experience with|experienced in|proficient in|familiar with|skilled in|expertise in|knowledge of)\s+([A-Za-z0-9\-\+\.#\s/]+)",
            r"([A-Za-z0-9\-\+\.#]+(?:\s[A-Za-z0-9\-\+\.#]+){0,3})\s+(?:experience|knowledge|skills|expertise)",
        ]
        for pat in patterns:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                grp = m.group(1).strip()
                parts = [p.strip() for p in re.split(r",|;|/| and ", grp) if p.strip()]
                for p in parts:
                    if 1 <= len(p) <= 50:
                        candidates.append(p)

        # 4) Skills section heuristic
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for i, ln in enumerate(lines):
            if re.match(r"(?i)^(technical\s+)?skills\b", ln):
                for j in range(1, 5):
                    if i + j < len(lines):
                        chunk = lines[i + j]
                        parts = [p.strip() for p in re.split(r",|\||\s*•\s*|;|/| and ", chunk) if p.strip()]
                        candidates.extend(parts)

        # 5) Token scan for single-word aliases
        for token in doc:
            if not token.is_stop and not token.is_punct and len(token.text) >= 2:
                tl = token.text.lower()
                if tl in self.alias_lookup:
                    candidates.append(token.text)

        # Normalize & deduplicate
        normalized = []
        seen: set = set()
        for c in candidates:
            low = c.strip().lower()
            if not self._is_valid_phrase(low):
                continue
            if low not in seen:
                seen.add(low)
                normalized.append(low)
            if len(normalized) >= max_phrases:
                break

        return normalized

    def _is_valid_phrase(self, phrase: str) -> bool:
        if len(phrase) < 2:
            return False
        if len(phrase.split()) > 6:
            return False
        if all(ch in ".,;-/()[]{}#+" for ch in phrase):
            return False
        if phrase in PHRASE_BLACKLIST:
            return False
        if re.match(r"^\d+(\+)?\s*(years?|yrs?)?$", phrase):
            return False
        return True

    def map_phrase_to_skill(self, phrase: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Map a candidate phrase to canonical skills."""
        if not phrase.strip():
            return []

        phrase_l = phrase.strip().lower()

        # 1) Alias lookup
        if phrase_l in self.alias_lookup:
            return [(self.alias_lookup[phrase_l], 1.0)]

        # 2) Canonical exact match
        if phrase_l in self._skill_lower_to_idx:
            idx = self._skill_lower_to_idx[phrase_l]
            return [(self.canonical_skills[idx], 1.0)]

        # 3) Embedding similarity
        emb = self.embedder.encode([phrase], convert_to_numpy=True)
        emb = normalize(emb, axis=1)
        sims = cosine_similarity(emb, self.skill_embs)[0]
        top_idx = np.argsort(-sims)[:top_k]
        return [(self.canonical_skills[int(i)], float(sims[int(i)])) for i in top_idx]

    def map_phrases_to_skills(
        self, phrases: List[str], threshold: Optional[float] = None, top_k: int = 3
    ) -> Dict[str, Any]:
        thr = threshold if threshold is not None else self.threshold
        mappings = []
        chosen: List[str] = []

        for p in phrases:
            cands = self.map_phrase_to_skill(p, top_k=top_k)
            chosen_skill = None
            if cands and cands[0][1] >= thr:
                chosen_skill = cands[0][0]
                if chosen_skill not in chosen:
                    chosen.append(chosen_skill)
            mappings.append({"phrase": p, "candidates": cands, "chosen": chosen_skill})

        return {"mappings": mappings, "chosen_skills": chosen}

    def extract_skills(self, text: str, phrase_threshold: Optional[float] = None, top_k: int = 3) -> Dict[str, Any]:
        """Full pipeline: extract phrases -> map to canonical skills."""
        phrases = self.extract_skill_phrases(text)
        mapped = self.map_phrases_to_skills(phrases, threshold=phrase_threshold, top_k=top_k)
        return {
            "phrases": phrases,
            "mappings": mapped["mappings"],
            "skills": mapped["chosen_skills"],
        }

    def add_custom_skills(self, new_skills: List[str], recompute_embeddings: bool = True):
        for s in new_skills:
            if s not in self.canonical_skills:
                self.canonical_skills.append(s)
                self.canonical_skills_lower.append(s.lower())
                self._skill_lower_to_idx[s.lower()] = len(self.canonical_skills) - 1
                self.alias_lookup[s.lower()] = s
        if recompute_embeddings:
            self._prepare_skill_embeddings()

    def canonicalize(self, skill: str) -> Optional[str]:
        res = self.map_phrase_to_skill(skill, top_k=1)
        if res:
            return res[0][0]
        return None
