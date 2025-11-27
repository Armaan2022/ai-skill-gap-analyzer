from __future__ import annotations
import json
import os
from pathlib import Path
from pydoc import text
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# Constants
DEFAULT_MODEL = "BAAI/bge-small-en"    # recommended; fallback if unavailable
FALLBACK_MODEL = "all-MiniLM-L6-v2"    # faster, widely available
BASE_DIR = Path(__file__).resolve().parent
SKILLS_MASTER = BASE_DIR / "skills_master.json"
EMBED_CACHE_FILE = BASE_DIR / "skills_embeddings.npy"
EMBED_CACHE_META = BASE_DIR / "skills_embeddings_meta.json"

class SkillExtractor:
    """
        Hybrid SkillExtractor
        - Rule-based phrase extraction using spaCy (noun chunks + simple patterns)
        - Embedding-based normalization (semantic matching) against a canonical skill list (skills_master.json)

        Usage:
            from ml.skills.skill_extractor import SkillExtractor
            se = SkillExtractor(model_name="BAAI/bge-small-en")  # recommended
            phrases = se.extract_skill_phrases(text)            # raw candidate phrases
            mapped = se.map_phrases_to_skills(phrases)          # canonical mapping with scores
            skills = se.extract_skills(text)                    # convenience: pipeline -> canonical skills
    """
    def __init__(
        self,
        model_name: Optional[str] = DEFAULT_MODEL,
        skill_file: Optional[Path] = None,
        device: Optional[str] = None,
        reuse_cache: bool = True,
    ):
        """
        Initialize SkillExtractor.

        Args:
            model_name: HuggingFace / SentenceTransformers model name. If unavailable, falls back.
            skill_file: Path to skills_master.json (defaults to ml/skills/skills_master.json)
            device: 'cpu' or 'cuda' (optional)
            reuse_cache: if True, will try to load precomputed canonical skill embeddings
        """
        self.skill_file = Path(skill_file) if skill_file else SKILLS_MASTER
        if not self.skill_file.exists():    
            raise FileNotFoundError(f"Skill master file not found: {self.skill_file}")

        self.load_skill_master()

        # Load spaCy model for phrase extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            raise RuntimeError("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm") from e
        
        # Load sentence-transformers model (try recommended, fallback if necessary)
        self.model_name = model_name or DEFAULT_MODEL
        try:
            self.embedder = SentenceTransformer(self.model_name, device=device)
        except Exception:
            self.model_name = FALLBACK_MODEL
            self.embedder = SentenceTransformer(self.model_name, device=device)
        
        # Prepare canonical skill embeddings (compute or load cache)
        self.reuse_cache = reuse_cache
        self._prepare_skill_embeddings()
    
    def load_skill_master(self):
        """
        Load canonical skills from skills_master.json
        """
        with open(self.skill_file, "r", encoding="utf-8") as f:
            js = json.load(f)
        raw_skills = js.get("skills") if isinstance(js, dict) else js
        if raw_skills is None:
            raise ValueError("skills_master.json should contain a top-level 'skills' array")

        self.canonical_skills: List[str] = [s.strip() for s in raw_skills if isinstance(s, str)]
        self.canonical_skills_lower = [s.lower() for s in self.canonical_skills]
    

    def _prepare_skill_embeddings(self):
        """
        Load or compute embeddings for the canonical skill list.
        Saves to .npy cache + json meta for faster startup.
        """
        # If cache exists and reuse_cache True -> try to load and ensure lengths match
        if self.reuse_cache and EMBED_CACHE_FILE.exists() and EMBED_CACHE_META.exists():
            try:
                meta = json.loads(EMBED_CACHE_META.read_text(encoding="utf-8"))
                if meta.get("model_name") == self.model_name and meta.get("n_skills") == len(self.canonical_skills):
                    arr = np.load(EMBED_CACHE_FILE)
                    # normalize
                    self.skill_embs = normalize(arr, axis=1)
                    return
            except Exception:
                # ignore and recompute
                pass
        
        # Compute embeddings for canonical skills
        # We compute embeddings in batches to avoid OOM on long lists
        batch_size = 64
        all_texts = self.canonical_skills
        embs = []
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i+batch_size]
            enc = self.embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embs.append(enc)
        embs_arr = np.vstack(embs).astype(np.float32)
        # normalize for cosine similarity
        embs_arr = normalize(embs_arr, axis=1)
        self.skill_embs = embs_arr

        # Save to cache
        try:
            np.save(EMBED_CACHE_FILE, embs_arr)
            meta = {
                "model_name": self.model_name,
                "n_skills": len(self.canonical_skills),
            }
            EMBED_CACHE_META.write_text(json.dumps(meta), encoding="utf-8")
        except Exception:
            pass

    def extract_skill_phrases(self, text: str, max_phrases: int = 60) -> List[str]:
        """
        Extract candidate skill phrases from text using spaCy noun-chunks + pattern matching.
        Returns a deduplicated list of short phrases (lowercased).
        spaCy + heuristics
        """

        if not text:
            return []
        
        # NEW: simple shortcut for comma-separated or short lists
        import re
        simple_separators = r",|;|/|\||\sand\s"
        if len(text.split()) <= 15:  
            # Usually means the input is a short skill list like: "Python, React, AWS"
            parts = [p.strip().lower() for p in re.split(simple_separators, text) if p.strip()]
            if len(parts) > 1:  # only use this if it actually splits
                return parts
        
        doc = self.nlp(text)
        candidates = []

        # 1) Noun chunks (short ones)
        for nc in doc.noun_chunks:
            phrase = nc.text.strip()
            if 2 <= len(phrase.split()) <= 80:
                candidates.append(phrase)
        
        # 2) Entities and proper nouns (ORG/PRODUCT etc.)
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART", "NORP", "TECH"):
                phrase = ent.text.strip()
                candidates.append(phrase)
        
        # 3) Simple regex-based patterns: "experience with X", "proficient in X", "familiar with X"

        patterns = [
            r"(?:experience with|experienced in|proficient in|familiar with|skilled in)\s+([A-Za-z0-9\-\+\.#\s/]+)",
            r"([A-Za-z0-9\-\+\.#]+)\s+(?:experience|knowledge|skills)",
        ]
        for pat in patterns:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                grp = m.group(1).strip()
                # split comma-separated lists
                parts = [p.strip() for p in re.split(r",|;|/| and ", grp) if p.strip()]
                for p in parts:
                    if 1 <= len(p) <= 80:
                        candidates.append(p)
        
        # 4) Heuristic: find lines under a "skills" section if present
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for i, ln in enumerate(lines):
            if ln.lower().startswith("skills") or ln.lower().startswith("technical skills"):
                # capture next up to 3 lines as skill lists
                for j in range(1, 4):
                    if i + j < len(lines):
                        chunk = lines[i + j]
                        # split commas / pipes / semicolons
                        parts = [p.strip() for p in re.split(r",|\||;|/| and ", chunk) if p.strip()]
                        for p in parts:
                            candidates.append(p)

        # Normalize & deduplicate (keep order)
        normalized = []
        seen = set()
        for c in candidates:
            low = c.strip().lower()
            # remove very short tokens
            if len(low) < 2 or len(low.split()) > 8:
                continue
            # remove punctuation-only
            if all(ch in ".,;-/()[]{}" for ch in low):
                continue
            if low not in seen:
                seen.add(low)
                normalized.append(low)

            if len(normalized) >= max_phrases:
                break

        return normalized
    
    def map_phrase_to_skill(self, phrase: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Map a single candidate phrase to canonical skills using cosine similarity.
        Returns a list of (canonical_skill, score) sorted by descending score.
        Mapping / normalization (embedding-based)
        """
        if not phrase.strip():
            return []

        # Exact match quick path
        phrase_l = phrase.strip().lower()
        if phrase_l in self.canonical_skills_lower:
            idx = self.canonical_skills_lower.index(phrase_l)
            return [(self.canonical_skills[idx], 1.0)]
        
        # Embed phrase
        emb = self.embedder.encode([phrase], convert_to_numpy=True)
        # normalize
        emb = normalize(emb, axis=1)
        #compute cosine similarity
        sims = cosine_similarity(emb, self.skill_embs)[0]  # shape (n_skills,) 
        # Get top_k matches
        top_idx = np.argsort(-sims)[:top_k]
        results = [(self.canonical_skills[int(i)], float(sims[int(i)])) for i in top_idx]
        return results

    def map_phrases_to_skills(self, phrases: List[str], threshold: float = 0.65, top_k: int = 3) -> Dict[str, Any]:
        """
        Map many phrases to canonical skills.
        Returns a dict with:
            - mappings: list of { phrase, candidates: [(skill,score), ...], chosen: optional }
            - chosen_skills: list of skills whose score >= threshold (deduped)
        """

        mappings = []
        chosen = []

        for p in phrases:
            cands = self.map_phrase_to_skill(p, top_k=top_k)
            # choose best candidate if above threshold
            chosen_skill = None
            if cands and cands[0][1] >= threshold:
                chosen_skill = cands[0][0]
                if chosen_skill not in chosen:
                    chosen.append(chosen_skill)
            mappings.append({"phrase": p, "candidates": cands, "chosen": chosen_skill})
        return {"mappings": mappings, "chosen_skills": chosen}
    

    def extract_skills(self, text: str, phrase_threshold: float = 0.65, top_k: int = 3) -> Dict[str, Any]:
        """
        Full pipeline for convenience:
          1. extract skill-like phrases
          2. map phrases to canonical skills (embedding)
          3. return chosen skills + raw mappings
        """
        phrases = self.extract_skill_phrases(text)
        mapped = self.map_phrases_to_skills(phrases, threshold=phrase_threshold, top_k=top_k)
        return {
            "phrases": phrases,
            "mappings": mapped["mappings"],
            "skills": mapped["chosen_skills"],
    }

    # -------------------------
    # Helper: add custom skills at runtime
    # -------------------------
    def add_custom_skills(self, new_skills: List[str], recompute_embeddings: bool = True):
        """
        Add extra canonical skills at runtime (useful for domain-specific skill insertion).
        """
        for s in new_skills:
            if s not in self.canonical_skills:
                self.canonical_skills.append(s)
                self.canonical_skills_lower.append(s.lower())
        if recompute_embeddings:
            self._prepare_skill_embeddings()
    
    def canonicalize(self, skill: str) -> Optional[str]:
        """
        Utility method.
        Return the canonical skill string for an exact or best-effort mapping.
        """
        res = self.map_phrase_to_skill(skill, top_k=1)
        if res:
            return res[0][0]
        return None

    