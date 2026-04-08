"""
Generate labeled training pairs from skill_meta.json aliases.

Produces ml/training/training_pairs.json with entries like:
  { "phrase": "py", "canonical": "Python", "label": 1.0 }
  { "phrase": "golang", "canonical": "Go", "label": 1.0 }

Also generates hard negatives (alias of skill A vs canonical of skill B, same category).

Run:
    python -m ml.training.generate_training_data
"""
from __future__ import annotations
import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SKILL_META = BASE_DIR.parent / "skills" / "skill_meta.json"
OUTPUT = BASE_DIR / "training_pairs.json"


def main():
    meta: dict = json.loads(SKILL_META.read_text(encoding="utf-8"))

    positives = []
    # canonical → itself
    for canonical in meta:
        positives.append({"phrase": canonical, "canonical": canonical, "label": 1.0})
    # aliases → canonical
    for canonical, info in meta.items():
        for alias in info.get("aliases", []):
            if alias:
                positives.append({"phrase": alias, "canonical": canonical, "label": 1.0})

    # Hard negatives: phrase of skill A vs canonical of a different skill in same category
    by_category: dict[str, list[str]] = {}
    for canonical, info in meta.items():
        cat = info.get("category", "default")
        by_category.setdefault(cat, []).append(canonical)

    negatives = []
    for canonical, info in meta.items():
        cat = info.get("category", "default")
        same_cat = [c for c in by_category.get(cat, []) if c != canonical]
        if not same_cat:
            continue
        # pick up to 2 hard negatives per skill
        neg_targets = random.sample(same_cat, min(2, len(same_cat)))
        for neg_canonical in neg_targets:
            # use an alias of the current skill paired with a different skill's canonical
            aliases = info.get("aliases", [canonical])
            phrase = random.choice(aliases) if aliases else canonical
            negatives.append({"phrase": phrase, "canonical": neg_canonical, "label": 0.0})

    pairs = positives + negatives
    random.shuffle(pairs)

    OUTPUT.write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    print(f"Generated {len(positives)} positives + {len(negatives)} negatives = {len(pairs)} pairs")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
