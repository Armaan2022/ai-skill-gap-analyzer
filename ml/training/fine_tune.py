"""
Fine-tune a sentence-transformer model on skill phrase → canonical skill pairs.

Uses CosineSimilarityLoss: positive pairs get label 1.0, negatives get 0.0.
The fine-tuned model is saved to ml/training/fine_tuned_model/.

Prerequisites:
    pip install sentence-transformers

Run:
    python -m ml.training.fine_tune [--base-model BAAI/bge-small-en-v1.5] [--epochs 3] [--batch-size 32]
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PAIRS_FILE = BASE_DIR / "training_pairs.json"
OUTPUT_DIR = BASE_DIR / "fine_tuned_model"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=50)
    args = parser.parse_args()

    # Lazy imports (only needed when training)
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    if not PAIRS_FILE.exists():
        print(f"Training pairs not found at {PAIRS_FILE}")
        print("Run: python -m ml.training.generate_training_data")
        return

    raw = json.loads(PAIRS_FILE.read_text(encoding="utf-8"))
    examples = [
        InputExample(texts=[row["phrase"], row["canonical"]], label=float(row["label"]))
        for row in raw
    ]
    print(f"Loaded {len(examples)} training examples")

    model = SentenceTransformer(args.base_model)
    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True)
    loss_fn = losses.CosineSimilarityLoss(model)

    total_steps = math.ceil(len(examples) / args.batch_size) * args.epochs
    print(f"Training for {args.epochs} epochs ({total_steps} steps) ...")

    model.fit(
        train_objectives=[(loader, loss_fn)],
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        output_path=str(OUTPUT_DIR),
        show_progress_bar=True,
    )

    print(f"Fine-tuned model saved to {OUTPUT_DIR}")
    print("To use it, pass model_name to SkillExtractor:")
    print(f'  SkillExtractor(model_name="{OUTPUT_DIR}")')


if __name__ == "__main__":
    main()
