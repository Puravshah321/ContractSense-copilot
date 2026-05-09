"""Train a cross-encoder reranker on query-clause relevance pairs.

The expected JSONL format is:

    {"query": "...", "clause_text": "...", "label": 1.0}

This script is intentionally simple so it can be used from a notebook or from
the command line once you have prepared training pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader


DEFAULT_BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_training_examples(path: str | Path) -> list[InputExample]:
    examples: list[InputExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            examples.append(
                InputExample(
                    texts=[str(record["query"]), str(record["clause_text"])],
                    label=float(record.get("label", 0.0)),
                )
            )
    return examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a cross-encoder reranker.")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    examples = load_training_examples(args.train_path)
    if not examples:
        raise ValueError(f"No training examples found in {args.train_path}")

    model = CrossEncoder(args.base_model)
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    warmup_steps = max(1, int(len(train_dataloader) * args.epochs * 0.1))

    model.fit(
        train_dataloader=train_dataloader,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=str(args.output_path),
        show_progress_bar=True,
    )
    # Ensure a loadable checkpoint exists even if fit() skips writing metadata.
    model.save(str(args.output_path))
    print(f"saved fine-tuned reranker to {args.output_path}")


if __name__ == "__main__":
    main()