"""Clause segmentation pipeline for CUAD contracts.

This module loads the CUAD dataset from disk, extracts full contract text,
splits it into clause-like chunks, and writes one JSONL record per clause.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Iterator

from datasets import load_from_disk

SECTION_SPLIT_PATTERN = re.compile(
    r"\n(?=(?:\d+\.|SECTION\s+\d+|ARTICLE\s+[IVXLC]+)\b)",
    flags=re.IGNORECASE,
)


def extract_contract_text(sample: dict) -> str:
    """Return normalized contract text from a CUAD sample."""
    if "context" in sample and sample["context"]:
        return str(sample["context"]).strip()

    pdf = sample.get("pdf")
    if pdf is None:
        return ""

    pages = []
    for page in pdf.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)

    return "\n".join(pages).strip()


def split_into_clauses(text: str) -> list[str]:
    """Split a contract into clause-like segments.

    The heuristic prefers numbered sections and article headers. If no split
    points are found, the full text is returned as one segment.
    """
    if not text:
        return []

    chunks = [chunk.strip() for chunk in SECTION_SPLIT_PATTERN.split(text)]
    clauses = [chunk for chunk in chunks if len(chunk) > 80]
    return clauses if clauses else [text.strip()]


def iter_cuad_clauses(dataset_path: str | Path) -> Iterator[dict]:
    """Yield clause records for every available CUAD split."""
    dataset = load_from_disk(str(dataset_path))

    for split_name in dataset.keys():
        split = dataset[split_name]
        for contract_index, sample in enumerate(split):
            contract_text = extract_contract_text(sample)
            clauses = split_into_clauses(contract_text)
            contract_id = sample.get("id") or f"{split_name}_{contract_index:05d}"

            for clause_index, clause_text in enumerate(clauses):
                yield {
                    "split": split_name,
                    "contract_id": contract_id,
                    "clause_id": f"{contract_id}_clause_{clause_index:03d}",
                    "clause_index": clause_index,
                    "num_clauses": len(clauses),
                    "char_count": len(clause_text),
                    "clause_text": clause_text,
                }


def write_jsonl(records: Iterable[dict], output_path: str | Path) -> int:
    """Write records to JSONL and return the number of rows written."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as file_handle:
        for record in records:
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build clause-level CUAD JSONL.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/raw/cuad"),
        help="Path to the CUAD dataset saved with datasets.save_to_disk().",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/clauses.jsonl"),
        help="Where to write the processed clause JSONL file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    record_count = write_jsonl(iter_cuad_clauses(args.dataset_path), args.output_path)
    print(f"saved {record_count} clause records to {args.output_path}")


if __name__ == "__main__":
    main()
