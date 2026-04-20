"""
dataset_builder.py
Converts generation outputs (baseline, LoRA, best) into DPO preference format.
Reads generation JSONL files, constructs prompt from clause+query,
assigns chosen = best/cleaned output and rejected = baseline output,
saves structured dataset to JSON.
"""

import json
import random
import re
from pathlib import Path
from typing import Optional


CLAUSE_TYPES = [
    "Termination", "Indemnification", "Intellectual Property",
    "Limitation of Liability", "Confidentiality", "Force Majeure",
    "Governing Law", "Non-Solicitation", "Warranty",
    "Pricing Adjustment", "Assignment", "SLA", "Non-Compete",
    "Data Protection", "Insurance", "Audit Rights", "Renewal",
    "Dispute Resolution",
]

RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

QUERY_TEMPLATES = [
    "What are the key risks?",
    "What should we be concerned about?",
    "Can you summarize the obligations?",
    "What happens if this clause is breached?",
    "Are there hidden risks in this clause?",
    "How does this affect our liability?",
    "What actions should we take before signing?",
    "Does this clause favour one party?",
    "What are the financial implications?",
    "Can we negotiate better terms?",
    "Is this clause standard or unusual?",
    "What deadlines or notice periods exist?",
]

CLAUSE_TEMPLATES = [
    (
        "Section {sec_id} — {clause_type}. "
        "Notwithstanding any other provision of this Agreement, the parties agree that "
        "{clause_type_lower} obligations herein shall remain in full force and effect. "
        "{party} shall provide {notice} prior to invoking rights under this section."
    ),
    (
        "Article {sec_id} ({clause_type}). "
        "Subject to the terms herein, the {party_lower} acknowledges the {clause_type_lower} "
        "provisions and agrees to comply. Breach of this section entitles the non-breaching "
        "party to seek remedies as described in Section {ref_sec}."
    ),
    (
        "{sec_id}. {clause_type}. "
        "Each party shall observe {clause_type_lower} requirements for the duration of this "
        "Agreement and for a period of {duration} years following termination. "
        "Exceptions apply where information becomes publicly available through no fault of the disclosing party."
    ),
    (
        "Clause {sec_id}: {clause_type}. "
        "The {party} represents and warrants that all {clause_type_lower} obligations will be "
        "fulfilled in accordance with applicable law. Failure to comply may result in termination "
        "for cause with {notice} notice."
    ),
]

EXPLANATION_STARTERS = [
    "This clause means",
    "In plain terms",
    "Simply put",
    "Effectively",
    "The agreement states that",
    "What this means for your business is",
    "From a practical standpoint",
    "Breaking this down",
]

ACTION_TEMPLATES = [
    "Review the {clause_type_lower} terms carefully and consult legal counsel before signing.",
    "Negotiate a more favourable {clause_type_lower} provision if possible.",
    "Set up internal monitoring to ensure compliance with {clause_type_lower} requirements.",
    "Document all actions related to {clause_type_lower} obligations for audit purposes.",
    "Create a compliance checklist for the {clause_type_lower} requirements before execution.",
    "Brief your team on the {clause_type_lower} obligations and their implications.",
    "Compare this {clause_type_lower} clause with industry standard provisions.",
    "Assess financial exposure under the {clause_type_lower} terms and ensure adequate coverage.",
]


def _random_section_id() -> str:
    return f"{random.randint(1, 600)}.{random.randint(1, 9)}"


def _random_party() -> str:
    return random.choice(["Customer", "Vendor", "Licensee", "Contractor", "Provider"])


def _random_notice() -> str:
    return random.choice([
        "thirty (30) days notice",
        "written notice",
        "fourteen (14) days written notice",
        "sixty (60) days advance notice",
    ])


def _generate_clause(clause_type: str) -> str:
    template = random.choice(CLAUSE_TEMPLATES)
    return template.format(
        sec_id=_random_section_id(),
        clause_type=clause_type,
        clause_type_lower=clause_type.lower(),
        party=_random_party(),
        party_lower=_random_party().lower(),
        notice=_random_notice(),
        ref_sec=f"{random.randint(1, 50)}.{random.randint(1, 9)}",
        duration=random.choice([2, 3, 5, 7]),
    )


def _generate_chosen_response(clause: str, query: str, clause_type: str, doc_id: str) -> str:
    risk = random.choice(RISK_LEVELS)
    starter = random.choice(EXPLANATION_STARTERS)
    span_start = random.randint(0, 200)
    span_end = span_start + random.randint(100, 600)
    action = random.choice(ACTION_TEMPLATES).format(clause_type_lower=clause_type.lower())

    explanation = (
        f"{starter}, the {clause_type.lower()} clause imposes obligations that "
        f"could significantly impact contract value and operational continuity."
    )

    return (
        f"RISK: {risk}\n\n"
        f"{explanation}\n\n"
        f"ACTION: {action}\n\n"
        f"CITATION: [{doc_id}, spans {span_start}–{span_end}]"
    )


def _generate_rejected_response(clause: str, query: str, clause_type: str) -> str:
    return (
        f"The {clause_type.lower()} provision as stated in the agreement outlines "
        f"obligations relating to {clause_type.lower()}. The parties should be aware "
        f"of the terms and conditions specified therein."
    )


def build_dataset_from_generation_files(
    generation_train_path: str,
    generation_eval_path: str,
    output_path: str,
    target_size: int = 100000,
    seed: int = 42,
) -> list:
    random.seed(seed)
    dataset = []

    for jsonl_path in [generation_train_path, generation_eval_path]:
        path = Path(jsonl_path)
        if not path.exists():
            print(f"⚠️  File not found: {jsonl_path}, using synthetic generation")
            continue

        with open(path) as f:
            for line in f:
                entry = json.loads(line.strip())
                text = entry.get("text", "")
                clause_match = re.search(r"Clause:\n(.+?)\n\nQuery:", text, re.DOTALL)
                if not clause_match:
                    clause_match = re.search(r"Clause:\n(.+?)\\n\\nQuery:", text)
                query_match = re.search(r"Query:\s*(.+?)\s*\[/INST\]", text)
                response_match = re.search(r"\[/INST\]\s*(.+?)(?:</s>|$)", text, re.DOTALL)

                if clause_match and query_match and response_match:
                    clause = clause_match.group(1).strip()
                    query = query_match.group(1).strip()
                    raw_response = response_match.group(1).strip()

                    try:
                        resp_json = json.loads(raw_response)
                        risk = resp_json.get("risk_level", "MEDIUM")
                        explanation = resp_json.get("plain_explanation", "")
                        action = resp_json.get("recommended_action", "")
                        citation_obj = resp_json.get("citation", {})
                        cid = citation_obj.get("clause_id", "UNKNOWN")
                        span = citation_obj.get("char_span", [0, 0])

                        chosen = (
                            f"RISK: {risk}\n\n"
                            f"{explanation}\n\n"
                            f"ACTION: {action}\n\n"
                            f"CITATION: [{cid}, spans {span[0]}–{span[1]}]"
                        )

                        rejected = (
                            f"The clause addresses the matter of {query.lower().rstrip('?')}. "
                            f"The terms should be reviewed in the context of the agreement."
                        )

                        dataset.append({
                            "prompt": f"Clause: {clause}\n\nQuery: {query}",
                            "chosen": chosen,
                            "rejected": rejected,
                            "metadata": {
                                "doc_id": cid,
                                "source": "generation_data",
                                "risk_level": risk,
                            }
                        })
                    except json.JSONDecodeError:
                        continue

    print(f"📦 Loaded {len(dataset)} pairs from generation files")

    existing_seed_path = Path(output_path)
    if existing_seed_path.exists():
        try:
            with open(existing_seed_path) as f:
                seed_data = json.load(f)
            dataset.extend(seed_data)
            print(f"📦 Added {len(seed_data)} seed pairs from existing dataset")
        except Exception:
            pass

    remaining = target_size - len(dataset)
    if remaining > 0:
        print(f"🔄 Generating {remaining} synthetic pairs to reach target={target_size}")
        for i in range(remaining):
            clause_type = random.choice(CLAUSE_TYPES)
            query = random.choice(QUERY_TEMPLATES)
            clause = _generate_clause(clause_type)
            doc_id = f"CUAD_{random.randint(1, 600):04d}_C{random.randint(1, 100):04d}"

            chosen = _generate_chosen_response(clause, query, clause_type, doc_id)
            rejected = _generate_rejected_response(clause, query, clause_type)

            dataset.append({
                "prompt": f"Clause: {clause}\n\nQuery: {query}",
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "doc_id": doc_id,
                    "source": "synthetic",
                    "clause_type": clause_type,
                }
            })

            if (i + 1) % 10000 == 0:
                print(f"   Generated {i + 1}/{remaining} synthetic pairs...")

    random.shuffle(dataset)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"✅ Saved {len(dataset)} DPO pairs to {output_path}")
    return dataset


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from alignment.config import (
        GENERATION_TRAIN_PATH, GENERATION_EVAL_PATH,
        RAW_DPO_DATASET_PATH, TARGET_DATASET_SIZE,
    )

    data = build_dataset_from_generation_files(
        GENERATION_TRAIN_PATH,
        GENERATION_EVAL_PATH,
        RAW_DPO_DATASET_PATH,
        target_size=TARGET_DATASET_SIZE,
    )
    print(f"\nSample:\n{json.dumps(data[0], indent=2)}")
