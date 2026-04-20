"""
dataset_augmenter.py
Expands the DPO dataset using rule-based augmentation:
- Paraphrases queries
- Injects clause variation (different section IDs, parties, notice periods)
- Generates additional chosen/rejected pairs
- Increases diversity for better model generalization
"""

import json
import random
import re
import copy
from pathlib import Path
from typing import Optional


QUERY_PARAPHRASES = {
    "What are the key risks?": [
        "What should concern us about this clause?",
        "Highlight the main dangers here.",
        "What risks does this clause create?",
        "Is there anything risky in this section?",
        "Summarize the threats from this provision.",
    ],
    "What should we be concerned about?": [
        "Are there any red flags here?",
        "What potential issues should we watch for?",
        "Should we worry about this clause?",
        "What are the downsides of this provision?",
    ],
    "Can you summarize the obligations?": [
        "What are we required to do under this clause?",
        "List the key obligations in this section.",
        "What does this clause obligate us to?",
        "Summarize the duties described here.",
    ],
    "What happens if this clause is breached?": [
        "What are the consequences of violating this?",
        "If we break this clause, what happens?",
        "Describe the penalties for non-compliance.",
        "What remedies exist for breach?",
    ],
}

PARTY_ALTERNATIVES = ["Company", "Client", "Vendor", "Contractor", "Provider", "Licensee", "Supplier"]
NOTICE_ALTERNATIVES = [
    "thirty (30) days written notice",
    "fourteen (14) days advance notice",
    "sixty (60) days prior written notice",
    "ninety (90) days notice",
    "written notice within ten (10) business days",
]

EXPLANATION_VARIANTS = [
    "This clause means",
    "In simple terms",
    "Put plainly",
    "What this boils down to is",
    "Practically speaking",
    "The bottom line is",
    "In everyday language",
    "Breaking this down for business use",
]

ACTION_VARIANTS = [
    "Review {ct} terms carefully and consult legal counsel.",
    "Negotiate more favourable {ct} provisions if possible.",
    "Set up compliance monitoring for {ct} requirements.",
    "Document all activities related to {ct} obligations.",
    "Create an internal checklist for {ct} compliance.",
    "Ensure your team understands the {ct} obligations before execution.",
    "Compare this {ct} with industry benchmarks and best practices.",
    "Assess your financial exposure under the {ct} terms.",
]


def _paraphrase_query(query: str) -> str:
    for original, paras in QUERY_PARAPHRASES.items():
        if query.strip() == original:
            return random.choice(paras)
    return query


def _inject_clause_variation(clause: str) -> str:
    result = clause

    result = re.sub(
        r"Section \d+\.\d+",
        f"Section {random.randint(1, 999)}.{random.randint(1, 9)}",
        result,
    )

    for party in PARTY_ALTERNATIVES:
        if party in result:
            new_party = random.choice([p for p in PARTY_ALTERNATIVES if p != party])
            result = result.replace(party, new_party, 1)
            break

    for notice in NOTICE_ALTERNATIVES:
        if notice.split()[0].lower() in result.lower():
            new_notice = random.choice(NOTICE_ALTERNATIVES)
            result = re.sub(
                r"(thirty|fourteen|sixty|ninety|ten)\s*\(\d+\)\s*(days?|business days?)\s*(written\s+)?notice",
                new_notice,
                result,
                flags=re.IGNORECASE,
                count=1,
            )
            break

    return result


def _vary_chosen(chosen: str, clause_type: str = "clause") -> str:
    result = chosen

    for exp in EXPLANATION_VARIANTS:
        if exp in result:
            new_exp = random.choice([e for e in EXPLANATION_VARIANTS if e != exp])
            result = result.replace(exp, new_exp, 1)
            break

    action_match = re.search(r"ACTION:\s*(.+?)(?:\n|$)", result)
    if action_match:
        new_action = random.choice(ACTION_VARIANTS).format(ct=clause_type.lower())
        result = result.replace(action_match.group(1), new_action, 1)

    return result


def _vary_rejected(rejected: str) -> str:
    fillers = [
        " as per the agreement terms.",
        " pursuant to the provisions herein.",
        " subject to the conditions specified.",
        " in accordance with the contractual framework.",
        " notwithstanding any other provisions.",
    ]
    return rejected.rstrip(".") + random.choice(fillers)


def augment_dataset(
    input_path: str,
    output_path: str,
    augmentation_factor: int = 10,
    target_size: Optional[int] = None,
    seed: int = 42,
) -> list:
    random.seed(seed)

    with open(input_path) as f:
        original = json.load(f)

    print(f"📥 Loaded {len(original)} pairs from {input_path}")

    augmented = list(original)

    if target_size:
        needed = target_size - len(original)
        effective_factor = max(1, needed // len(original) + 1)
    else:
        effective_factor = augmentation_factor
        needed = len(original) * effective_factor

    print(f"🔄 Augmenting with factor ~{effective_factor} (target: {target_size or needed + len(original)})")

    generated = 0
    while generated < needed:
        base_entry = random.choice(original)
        new_entry = copy.deepcopy(base_entry)

        prompt = new_entry["prompt"]
        clause_match = re.search(r"Clause:\s*(.+?)\n\nQuery:\s*(.+)", prompt, re.DOTALL)
        if clause_match:
            clause = clause_match.group(1)
            query = clause_match.group(2).strip()

            new_clause = _inject_clause_variation(clause)
            new_query = _paraphrase_query(query) if random.random() > 0.3 else query
            new_entry["prompt"] = f"Clause: {new_clause}\n\nQuery: {new_query}"

        clause_type = new_entry.get("metadata", {}).get("clause_type", "clause")
        new_entry["chosen"] = _vary_chosen(new_entry["chosen"], clause_type)
        new_entry["rejected"] = _vary_rejected(new_entry["rejected"])

        if "metadata" in new_entry:
            new_entry["metadata"]["source"] = "augmented"
            new_entry["metadata"]["aug_from"] = base_entry.get("metadata", {}).get("doc_id", "unknown")

        augmented.append(new_entry)
        generated += 1

        if generated % 10000 == 0:
            print(f"   Augmented {generated}/{needed}...")

    if target_size and len(augmented) > target_size:
        random.shuffle(augmented)
        augmented = augmented[:target_size]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(augmented, f, indent=2)

    print(f"✅ Saved {len(augmented)} augmented pairs to {output_path}")
    return augmented


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from alignment.config import VALIDATED_DATASET_PATH, AUGMENTED_DATASET_PATH, TARGET_DATASET_SIZE

    augment_dataset(
        VALIDATED_DATASET_PATH,
        AUGMENTED_DATASET_PATH,
        target_size=TARGET_DATASET_SIZE,
    )
