"""
Legal Concept Tagger — Production Upgrade.

Stamps every chunk at index-time with legal function tags using:
1. Keyword pattern matching (fast, zero-latency)
2. Multi-label classification (a chunk can belong to multiple types)

Legal clause types:
  - termination           (who can terminate, under what conditions, notice)
  - cure_period           (opportunity to fix breach before termination)
  - liability_cap         (maximum damages, liability limits)
  - gross_negligence      (carve-outs for gross negligence / willful misconduct)
  - survival              (clauses that persist after termination)
  - subcontractor         (contractor = you, attribution of third-party acts)
  - confidentiality       (non-disclosure, information protection)
  - data_protection       (GDPR, personal data, security obligations)
  - force_majeure         (third-party events, acts of god, excluded events)
  - payment               (invoice, fees, payment obligations)
  - indemnification       (hold harmless, indemnify, defend)
  - ip                    (ownership, assignment, work product)
  - dispute               (arbitration, governing law, jurisdiction)
"""
import re

# ── Legal Clause Type → Pattern Map ────────────────────────────────────────
# Each entry: (tag, [list of regex patterns])
# Patterns use word-boundary anchors for precision.

_CLAUSE_PATTERNS = [
    (
        "termination",
        [
            r"\bterminat\w*\b",
            r"\bright\s+to\s+terminat\w*\b",
            r"\bmay\s+terminat\w*\b",
            r"\bshall\s+terminat\w*\b",
            r"\bnotice\s+of\s+terminat\w*\b",
            r"\beffect\s+of\s+terminat\w*\b",
        ],
    ),
    (
        "cure_period",
        [
            r"\bcure\b",
            r"\bcure\s+period\b",
            r"\bremediat\w+\b",
            r"\bopportunity\s+to\s+(cure|remediat|correct|fix)\b",
            r"\b(\d+)\s*days?\s+(to\s+cure|to\s+correct|cure\s+period|notice\s+period)\b",
            r"\bnotice\s+and\s+opportunit\w*\b",
            r"\bfail\w*\s+to\s+cure\b",
        ],
    ),
    (
        "liability_cap",
        [
            r"\blimitation\s+of\s+liabilit\w*\b",
            r"\bliabilit\w*\s+(cap|limit|ceiling|maximum)\b",
            r"\bshall\s+not\s+(exceed|be\s+liable)\b",
            r"\bmaximum\s+(aggregate\s+)?liabilit\w*\b",
            r"\bin\s+no\s+event\s+(shall|will)\b",
            r"\bexclud\w*\s+(consequential|indirect|incidental|punitive)\b",
            r"\bcap\s+on\s+(damages|liabilit\w*)\b",
            r"\btotal\s+liabilit\w*\s+(shall\s+not|is\s+limited)\b",
        ],
    ),
    (
        "gross_negligence",
        [
            r"\bgross\s+negligence\b",
            r"\bwilful\s+misconduct\b",
            r"\bwillful\s+misconduct\b",
            r"\bfraud\w*\b",
            r"\bintentional\s+(breach|misconduct|act)\b",
            r"\bcarve[\-\s]out\b",
        ],
    ),
    (
        "survival",
        [
            r"\bsurviv\w*\s+(terminat\w*|expir\w*|expiration)\b",
            r"\bshall\s+surviv\w*\b",
            r"\bcontinue\s+(in\s+force|to\s+(apply|govern|bind))\s+(after|following|upon)\b",
            r"\bobligation\w*\s+(remain|persist|continue)\s+(after|following)\b",
            r"\b(after|following|upon)\s+(terminat\w*|expir\w*)\b.*\b(remain|continue|surviv)\w*\b",
        ],
    ),
    (
        "subcontractor",
        [
            r"\bsubcontract\w*\b",
            r"\bthird[\-\s]party\s+(act\w*|conduct|negligence|breach)\b",
            r"\bact\w*\s+(of|by|undertaken\s+by)\s+(contractor\w*|subcontractor\w*)\b",
            r"\bdeemed\s+(to\s+have\s+been\s+taken|taken)\s+by\s+you\b",
            r"\bcontractor\w*\s+shall\s+be\s+deemed\b",
            r"\bresponsib\w+\s+for\s+(the\s+acts?|acts?\s+of|action\w+\s+of)\s+(sub)?contractor\w*\b",
            r"\battribut\w+\s+to\b",
            r"\bvicariously\s+liabl\w*\b",
            r"\bprincipal\s+and\s+agent\b",
        ],
    ),
    (
        "confidentiality",
        [
            r"\bconfidential\w*\s+information\b",
            r"\bnon[\-\s]disclosure\b",
            r"\bshall\s+(not\s+)?(disclose|reveal|divulge)\b",
            r"\bduty\s+of\s+confidentialit\w*\b",
            r"\bobligat\w+\s+(of|to\s+maintain)\s+confidentialit\w*\b",
            r"\bprotect\w*\s+(confidential|proprietary)\b",
        ],
    ),
    (
        "data_protection",
        [
            r"\bpersonal\s+data\b",
            r"\bdata\s+protection\b",
            r"\bdata\s+(breach|incident|security)\b",
            r"\bsecurity\s+(obligations?|requirements?|standards?)\b",
            r"\bgdpr\b",
            r"\bdata\s+(processor|controller|subject)\b",
            r"\bcyber\w*\b",
            r"\bransomware\b",
            r"\bencrypt\w*\b",
            r"\bsecurity\s+incident\b",
        ],
    ),
    (
        "force_majeure",
        [
            r"\bforce\s+majeure\b",
            r"\bact\s+of\s+god\b",
            r"\bbeyond\s+(the\s+)?(reasonable\s+)?control\b",
            r"\bexcluded\s+(event\w*|cause\w*)\b",
            r"\bthird[\-\s]party\s+(event\w*|failure|action\w*)\b",
            r"\bexcused\s+(from\s+)?(performance|liabilit\w*)\b",
        ],
    ),
    (
        "payment",
        [
            r"\bpayment\w*\s+(term\w*|obligat\w*|schedul\w*)\b",
            r"\binvoice\w*\b",
            r"\bfee\w*\s+(due|payable|invoiced)\b",
            r"\bamount\w*\s+(due|owed|payable)\b",
            r"\bpayable\s+(upon|after|within)\b",
            r"\bnet\s+\d+\s+(day\w*)\b",
            r"\boverdue\b",
            r"\blate\s+(payment|fees?|charge\w*)\b",
        ],
    ),
    (
        "indemnification",
        [
            r"\bindemnif\w*\b",
            r"\bhold\s+harmless\b",
            r"\bdefend\w*\s+(and\s+)?(indemnif\w*|hold\s+harmless)\b",
            r"\bindemnitor\b",
            r"\bindemnitee\b",
            r"\bclaim\w*\s+(indemnit\w*|arising\s+from)\b",
        ],
    ),
    (
        "ip",
        [
            r"\bintellectual\s+property\b",
            r"\bwork\s+product\b",
            r"\bproprietary\s+right\w*\b",
            r"\bcopyright\b",
            r"\btrademark\b",
            r"\bpatent\b",
            r"\blicens\w+\b",
            r"\bassignment\s+of\s+(ip|intellectual|rights?)\b",
        ],
    ),
    (
        "dispute",
        [
            r"\barbitrat\w*\b",
            r"\bgoverning\s+law\b",
            r"\bjurisdiction\b",
            r"\bdispute\s+resolution\b",
            r"\bmediat\w*\b",
            r"\blitigation\b",
        ],
    ),
]

# ── Compiled pattern cache ──────────────────────────────────────────────────
_COMPILED = [
    (tag, [re.compile(p, re.IGNORECASE) for p in patterns])
    for tag, patterns in _CLAUSE_PATTERNS
]


def tag_chunk(chunk) -> list:
    """
    Tag a Chunk with all matching legal concept types.

    Returns a list of tag strings, e.g. ["termination", "cure_period"]
    Returns ["general"] if no specific type is detected.
    """
    text = f"{chunk.section} {chunk.text}".strip()
    tags = []
    for tag, patterns in _COMPILED:
        for pat in patterns:
            if pat.search(text):
                tags.append(tag)
                break  # Only add a tag once per category
    return tags if tags else ["general"]


def tag_all_chunks(chunks: list) -> list:
    """
    Tag every chunk in a list in-place (adds 'legal_tags' to chunk.metadata).
    Returns the same list for chaining.
    """
    for chunk in chunks:
        tags = tag_chunk(chunk)
        chunk.metadata["legal_tags"] = tags
    return chunks


# ── Tag → Legal Concept Map for retrieval lookup ───────────────────────────
# Maps a query concept to the clause tags most likely to contain relevant text

CONCEPT_TO_TAGS = {
    "termination":          ["termination", "cure_period", "survival", "payment"],
    "cure_period":          ["cure_period", "termination"],
    "liability":            ["liability_cap", "gross_negligence", "indemnification"],
    "gross_negligence":     ["gross_negligence", "liability_cap"],
    "survival":             ["survival", "confidentiality", "termination"],
    "subcontractor":        ["subcontractor", "force_majeure", "indemnification", "liability_cap"],
    "confidentiality":      ["confidentiality", "data_protection", "survival"],
    "data_use":             ["data_protection", "confidentiality"],
    "force_majeure":        ["force_majeure", "subcontractor"],
    "financial_commitment": ["payment", "liability_cap", "indemnification"],
    "risk":                 ["liability_cap", "gross_negligence", "data_protection", "subcontractor"],
    "dispute":              ["dispute"],
    "ip":                   ["ip"],
    "compliance":           ["data_protection", "confidentiality"],
}
