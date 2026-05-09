"""
Semantic + Structural Document Chunker.
Splits contract PDFs into evidence-grade chunks with full metadata.
"""
import re
from dataclasses import dataclass, field, asdict


@dataclass
class Chunk:
    chunk_id: str
    text: str
    section: str
    clause_id: str
    page: int
    char_start: int
    char_end: int
    token_count: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


_SECTION_RE = re.compile(
    r"^(?:"
    r"(?:section|article|clause|part)\s+\d+"             # Section 1, Article 2
    r"|\d+\.\d*"                                          # 1.1, 2.
    r"|\([a-z]\)"                                         # (a), (b)
    r"|\([ivx]+\)"                                        # (i), (ii)
    r")",
    re.IGNORECASE,
)

_HEADING_RE = re.compile(
    r"^(?:(?:section|article|clause|part|schedule|exhibit|annex|appendix)"
    r"\s+[\d\.]+|\d+(?:\.\d+)*)\s*[\-\:\.]?\s*(.+)",
    re.IGNORECASE,
)

_CLAUSE_KEYWORDS = [
    "termination", "indemnif", "liability", "confidential", "intellectual property",
    "force majeure", "governing law", "warranty", "payment", "non-compete",
    "non-solicitation", "data protection", "insurance", "assignment", "dispute",
    "arbitration", "notice", "amendment", "waiver", "severability", "renewal",
    "term", "remedies", "permitted disclosure", "need to know", "survival",
    "non-solicitation", "governing law", "authority", "no conflict",
]

MIN_CHUNK_TOKENS = 8
MAX_CHUNK_TOKENS = 400
TARGET_CHUNK_TOKENS = 250


def _count_tokens(text):
    return len(text.split())


def _detect_section_name(text):
    text_lower = text[:200].lower()
    match = _HEADING_RE.match(text.strip())
    if match:
        return match.group(1).strip().rstrip(".:- ")

    for kw in _CLAUSE_KEYWORDS:
        if kw in text_lower:
            return kw.replace("indemnif", "indemnification").title()
    return "General"


def _detect_clause_id(text):
    patterns = [
        r"((?:Section|Article|Clause)\s+\d+(?:\.\d+)*)",
        r"^(\d+(?:\.\d+)*\.)",
        r"^(\d+\.\d+(?:\.\d+)*)",
        r"^(\(\w+\))",
    ]
    for p in patterns:
        m = re.search(p, text[:100], re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return "unidentified"


def _split_into_paragraphs(text):
    """Split text by double newlines or section headers."""
    raw_splits = re.split(r"\n\s*\n", text)
    paragraphs = []
    for block in raw_splits:
        block = block.strip()
        if not block:
            continue
        sub_splits = re.split(
            r"\n(?=(?:Section|Article|Clause|ARTICLE|SECTION)\s+\d|\d{1,2}\.\s+[A-Z])",
            block,
        )
        for s in sub_splits:
            s = s.strip()
            if s:
                paragraphs.append(s)
    return paragraphs


def _split_long_paragraph(text, max_tokens=MAX_CHUNK_TOKENS):
    """Split a long paragraph at sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        sent_len = _count_tokens(sent)
        if current_len + sent_len > max_tokens and current:
            chunks.append(" ".join(current))
            current = [sent]
            current_len = sent_len
        else:
            current.append(sent)
            current_len += sent_len

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(text, source_name="uploaded_document"):
    """
    Chunk a contract document into evidence-grade pieces.

    Returns list of Chunk objects with full metadata.
    """
    paragraphs = _split_into_paragraphs(text)
    chunks = []
    char_offset = 0

    for para in paragraphs:
        token_count = _count_tokens(para)

        if token_count < MIN_CHUNK_TOKENS:
            char_offset += len(para) + 2
            continue

        if token_count > MAX_CHUNK_TOKENS:
            sub_chunks = _split_long_paragraph(para)
        else:
            sub_chunks = [para]

        for sub in sub_chunks:
            sub_tokens = _count_tokens(sub)
            if sub_tokens < MIN_CHUNK_TOKENS:
                continue

            start = text.find(sub, max(0, char_offset - 50))
            if start == -1:
                start = char_offset
            end = start + len(sub)

            page = text[:start].count("\f") + 1

            chunk = Chunk(
                chunk_id=f"{source_name}_chunk_{len(chunks):03d}",
                text=sub,
                section=_detect_section_name(sub),
                clause_id=_detect_clause_id(sub),
                page=page,
                char_start=start,
                char_end=end,
                token_count=sub_tokens,
                metadata={"source": source_name},
            )
            chunks.append(chunk)

        char_offset += len(para) + 2

    return chunks
