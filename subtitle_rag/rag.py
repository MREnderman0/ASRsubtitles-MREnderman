from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from subtitle_rag.parsers import GlossaryEntry, ReferenceDoc


@dataclass
class RagHit:
    source: str
    raw: str
    replacement: str
    reason: str
    confidence: float


@dataclass
class RagContext:
    glossary: list[GlossaryEntry] = field(default_factory=list)
    references: list[ReferenceDoc] = field(default_factory=list)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text).strip().lower())


def apply_glossary(text: str, glossary: list[GlossaryEntry]) -> tuple[str, list[RagHit]]:
    corrected = text
    hits: list[RagHit] = []
    for entry in sorted(glossary, key=lambda item: len(item.alias), reverse=True):
        if not entry.alias or not entry.canonical:
            continue
        if entry.alias in corrected and entry.alias != entry.canonical:
            corrected = corrected.replace(entry.alias, entry.canonical)
            hits.append(
                RagHit(
                    source="glossary",
                    raw=entry.alias,
                    replacement=entry.canonical,
                    reason=entry.note or f"Matched glossary file {entry.source}",
                    confidence=1.0,
                )
            )
    return corrected, hits


def reference_snippets(text: str, references: list[ReferenceDoc], limit: int = 3) -> list[str]:
    query_chars = set(normalize_text(text))
    if not query_chars:
        return []

    scored: list[tuple[float, str]] = []
    for doc in references:
        chunks = _split_reference(doc.text)
        for chunk in chunks:
            norm = normalize_text(chunk)
            if not norm:
                continue
            overlap = len(query_chars & set(norm)) / max(len(query_chars), 1)
            sequence = SequenceMatcher(None, normalize_text(text), norm[:300]).ratio()
            score = overlap * 0.7 + sequence * 0.3
            if score > 0.15:
                scored.append((score, f"[{doc.name}] {chunk[:800]}"))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [snippet for _, snippet in scored[:limit]]


def glossary_prompt(glossary: list[GlossaryEntry], limit: int = 80) -> str:
    lines = []
    for entry in glossary[:limit]:
        note = f" ({entry.note})" if entry.note else ""
        lines.append(f"- {entry.alias} => {entry.canonical}{note}")
    return "\n".join(lines)


def _split_reference(text: str, max_len: int = 600) -> list[str]:
    parts = re.split(r"[\r\n]+", text)
    chunks: list[str] = []
    buffer = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(buffer) + len(part) > max_len and buffer:
            chunks.append(buffer)
            buffer = part
        else:
            buffer = f"{buffer}\n{part}".strip()
    if buffer:
        chunks.append(buffer)
    return chunks

