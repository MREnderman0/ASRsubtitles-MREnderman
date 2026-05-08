from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from subtitle_rag.cleaning import CleanedSegment


@dataclass
class SubtitleItem:
    index: int
    start: float
    end: float
    text: str


PUNCTUATION = "，。！？；：、,.!?;:"
SENTENCE_PUNCTUATION = "。！？；.!?;"
COMMA_PUNCTUATION = "，,、"
SOFT_BREAKS = ("但是", "所以", "因为", "如果", "然后", "以及", "并且", "或者", "同时", "而且")
PROTECTED_PHRASES = (
    "罗伯特波伊尔",
    "利莫里克郡",
    "点石成金",
    "炼金术士",
    "炼金术",
    "学术界",
    "主流思想",
    "变为",
    "成为",
    "作为",
    "因为",
    "所以",
    "但是",
    "如果",
    "以及",
    "并且",
    "或者",
    "同时",
    "而且",
    "波伊尔",
    "爱尔兰",
    "生成物",
    "生存物",
    "古堡",
    "黄金",
)
SHORT_TAIL_CHARS = 5


def make_subtitles(
    segments: list[CleanedSegment],
    max_chars: int = 17,
    protected_phrases: list[str] | None = None,
) -> list[SubtitleItem]:
    subtitles: list[SubtitleItem] = []
    for segment in segments:
        phrase_set = _segment_protected_phrases(segment, protected_phrases)
        pieces = split_text(segment.cleaned_text, max_chars=max_chars, protected_phrases=phrase_set)
        if not pieces:
            continue
        timings = _piece_timings(segment, pieces)
        for piece, (start, end) in zip(pieces, timings):
            subtitles.append(SubtitleItem(index=len(subtitles) + 1, start=start, end=end, text=piece))
    return subtitles


def split_text(text: str, max_chars: int = 17, protected_phrases: list[str] | None = None) -> list[str]:
    text = _normalize_subtitle_text(text)
    if not text:
        return []
    phrases = _normalize_protected_phrases(protected_phrases)
    sentences = _sentence_chunks(text)
    pieces: list[str] = []
    for sentence in sentences:
        pieces.extend(_split_long_sentence(sentence, max_chars, phrases))
    return [piece for piece in (sanitize_subtitle_text(piece) for piece in pieces) if piece]


def write_srt(items: list[SubtitleItem], path: str | Path) -> None:
    path = Path(path)
    lines: list[str] = []
    for item in items:
        lines.extend(
            [
                str(item.index),
                f"{format_timestamp(item.start)} --> {format_timestamp(item.end)}",
                item.text,
                "",
            ]
        )
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def format_timestamp(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    whole_seconds = int(seconds % 60)
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    if milliseconds == 1000:
        whole_seconds += 1
        milliseconds = 0
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def display_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def sanitize_subtitle_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(f"[{re.escape(PUNCTUATION)}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_subtitle_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(f"\\s+([{re.escape(PUNCTUATION)}])", r"\1", text)
    return text.strip()


def _sentence_chunks(text: str) -> list[str]:
    chunks = re.split(f"(?<=[{re.escape(SENTENCE_PUNCTUATION)}])", text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _split_long_sentence(sentence: str, max_chars: int, protected_phrases: list[str]) -> list[str]:
    if display_len(sanitize_subtitle_text(sentence)) <= max_chars:
        return [sentence]

    result: list[str] = []
    current = sentence
    while display_len(sanitize_subtitle_text(current)) > max_chars:
        split_at = _find_split_index(current, max_chars, protected_phrases)
        head = current[:split_at].strip(PUNCTUATION + " ")
        if head:
            result.append(head)
        current = current[split_at:].strip()
    if current:
        result.append(current.strip())
    return [item for item in result if item]


def _find_split_index(text: str, max_chars: int, protected_phrases: list[str]) -> int:
    candidates = _candidate_split_indices(text, max_chars, protected_phrases, allow_short_tail=False)
    if candidates:
        return max(candidates, key=lambda idx: _split_score(text, idx, protected_phrases))

    candidates = _candidate_split_indices(text, max_chars, protected_phrases, allow_short_tail=True)
    if candidates:
        return max(candidates, key=lambda idx: _split_score(text, idx, protected_phrases))

    hard_limit = _index_for_display_len(text, max_chars)
    return max(1, _safe_split_index(text, hard_limit, max_chars, protected_phrases))


def _candidate_split_indices(text: str, max_chars: int, protected_phrases: list[str], allow_short_tail: bool) -> list[int]:
    candidates: list[int] = []
    for split_at in range(1, len(text)):
        if _valid_split(text, split_at, max_chars, protected_phrases, allow_short_tail=allow_short_tail):
            candidates.append(split_at)
    return candidates


def _valid_split(
    text: str,
    split_at: int,
    max_chars: int,
    protected_phrases: list[str],
    allow_short_tail: bool = False,
) -> bool:
    if _inside_protected_phrase(text, split_at, protected_phrases):
        return False
    if _is_bad_comma_split(text, split_at):
        return False

    left_len = display_len(sanitize_subtitle_text(text[:split_at]))
    right_len = display_len(sanitize_subtitle_text(text[split_at:]))
    if left_len == 0 or right_len == 0:
        return False
    if left_len > max_chars:
        return False
    if not allow_short_tail and left_len <= SHORT_TAIL_CHARS and text[split_at - 1 : split_at] not in PUNCTUATION:
        return False
    if allow_short_tail:
        return True
    return right_len > SHORT_TAIL_CHARS or right_len > max_chars


def _is_bad_comma_split(text: str, split_at: int) -> bool:
    left = text[:split_at]
    right = text[split_at:]
    comma = ""
    if left[-1:] in COMMA_PUNCTUATION:
        comma = left[-1:]
    elif right[:1] in COMMA_PUNCTUATION:
        comma = right[:1]
    if not comma:
        return False

    left_part = left.rstrip(COMMA_PUNCTUATION + " ")
    right_part = right.lstrip(COMMA_PUNCTUATION + " ")
    left_len = _nearest_clause_len(left_part, from_right=True)
    right_len = _nearest_clause_len(right_part, from_right=False)
    return left_len <= SHORT_TAIL_CHARS or right_len <= SHORT_TAIL_CHARS


def _nearest_clause_len(text: str, from_right: bool) -> int:
    separators = SENTENCE_PUNCTUATION + COMMA_PUNCTUATION
    if from_right:
        parts = re.split(f"[{re.escape(separators)}]", text)
        target = parts[-1] if parts else text
    else:
        parts = re.split(f"[{re.escape(separators)}]", text)
        target = parts[0] if parts else text
    return display_len(sanitize_subtitle_text(target))


def _split_score(text: str, split_at: int, protected_phrases: list[str]) -> float:
    left = text[:split_at]
    right = text[split_at:]
    left_len = display_len(sanitize_subtitle_text(left))
    right_len = display_len(sanitize_subtitle_text(right))

    score = float(left_len)
    if left[-1:] in PUNCTUATION:
        score += 1000
    if _starts_with_any(right, SOFT_BREAKS):
        score += 320
    if _ends_with_any(left, SOFT_BREAKS):
        score += 120
    if _starts_with_any(right, protected_phrases):
        score += 380
    if _ends_with_any(left, protected_phrases):
        score += 260
    if right_len <= SHORT_TAIL_CHARS:
        score -= 600
    if left_len <= SHORT_TAIL_CHARS:
        score -= 200
    score -= abs(left_len - right_len) * 0.25
    return score


def _safe_split_index(text: str, split_at: int, max_chars: int, protected_phrases: list[str]) -> int:
    bounds = _protected_phrase_bounds(text, split_at, protected_phrases)
    if not bounds:
        return split_at

    start, end = bounds
    candidates = [start, end]
    valid_candidates = [idx for idx in candidates if _valid_split(text, idx, max_chars, protected_phrases)]
    if valid_candidates:
        return max(valid_candidates, key=lambda idx: _split_score(text, idx, protected_phrases))

    length_valid = [
        idx
        for idx in candidates
        if display_len(sanitize_subtitle_text(text[:idx])) <= max_chars
        and display_len(sanitize_subtitle_text(text[:idx])) > 0
        and display_len(sanitize_subtitle_text(text[idx:])) > 0
    ]
    if length_valid:
        return max(length_valid, key=lambda idx: _split_score(text, idx, protected_phrases))

    return start if start > 0 else end


def _inside_protected_phrase(text: str, split_at: int, protected_phrases: list[str]) -> bool:
    return _protected_phrase_bounds(text, split_at, protected_phrases) is not None


def _protected_phrase_bounds(text: str, split_at: int, protected_phrases: list[str]) -> tuple[int, int] | None:
    for phrase in protected_phrases:
        start = text.find(phrase)
        while start != -1:
            end = start + len(phrase)
            if start < split_at < end:
                return start, end
            start = text.find(phrase, start + 1)
    return None


def _index_for_display_len(text: str, max_chars: int) -> int:
    count = 0
    for idx, char in enumerate(text):
        if not char.isspace():
            count += 1
        if count >= max_chars:
            return idx + 1
    return len(text)


def _piece_timings(segment: CleanedSegment, pieces: list[str]) -> list[tuple[float, float]]:
    fallback_points = _piece_durations(segment.start, segment.end, pieces)
    fallback = list(zip(fallback_points[:-1], fallback_points[1:]))
    timeline = _token_timeline(segment.tokens)
    if not timeline:
        return fallback

    pointer = 0
    timings: list[tuple[float, float]] = []
    for idx, piece in enumerate(pieces):
        chars = _timing_chars(piece)
        matches: list[dict[str, Any]] = []
        for char in chars:
            found_at = _find_next_token(timeline, char, pointer)
            if found_at is None:
                continue
            matches.append(timeline[found_at])
            pointer = found_at + 1

        if matches and len(matches) >= max(1, len(chars) // 2):
            start = float(matches[0]["start"])
            end = float(matches[-1]["end"])
            timings.append((start, max(end, start + 0.05)))
        else:
            timings.append(fallback[idx])

    return _make_timings_monotonic(timings, float(segment.start), float(segment.end))


def _token_timeline(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for token in tokens or []:
        text = sanitize_subtitle_text(str(token.get("text", "")).strip().strip('"'))
        if not text:
            continue
        start = float(token.get("start", 0.0))
        end = float(token.get("end", start))
        for char in _timing_chars(text):
            timeline.append({"text": char, "start": start, "end": end})
    return timeline


def _timing_chars(text: str) -> list[str]:
    return [char for char in re.sub(r"\s+", "", sanitize_subtitle_text(text)) if char]


def _find_next_token(timeline: list[dict[str, Any]], char: str, pointer: int) -> int | None:
    for idx in range(pointer, len(timeline)):
        if timeline[idx]["text"] == char:
            return idx
    return None


def _make_timings_monotonic(timings: list[tuple[float, float]], segment_start: float, segment_end: float) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    previous_end = segment_start
    for start, end in timings:
        start = max(float(start), previous_end)
        end = max(float(end), start + 0.05)
        result.append((start, min(end, max(segment_end, end))))
        previous_end = result[-1][1]
    return result


def _piece_durations(start: float, end: float, pieces: list[str]) -> list[float]:
    start = float(start)
    end = max(float(end), start + 0.1)
    weights = [max(display_len(piece), 1) for piece in pieces]
    total = sum(weights)
    points = [start]
    elapsed = 0.0
    duration = end - start
    for weight in weights[:-1]:
        elapsed += duration * (weight / total)
        points.append(start + elapsed)
    points.append(end)
    return points


def _segment_protected_phrases(segment: CleanedSegment, extra: list[str] | None) -> list[str]:
    candidates = list(PROTECTED_PHRASES)
    candidates.extend(extra or [])
    for uncertain in segment.uncertain_terms:
        candidates.append(str(uncertain.get("raw_asr_text", "")))
        candidates.append(str(uncertain.get("suggested_text", "")))
    return _normalize_protected_phrases(candidates)


def _normalize_protected_phrases(phrases: list[str] | None) -> list[str]:
    normalized = {
        re.sub(r"\s+", "", sanitize_subtitle_text(phrase))
        for phrase in (phrases or PROTECTED_PHRASES)
        if 2 <= len(re.sub(r"\s+", "", sanitize_subtitle_text(phrase))) <= 30
    }
    return sorted(normalized, key=len, reverse=True)


def _starts_with_any(text: str, phrases: tuple[str, ...] | list[str]) -> bool:
    return any(text.startswith(phrase) for phrase in phrases if phrase)


def _ends_with_any(text: str, phrases: tuple[str, ...] | list[str]) -> bool:
    return any(text.endswith(phrase) for phrase in phrases if phrase)
