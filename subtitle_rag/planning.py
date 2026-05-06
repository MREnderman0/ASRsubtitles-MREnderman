from __future__ import annotations

import json
import difflib
import html
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json_repair
import pandas as pd

from core.utils.ask_gpt import ask_gpt
from subtitle_rag.cleaning import Segment


BOUNDARY_BLOCK_SECONDS = 600.0
BOUNDARY_OVERLAP_SECONDS = 30.0
BOUNDARY_MAX_ATTEMPTS = 3
BOUNDARY_RETRY_SKIPPED_RATIO = 0.10
STRONG_PUNCTUATION = "，。？！,?!"
ALL_PUNCTUATION = "，。？！；：、,.!?;:《》〈〉“”‘’\"'（）()【】[]{}—…-"
SLASH = "/"
ProgressCallback = Any


@dataclass
class BoundaryStats:
    enabled: bool = True
    block_seconds: int = 600
    overlap_seconds: int = 30
    block_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    fallback_used: bool = False
    segment_count: int = 0
    retry_count: int = 0
    report_path: str = ""
    llm_path: str = ""
    debug_path: str = ""


def plan_segments_from_words(
    words: pd.DataFrame,
    max_chars: int,
    run_dir: Path,
    window_seconds: float | None = None,
    overlap_seconds: float | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[list[Segment], dict[str, Any]]:
    report_path = run_dir / "boundary_plan_report.csv"
    llm_path = run_dir / "boundary_plan_llm.json"
    debug_path = run_dir / "boundary_plan_debug.md"
    window_seconds = float(window_seconds or BOUNDARY_BLOCK_SECONDS)
    overlap_seconds = float(overlap_seconds if overlap_seconds is not None else BOUNDARY_OVERLAP_SECONDS)
    stats = BoundaryStats(
        block_seconds=int(window_seconds),
        overlap_seconds=int(overlap_seconds),
        report_path=str(report_path),
        llm_path=str(llm_path),
        debug_path=str(debug_path),
    )
    debug_path.write_text("# Boundary Planning Debug\n\n", encoding="utf-8")
    tokens = _tokens_from_words(words)
    if not tokens:
        stats.fallback_used = True
        _write_report(report_path, [])
        llm_path.write_text("[]\n", encoding="utf-8")
        return [], _stats_dict(stats)

    blocks = _build_blocks(tokens, window_seconds=window_seconds, overlap_seconds=overlap_seconds)
    stats.block_count = len(blocks)
    payloads: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    planned_candidates: list[dict[str, Any]] = []

    for block_index, block in enumerate(blocks, start=1):
        block_tokens = _tokens_in_range(tokens, block["input_start"], block["input_end"])
        if not block_tokens:
            continue
        raw_text = _join_token_text(block_tokens)
        try:
            planned_text, prompt, attempts, retry_records = _plan_boundary_text(
                raw_text,
                max_chars=max_chars,
                debug_path=debug_path,
                block_index=block_index,
                block=block,
                progress=progress,
            )
            sentence_candidates = _candidate_sentences(block_tokens, planned_text, block)
            planned_candidates.extend(sentence_candidates)
            stats.success_count += 1
            stats.retry_count += max(0, attempts - 1)
            rows.append(_report_row(block_index, block, "success", f"attempts={attempts}; retries={max(0, attempts - 1)}", raw_text, planned_text))
            payloads.append({
                "block": block,
                "raw_text": raw_text,
                "planned_text": planned_text,
                "attempts": attempts,
                "retry_records": retry_records,
                "prompt_chars": len(prompt),
            })
        except Exception as exc:
            stats.failed_count += 1
            rows.append(_report_row(block_index, block, "failed", str(exc), raw_text, ""))

    selected = _select_sentence_candidates(planned_candidates)
    if selected:
        segments = _segments_from_sentence_candidates(selected, max_chars=max_chars)
        stats.segment_count = len(segments)
        stats.fallback_used = stats.failed_count > 0
    else:
        segments = []
        stats.fallback_used = True

    _write_report(report_path, rows)
    llm_path.write_text(json.dumps(payloads, ensure_ascii=False, indent=2), encoding="utf-8")
    return segments, _stats_dict(stats)


def _tokens_from_words(words: pd.DataFrame) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    active_segment_text = ""
    for idx, row in words.iterrows():
        text = str(row["text"]).strip().strip('"').strip()
        if not text:
            continue
        segment_text = _clean_optional_text(row.get("segment_text", ""))
        if segment_text:
            active_segment_text = segment_text
        tokens.append(
            {
                "idx": int(idx),
                "text": text,
                "start": float(row["start"]),
                "end": float(row["end"]),
                "segment_text": active_segment_text,
            }
        )
    _annotate_source_offsets(tokens)
    return tokens


def _clean_optional_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip().strip('"').strip()


def _build_blocks(tokens: list[dict[str, Any]], window_seconds: float, overlap_seconds: float) -> list[dict[str, float]]:
    max_end = max(float(item["end"]) for item in tokens)
    blocks: list[dict[str, float]] = []
    core_start = 0.0
    while core_start < max_end:
        core_end = min(core_start + window_seconds, max_end + 0.001)
        blocks.append(
            {
                "core_start": core_start,
                "core_end": core_end,
                "input_start": max(0.0, core_start - overlap_seconds),
                "input_end": min(max_end + 0.001, core_end + overlap_seconds),
            }
        )
        core_start += window_seconds
    return blocks


def _tokens_in_range(tokens: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [item for item in tokens if start <= _center(item) < end]


def _join_token_text(tokens: list[dict[str, Any]]) -> str:
    punctuated = _punctuated_text_for_tokens(tokens)
    if punctuated:
        return punctuated
    text = ""
    for token in tokens:
        value = str(token["text"])
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-']*", value):
            text = f"{text} {value}".strip()
        else:
            text += value
    return re.sub(r"\s+([，。？！；：,.!?;:])", r"\1", text).strip()


def _annotate_source_offsets(tokens: list[dict[str, Any]]) -> None:
    group_start = 0
    while group_start < len(tokens):
        source = str(tokens[group_start].get("segment_text", "") or "")
        group_end = group_start + 1
        while group_end < len(tokens) and str(tokens[group_end].get("segment_text", "") or "") == source:
            group_end += 1
        if source:
            _annotate_group_offsets(tokens[group_start:group_end], source)
        group_start = group_end


def _annotate_group_offsets(tokens: list[dict[str, Any]], source: str) -> None:
    compact_chars: list[str] = []
    source_positions: list[int] = []
    for pos, char in enumerate(source):
        if char.isspace() or char in ALL_PUNCTUATION or char == SLASH:
            continue
        compact_chars.append(char)
        source_positions.append(pos)
    compact_source = "".join(compact_chars)
    cursor = 0
    for token in tokens:
        token["source_start"] = None
        token["source_end"] = None
        norm = _countable_text(str(token.get("text", "")))
        if not norm:
            continue
        found = compact_source.find(norm, cursor)
        if found < 0:
            found = compact_source.find(norm, max(0, cursor - 20))
        if found < 0:
            continue
        token["source_start"] = source_positions[found]
        token["source_end"] = source_positions[found + len(norm) - 1] + 1
        cursor = found + len(norm)


def _punctuated_text_for_tokens(tokens: list[dict[str, Any]]) -> str:
    if not tokens:
        return ""
    pieces: list[str] = []
    group: list[dict[str, Any]] = []
    current_source = None
    for token in tokens:
        source = str(token.get("segment_text", "") or "")
        if group and source != current_source:
            pieces.append(_punctuated_group_text(group))
            group = []
        current_source = source
        group.append(token)
    if group:
        pieces.append(_punctuated_group_text(group))
    text = "".join(part for part in pieces if part)
    return re.sub(r"\s+([，。？！；：,.!?;:])", r"\1", text).strip()


def _punctuated_group_text(tokens: list[dict[str, Any]]) -> str:
    source = str(tokens[0].get("segment_text", "") or "")
    starts = [item.get("source_start") for item in tokens if item.get("source_start") is not None]
    ends = [item.get("source_end") for item in tokens if item.get("source_end") is not None]
    if not source or not starts or not ends:
        return ""
    start = int(min(starts))
    end = int(max(ends))
    return source[start:end].strip()


def _plan_boundary_text(
    raw_text: str,
    max_chars: int,
    debug_path: Path | None = None,
    block_index: int | None = None,
    block: dict[str, float] | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[str, str, int, list[dict[str, Any]]]:
    feedback = ""
    last_error: Exception | None = None
    last_prompt = ""
    retry_records: list[dict[str, Any]] = []
    for attempt in range(1, BOUNDARY_MAX_ATTEMPTS + 1):
        planned_text, prompt = _ask_boundary_llm(raw_text, max_chars=max_chars, feedback=feedback)
        last_prompt = prompt
        try:
            planned_text, projection = _project_planned_boundaries(raw_text, planned_text)
            total_slashes = max(1, projection["projected_count"] + projection["skipped_count"])
            skipped_ratio = projection["skipped_count"] / total_slashes
            retry_records.append(
                {
                    "attempt": attempt,
                    "projected_count": projection["projected_count"],
                    "skipped_count": projection["skipped_count"],
                    "skipped_ratio": round(skipped_ratio, 4),
                }
            )
            if skipped_ratio > BOUNDARY_RETRY_SKIPPED_RATIO and attempt < BOUNDARY_MAX_ATTEMPTS:
                message = (
                    f"LLM 分词 block {block_index or '?'} 第 {attempt} 次未匹配 / 比例 "
                    f"{skipped_ratio:.1%}，重新分词"
                )
                if progress:
                    progress(message, 0.53)
                feedback = (
                    f"Previous output had too many unusable slash boundaries: "
                    f"{projection['skipped_count']} skipped out of {total_slashes} "
                    f"({skipped_ratio:.1%}). Do not put / next to punctuation. "
                    "Do not rewrite text. Insert / only at stable phrase boundaries that can be aligned back to the input."
                )
                continue
            if debug_path:
                _append_boundary_debug(
                    debug_path,
                    block_index=block_index,
                    attempt=attempt,
                    block=block,
                    raw_text=raw_text,
                    llm_text=projection["llm_text"],
                    highlighted_llm_text=projection["highlighted_llm_text"],
                    applied_text=planned_text,
                    projected_count=projection["projected_count"],
                    skipped_count=projection["skipped_count"],
                    skipped_ratio=skipped_ratio,
                    retry_decision="accept",
                )
            _validate_planned_text(raw_text, planned_text, max_chars=max_chars, enforce_max_units=False)
            return planned_text, prompt, attempt, retry_records
        except Exception as exc:
            last_error = exc
            feedback = (
                f"Previous output failed validation: {exc}. "
                "Insert more / boundaries only; do not change source characters or punctuation."
            )
    raise ValueError(f"boundary planning failed after {BOUNDARY_MAX_ATTEMPTS} attempts: {last_error}; last_prompt_chars={len(last_prompt)}")


def _project_planned_boundaries(raw_text: str, planned_text: str) -> tuple[str, dict[str, Any]]:
    raw_compact, raw_positions = _compact_with_positions(raw_text)
    planned_compact, _ = _compact_with_positions(planned_text, skip_slash=True)
    planned_to_raw = _planned_to_raw_index_map(raw_compact, planned_compact)
    slash_boundaries = _slash_boundaries(planned_text)
    raw_boundaries: set[int] = set()
    skipped_boundaries: set[int] = set()
    for boundary in slash_boundaries:
        raw_boundary = _project_boundary(boundary, planned_to_raw, len(raw_compact))
        if raw_boundary is None or raw_boundary <= 0 or raw_boundary >= len(raw_compact):
            skipped_boundaries.add(boundary)
            continue
        left_char = raw_compact[raw_boundary - 1]
        right_char = raw_compact[raw_boundary]
        if left_char in ALL_PUNCTUATION or right_char in ALL_PUNCTUATION:
            skipped_boundaries.add(boundary)
            continue
        raw_boundaries.add(raw_boundary)

    pieces: list[str] = []
    for compact_index, source_pos in enumerate(raw_positions):
        if compact_index in raw_boundaries:
            pieces.append(SLASH)
        pieces.append(raw_text[source_pos])
    applied_text = "".join(pieces)
    return applied_text, {
        "llm_text": planned_text,
        "highlighted_llm_text": _highlight_skipped_slashes(planned_text, skipped_boundaries),
        "projected_count": len(raw_boundaries),
        "skipped_count": len(skipped_boundaries),
    }


def _compact_with_positions(text: str, skip_slash: bool = False) -> tuple[str, list[int]]:
    chars: list[str] = []
    positions: list[int] = []
    for pos, char in enumerate(str(text or "")):
        if char.isspace():
            continue
        if skip_slash and char == SLASH:
            continue
        chars.append(char)
        positions.append(pos)
    return "".join(chars), positions


def _edit_distance_estimate(left: str, right: str) -> int:
    matcher = difflib.SequenceMatcher(None, left, right)
    edits = 0
    for op, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if op == "equal":
            continue
        edits += max(left_end - left_start, right_end - right_start)
    return edits


def _planned_to_raw_index_map(raw_compact: str, planned_compact: str) -> dict[int, int]:
    mapping: dict[int, int] = {}
    matcher = difflib.SequenceMatcher(None, planned_compact, raw_compact)
    for op, planned_start, planned_end, raw_start, raw_end in matcher.get_opcodes():
        planned_len = planned_end - planned_start
        raw_len = raw_end - raw_start
        if op == "equal":
            for offset in range(planned_len):
                mapping[planned_start + offset] = raw_start + offset
    return mapping


def _slash_boundaries(planned_text: str) -> list[int]:
    boundaries: list[int] = []
    compact_index = 0
    for char in str(planned_text or ""):
        if char.isspace():
            continue
        if char == SLASH:
            boundaries.append(compact_index)
            continue
        compact_index += 1
    return boundaries


def _project_boundary(boundary: int, planned_to_raw: dict[int, int], raw_len: int) -> int | None:
    if boundary in planned_to_raw:
        return planned_to_raw[boundary]
    if boundary - 1 in planned_to_raw:
        return planned_to_raw[boundary - 1] + 1
    for distance in range(1, 8):
        left = boundary - distance
        right = boundary + distance
        if left in planned_to_raw:
            return min(planned_to_raw[left] + 1, raw_len)
        if right in planned_to_raw:
            return planned_to_raw[right]
    return None


def _highlight_skipped_slashes(planned_text: str, skipped_boundaries: set[int]) -> str:
    pieces: list[str] = []
    compact_index = 0
    for char in str(planned_text or ""):
        if char.isspace():
            pieces.append(html.escape(char))
            continue
        if char == SLASH:
            if compact_index in skipped_boundaries:
                pieces.append("<mark>/</mark>")
            else:
                pieces.append("/")
            continue
        pieces.append(html.escape(char))
        compact_index += 1
    return "".join(pieces)


def _append_boundary_debug(
    path: Path,
    block_index: int | None,
    attempt: int,
    block: dict[str, float] | None,
    raw_text: str,
    llm_text: str,
    highlighted_llm_text: str,
    applied_text: str,
    projected_count: int,
    skipped_count: int,
    skipped_ratio: float,
    retry_decision: str,
) -> None:
    block = block or {}
    section = [
        f"## Block {block_index or '?'} Attempt {attempt}",
        "",
        f"- core: {block.get('core_start', '')} - {block.get('core_end', '')}",
        f"- input: {block.get('input_start', '')} - {block.get('input_end', '')}",
        f"- projected slash count: {projected_count}",
        f"- skipped slash count: {skipped_count}",
        f"- skipped ratio: {skipped_ratio:.1%}",
        f"- retry decision: {retry_decision}",
        "",
        "### 原文",
        "",
        "```text",
        raw_text,
        "```",
        "",
        "### LLM 返回（未匹配 / 高亮）",
        "",
        highlighted_llm_text,
        "",
        "",
        "### 应用回本地的结果",
        "",
        "```text",
        applied_text,
        "```",
        "",
    ]
    with path.open("a", encoding="utf-8") as file:
        file.write("\n".join(section))
        file.write("\n")


def _ask_boundary_llm(raw_text: str, max_chars: int, feedback: str = "") -> tuple[str, str]:
    retry_note = f"\nValidation feedback: {feedback}\n" if feedback else ""
    prompt = f"""
你是中文字幕分词和短语边界规划器。请在输入原文中只插入斜杠 / 作为词语或短语边界。

硬性规则：
1. 只能新增 /，不能删除、替换、改写、移动任何原文字或标点。
2. 去掉所有 / 后，必须与输入原文逐字完全一致。
3. / 不能紧贴标点，禁止“词 / ，”“词， / 下文”“词 / 。”这类格式。
4. 保留原标点。不要把标点替换成 /。
5. 在中文词语、专有名词、固定搭配、英文词、数字单位内部不要插入 /。
6. 对长句插入足够多的 /，让本地程序可以在不拆词的情况下按 {max_chars} 字左右切成字幕。
7. 只返回 JSON，不要解释。

输入原文：
{raw_text}

返回格式：
{{
  "text": "只新增 / 后的原文"
}}
"""
    prompt += f"""
Hard validation rule:
- Every unit between two "/" boundaries or strong punctuation marks must have <= {max_chars} non-punctuation, non-space characters.
- If a unit is longer than {max_chars}, add more "/" at natural word or phrase boundaries.
- Do not split inside short words such as 研究所, 大学, 化石, 二氧化碳, or English words.
- For long organization names, split between semantic components, not inside the final word.
Fine-grained planning rule:
- Do not insert "/" only when a subtitle would be too long. Mark useful optional boundaries throughout the sentence.
- Prefer natural phrase units of about 2-8 Chinese characters: subjects, predicates, objects, modifiers, proper nouns, verb-object phrases, prepositional phrases, and parallel phrases.
- Keep complete words and named entities intact. "/" means a safe optional boundary, not a mandatory subtitle break.
- Example input: 我主要从事人类演化研究，具体呢，我是做人类的行为与文化演化。
- Good output: 我主要 / 从事 / 人类演化研究，具体呢，我是 / 做 / 人类的行为 / 与 / 文化演化。
- Example input: 是从百万年里的尘埃里来寻找我们人类演化留下的片段性的故事。
- Good output: 是从 / 百万年里的尘埃里 / 来寻找 / 我们人类演化 / 留下的 / 片段性的故事。
{retry_note}
"""
    response = ask_gpt(prompt, resp_type="json", log_title="subtitle_rag_boundary_plan")
    if isinstance(response, str):
        response = json_repair.loads(response)
    if not isinstance(response, dict):
        raise ValueError("LLM did not return a JSON object")
    planned = str(response.get("text", "")).strip()
    if not planned:
        raise ValueError("LLM returned empty text")
    return planned, prompt


def _validate_planned_text(raw_text: str, planned_text: str, max_chars: int, enforce_max_units: bool = True) -> str:
    compact = planned_text.replace(SLASH, "")
    compact = re.sub(r"\s+", "", compact)
    raw_compact = re.sub(r"\s+", "", raw_text)
    if compact != raw_compact:
        raise ValueError("planned text changed original characters")
    if re.search(rf"{re.escape(SLASH)}\s*[{re.escape(ALL_PUNCTUATION)}]", planned_text):
        raise ValueError("slash before punctuation")
    if re.search(rf"[{re.escape(ALL_PUNCTUATION)}]\s*{re.escape(SLASH)}", planned_text):
        raise ValueError("slash after punctuation")
    if enforce_max_units:
        overlong = _overlong_boundary_units(planned_text, max_chars=max_chars)
        if overlong:
            raise ValueError(f"boundary unit exceeds {max_chars} chars: {overlong[0]}")
    return planned_text


def _overlong_boundary_units(planned_text: str, max_chars: int) -> list[str]:
    units = re.split(rf"{re.escape(SLASH)}|[{re.escape(STRONG_PUNCTUATION)}]", planned_text)
    overlong: list[str] = []
    for unit in units:
        clean = _countable_text(unit)
        if len(clean) > max_chars:
            overlong.append(clean[:40])
    return overlong


def _candidate_sentences(tokens: list[dict[str, Any]], planned_text: str, block: dict[str, float]) -> list[dict[str, Any]]:
    raw_parts = _split_planned_sentences(planned_text)
    candidates: list[dict[str, Any]] = []
    pointer = 0
    for planned_sentence in raw_parts:
        sentence_text = planned_sentence.replace(SLASH, "")
        matched_count = _match_token_count(tokens, pointer, sentence_text)
        if matched_count == 0:
            continue
        selected = tokens[pointer : pointer + matched_count]
        pointer += matched_count
        if not selected:
            continue
        start = float(selected[0]["start"])
        end = float(selected[-1]["end"])
        center = (start + end) / 2
        if block["input_start"] <= center < block["input_end"]:
            candidates.append(
                {
                    "planned_text": planned_sentence,
                    "tokens": [dict(item) for item in selected],
                    "start": start,
                    "end": end,
                    "core_start": block["core_start"],
                    "core_end": block["core_end"],
                    "score": _candidate_score(start, end, block),
                }
            )
    return candidates


def _match_token_count(tokens: list[dict[str, Any]], pointer: int, text: str) -> int:
    compact = _countable_text(text)
    if not compact:
        return 0
    cursor = 0
    count = 0
    for token in tokens[pointer:]:
        token_text = _countable_text(str(token["text"]))
        if not token_text:
            count += 1
            continue
        if compact[cursor : cursor + len(token_text)] != token_text:
            break
        cursor += len(token_text)
        count += 1
        if cursor >= len(compact):
            return count
    return 0


def _split_planned_sentences(planned_text: str) -> list[str]:
    parts: list[str] = []
    buffer = ""
    for char in planned_text:
        buffer += char
        if char in STRONG_PUNCTUATION:
            if buffer.strip():
                parts.append(buffer.strip())
            buffer = ""
    if buffer.strip():
        parts.append(buffer.strip())
    return parts


def _candidate_score(start: float, end: float, block: dict[str, float]) -> float:
    center = (start + end) / 2
    core_start = block["core_start"]
    core_end = block["core_end"]
    core_center = (core_start + core_end) / 2
    distance_from_center = abs(center - core_center)
    in_core = core_start <= center < core_end
    edge_margin = min(abs(center - core_start), abs(core_end - center))
    return (100000 if in_core else 0) + edge_margin - distance_from_center * 0.001


def _select_sentence_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for item in candidates:
        tokens = item["tokens"]
        key = (int(tokens[0]["idx"]), int(tokens[-1]["idx"]))
        current = grouped.get(key)
        if current is None or float(item["score"]) > float(current["score"]):
            grouped[key] = item
    return [grouped[key] for key in sorted(grouped)]


def _segments_from_sentence_candidates(candidates: list[dict[str, Any]], max_chars: int) -> list[Segment]:
    segments: list[Segment] = []
    for candidate in candidates:
        for chunk_tokens in _split_candidate_by_slashes(candidate, max_chars=max_chars):
            text = _join_token_words(chunk_tokens)
            text = text.strip()
            if not text:
                continue
            segments.append(
                Segment(
                    id=len(segments) + 1,
                    start=float(chunk_tokens[0]["start"]),
                    end=float(chunk_tokens[-1]["end"]),
                    text=text,
                    tokens=[dict(item) for item in chunk_tokens],
                )
            )
    return segments


def _split_candidate_by_slashes(candidate: dict[str, Any], max_chars: int) -> list[list[dict[str, Any]]]:
    planned_text = str(candidate["planned_text"])
    tokens = list(candidate["tokens"])
    groups = _slash_groups(planned_text, tokens)
    chunks: list[list[dict[str, Any]]] = []
    current: list[list[dict[str, Any]]] = []
    for group in groups:
        trial = [*current, group]
        if _groups_len(trial) <= max_chars or not current:
            current = trial
            continue
        chunks.append(_flatten_groups(current))
        current = [group]
    if current:
        chunks.append(_flatten_groups(current))
    output: list[list[dict[str, Any]]] = []
    for chunk in chunks:
        if _tokens_len(chunk) <= max_chars:
            output.append(chunk)
        else:
            output.extend(_hard_split_tokens(chunk, max_chars=max_chars))
    return output


def _slash_groups(planned_text: str, tokens: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    pointer = 0
    units = [part for part in re.split(f"({re.escape(SLASH)})", planned_text) if part]
    for unit in units:
        if unit == SLASH:
            if current:
                groups.append(current)
                current = []
            continue
        count = _match_token_count(tokens, pointer, unit)
        if count <= 0:
            continue
        current.extend(tokens[pointer : pointer + count])
        pointer += count
        if any(char in STRONG_PUNCTUATION for char in unit):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return [group for group in groups if group]


def _join_token_words(tokens: list[dict[str, Any]]) -> str:
    text = ""
    for token in tokens:
        value = str(token.get("text", ""))
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-']*", value):
            text = f"{text} {value}".strip()
        else:
            text += value
    return re.sub(r"\s+([，。？！；：,.!?;:])", r"\1", text).strip()


def _groups_len(groups: list[list[dict[str, Any]]]) -> int:
    return _tokens_len(_flatten_groups(groups))


def _tokens_len(tokens: list[dict[str, Any]]) -> int:
    return sum(len(_countable_text(str(item.get("text", "")))) for item in tokens)


def _countable_text(text: str) -> str:
    text = re.sub(r"\s+", "", str(text or ""))
    text = text.replace(SLASH, "")
    return "".join(char for char in text if char not in ALL_PUNCTUATION)


def _flatten_groups(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [item for group in groups for item in group]


def _hard_split_tokens(tokens: list[dict[str, Any]], max_chars: int) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    count = 0
    for token in tokens:
        is_punct = str(token.get("text", "")) in ALL_PUNCTUATION
        if current and count >= max_chars and not is_punct:
            chunks.append(current)
            current = []
            count = 0
        current.append(token)
        if not is_punct:
            count += 1
    if current:
        chunks.append(current)
    return chunks


def _center(item: dict[str, Any]) -> float:
    return (float(item["start"]) + float(item["end"])) / 2


def _report_row(block_index: int, block: dict[str, float], status: str, message: str, raw_text: str, planned_text: str) -> dict[str, Any]:
    return {
        "block_index": block_index,
        "status": status,
        "message": message,
        "core_start": block["core_start"],
        "core_end": block["core_end"],
        "input_start": block["input_start"],
        "input_end": block["input_end"],
        "raw_chars": len(raw_text),
        "planned_chars": len(planned_text),
    }


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = ["block_index", "status", "message", "core_start", "core_end", "input_start", "input_end", "raw_chars", "planned_chars"]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")


def _stats_dict(stats: BoundaryStats) -> dict[str, Any]:
    return {
        "boundary_plan_enabled": stats.enabled,
        "boundary_plan_block_seconds": stats.block_seconds,
        "boundary_plan_overlap_seconds": stats.overlap_seconds,
        "boundary_plan_block_count": stats.block_count,
        "boundary_plan_success_count": stats.success_count,
        "boundary_plan_failed_count": stats.failed_count,
        "boundary_plan_fallback_used": stats.fallback_used,
        "boundary_plan_segment_count": stats.segment_count,
        "boundary_plan_retry_count": stats.retry_count,
        "boundary_plan_report": stats.report_path,
        "boundary_plan_llm": stats.llm_path,
        "boundary_plan_debug": stats.debug_path,
    }
