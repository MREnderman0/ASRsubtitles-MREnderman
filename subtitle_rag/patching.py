from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path
from typing import Any

import json_repair
import pandas as pd

from core.utils.ask_gpt import ask_gpt
from subtitle_rag.cleaning import CleanedSegment
from subtitle_rag.rag import RagContext, glossary_prompt
from subtitle_rag.subtitle import SubtitleItem, display_len, sanitize_subtitle_text, split_text


PATCH_BLOCK_SECONDS = 600.0
PATCH_OVERLAP_SECONDS = 30.0
PATCH_LENGTH_EXTRA_CHARS = 5
PATCH_STAGE = "content_review"
PATCH_WRITE_LOCK = Lock()


def review_and_apply_patches(
    draft_items: list[SubtitleItem],
    segments: list[CleanedSegment],
    rag_context: RagContext,
    max_chars: int,
    run_dir: Path,
    window_seconds: float | None = None,
    overlap_seconds: float | None = None,
    max_workers: int | None = None,
    progress=None,
) -> tuple[list[SubtitleItem], list[dict[str, Any]], dict[str, Any]]:
    patch_limit = int(max_chars) + PATCH_LENGTH_EXTRA_CHARS
    window_seconds = float(window_seconds or PATCH_BLOCK_SECONDS)
    overlap_seconds = float(overlap_seconds if overlap_seconds is not None else PATCH_OVERLAP_SECONDS)
    max_workers = max(1, int(max_workers or 1))
    patches_path = run_dir / "llm_patches.json"
    report_path = run_dir / "patch_report.csv"

    all_patch_payloads: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    uncertain_terms: list[dict[str, Any]] = []
    failed_blocks = 0
    estimated_input_tokens = 0
    estimated_output_tokens = 0

    patched = [_copy_item(item) for item in draft_items]
    blocks = _build_blocks(patched, window_seconds=window_seconds, overlap_seconds=overlap_seconds)

    block_results = _review_patch_blocks(
        patched=patched,
        segments=segments,
        rag_context=rag_context,
        max_chars=max_chars,
        patch_limit=patch_limit,
        blocks=blocks,
        max_workers=max_workers,
        progress=progress,
    )

    for block_result in sorted(block_results, key=lambda item: float(item["block"]["core_start"]), reverse=True):
        block = block_result["block"]
        block_index = int(block_result["block_index"])
        if block_result["status"] != "success":
            failed_blocks += 1
            report_rows.append(
                _report_row(
                    status="llm_failed",
                    op="block",
                    indexes=[],
                    message=str(block_result["error"]),
                    block=block,
                )
            )
            continue

        response = block_result["response"]
        prompt_text = str(block_result["prompt_text"])

        estimated_input_tokens += _estimate_tokens(prompt_text)
        estimated_output_tokens += _estimate_tokens(json.dumps(response, ensure_ascii=False))

        response = _strip_reason_fields(response)
        all_patch_payloads.append({"stage": PATCH_STAGE, "block": block, "prompt_text": prompt_text, "response": response})

        pending_uncertain = [item for item in response.get("uncertain_terms", []) or [] if isinstance(item, dict)]

        for patch in _sorted_patches(response.get("patches", []) or []):
            if not isinstance(patch, dict):
                continue
            if str(patch.get("op", "")).strip() == "mark_uncertain":
                uncertain_terms.append(_normalize_uncertain(patch, block))
                applied, patched, message = _apply_uncertain_suggestion(patched, patch, block, patch_limit)
                report_rows.append(
                    _report_row(
                        status="applied" if applied else "skipped",
                        op="mark_uncertain_replace",
                        indexes=_patch_indexes(patch),
                        message=message,
                        block=block,
                        old_text=patch.get("raw_asr_text", patch.get("old_text", "")),
                        new_text=patch.get("suggested_text", patch.get("new_text", "")),
                    )
                )
                continue

            applied, patched, message = _apply_patch(patched, patch, block, patch_limit)
            report_rows.append(
                _report_row(
                    status="applied" if applied else "skipped",
                    op=str(patch.get("op", "")),
                    indexes=_patch_indexes(patch),
                    message=message,
                    block=block,
                    old_text=patch.get("old_text", ""),
                    new_text=patch.get("new_text", patch.get("new_texts", "")),
                )
            )

        for uncertain in pending_uncertain:
            uncertain_terms.append(_normalize_uncertain(uncertain, block))
            applied, patched, message = _apply_uncertain_suggestion(patched, uncertain, block, patch_limit)
            report_rows.append(
                _report_row(
                    status="applied" if applied else "skipped",
                    op="uncertain_replace",
                    indexes=uncertain.get("indexes", uncertain.get("index", [])),
                    message=message,
                    block=block,
                    old_text=uncertain.get("raw_asr_text", ""),
                    new_text=uncertain.get("suggested_text", ""),
                )
            )

    patched = _renumber(patched)
    with PATCH_WRITE_LOCK:
        patches_path.write_text(json.dumps(all_patch_payloads, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(
            report_rows,
            columns=[
                "stage",
                "status",
                "op",
                "indexes",
                "old_text",
                "new_text",
                "message",
                "core_start",
                "core_end",
                "input_start",
                "input_end",
            ],
        ).to_csv(report_path, index=False, encoding="utf-8-sig")

    stats = {
        "llm_patch_enabled": True,
        "patch_block_seconds": int(window_seconds),
        "patch_overlap_seconds": int(overlap_seconds),
        "patch_length_extra_chars": PATCH_LENGTH_EXTRA_CHARS,
        "patch_stage_count": 1,
        "patch_stages": [PATCH_STAGE],
        "patch_block_count": len(blocks),
        "patch_max_workers": max_workers,
        "patch_applied_count": sum(1 for row in report_rows if row["status"] == "applied"),
        "patch_failed_count": sum(1 for row in report_rows if row["status"] != "applied"),
        "llm_failed_block_count": failed_blocks,
        "patch_estimated_input_tokens": estimated_input_tokens,
        "patch_estimated_output_tokens": estimated_output_tokens,
        "llm_patches": str(patches_path),
        "patch_report": str(report_path),
    }
    return patched, uncertain_terms, stats


def _review_patch_blocks(
    patched: list[SubtitleItem],
    segments: list[CleanedSegment],
    rag_context: RagContext,
    max_chars: int,
    patch_limit: int,
    blocks: list[dict[str, float]],
    max_workers: int,
    progress,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for block_index, block in enumerate(blocks, start=1):
        block_items = _items_in_range(patched, block["input_start"], block["input_end"])
        block_raw_text = _raw_text_for_range(segments, block["input_start"], block["input_end"])
        if block_items and block_raw_text.strip():
            jobs.append(
                {
                    "block_index": block_index,
                    "block": block,
                    "items": block_items,
                    "raw_text": block_raw_text,
                }
            )
    if not jobs:
        return []

    results: list[dict[str, Any]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as executor:
        futures = [
            executor.submit(_review_patch_block, job, rag_context, max_chars, patch_limit)
            for job in jobs
        ]
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            results.append(result)
            if progress:
                block_index = result["block_index"]
                label = f"LLM content review block {block_index}/{len(blocks)}"
                if result["status"] != "success":
                    label = f"FAIL:LLM 内容校对分块 {block_index}/{len(blocks)} 失败：{result['error']}"
                progress(label, 0.88 + 0.08 * (completed / max(len(jobs), 1)))
    return sorted(results, key=lambda item: int(item["block_index"]))


def _review_patch_block(
    job: dict[str, Any],
    rag_context: RagContext,
    max_chars: int,
    patch_limit: int,
) -> dict[str, Any]:
    try:
        response, prompt_text = _ask_patch_llm(
            block=job["block"],
            items=job["items"],
            raw_text=job["raw_text"],
            rag_context=rag_context,
            max_chars=max_chars,
            patch_limit=patch_limit,
        )
        return {**job, "status": "success", "response": response, "prompt_text": prompt_text}
    except Exception as exc:
        return {**job, "status": "failed", "error": str(exc)}


def _ask_patch_llm(
    block: dict[str, float],
    items: list[SubtitleItem],
    raw_text: str,
    rag_context: RagContext,
    max_chars: int,
    patch_limit: int,
) -> tuple[dict[str, Any], str]:
    draft_payload = [
        {
            "index": item.index,
            "start": round(float(item.start), 3),
            "end": round(float(item.end), 3),
            "text": item.text,
        }
        for item in items
    ]
    refs = _all_reference_texts(rag_context)
    prompt = f"""
你是中文原文字幕内容校对器。请审查 draft_srt 中明确需要修改的内容问题，只返回补丁；不要返回无问题字幕，不要返回 reason 或解释。

职责范围：
1. 修正明显 ASR 错字、近音误识别、术语、人名、机构名和专业概念错误。
2. 删除确认不影响语义的语气词、口头填充、口吃和误触发重复；不要删除正常叠词、固定表达或礼貌表达。
3. 自定义词汇表优先级最高，其次参考资料片段，最后才使用上下文常识判断。
4. 能明确修正的直接给 patches；无法明确确认但有候选的放入 uncertain_terms。
5. 不做字幕边界审稿，不要为了修复断词而重排字幕；断词和边界由前置 boundary planning 处理。
6. 不生成时间戳，程序会保留或重新分配时间。
7. 最终字幕不应包含标点。
8. 单条字幕默认不超过 {max_chars} 个非空白字符；patch 后最多允许 {patch_limit} 个非空白字符。
9. replace 和 merge_replace 必须包含 new_text；split_replace 必须包含 new_texts。merge_replace/split_replace 只用于内容修正或内容修正后长度处理，不用于边界审稿。
10. 每个 replace、merge_replace、split_replace 都必须包含 indexes 和 old_text。old_text 必须逐字等于这些 indexes 当前字幕 text 按顺序用一个空格拼接后的内容；无法精确填写 old_text 就不要返回 patch。
11. 不要返回空字符串 replace。删除单个语气词时，优先并入相邻字幕的 merge_replace 或 split_replace。
12. 返回前自检长度：任何 new_text 或 new_texts 单项超过 {patch_limit} 个非空白字符都是错误输出，必须拆短。
13. uncertain_terms 只记录“有不同候选但无法确认”的词；raw_asr_text 和 suggested_text 必须不同。已经能明确修正的内容不要放入 uncertain_terms。
14. 只处理核心区间内的字幕；交叠区只作为上下文。

近音校对原则：
- ASR 错字和近音误识别必须读音相近优先，再考虑语义、主题和专业领域。
- 不要仅因为参考资料或全局诊断中出现某个主题词，就把局部词替换成语义相关但读音不近的词。
- 如果 raw_asr_text 的错误词与某个候选词读音高度接近，应优先选择读音接近的候选；如果候选词只语义相关但读音不近，且没有自定义词汇表或参考资料明确证明原文就是该词，不要直接 replace，应放入 uncertain_terms。
- 示例：“量子中计”应优先考虑“量子中继”，不要联想到“量子计算”或“量子模拟”，因为“中计/中继”读音接近，而“计算/模拟”只是主题语义相关。

允许的 patch：
- replace: 替换一条字幕文本。
- merge_replace: 合并多条相邻字幕并替换文本，仅用于内容修正需要跨条处理。
- split_replace: 把一条或多条相邻字幕重新拆成多条，仅用于内容修正后超长或内容修正需要。
- mark_uncertain: 写入待确认，并尽量提供 suggested_text。

核心区间：{block['core_start']:.3f}-{block['core_end']:.3f}
输入区间：{block['input_start']:.3f}-{block['input_end']:.3f}

自定义词汇表：
{glossary_prompt(rag_context.glossary)}

参考资料片段：
{chr(10).join(refs)}

raw_asr_text:
{raw_text}

draft_srt_items:
{json.dumps(draft_payload, ensure_ascii=False, indent=2)}

只返回 JSON，且不要包含 reason 字段：
{{
  "patches": [
    {{
      "op": "replace",
      "indexes": [15],
      "old_text": "错误字幕文本",
      "new_text": "修正后的字幕文本"
    }}
  ],
  "uncertain_terms": [
    {{
      "indexes": [20],
      "raw_asr_text": "模糊词",
      "suggested_text": "候选词",
      "confidence": 0.62
    }}
  ]
}}
"""
    response = ask_gpt(prompt, resp_type="json", log_title="subtitle_rag_patch_review")
    if isinstance(response, str):
        response = json_repair.loads(response)
    if not isinstance(response, dict):
        return {"patches": [], "uncertain_terms": []}, prompt
    return response, prompt


def _all_reference_texts(rag_context: RagContext) -> list[str]:
    snippets: list[str] = []
    for doc in rag_context.references:
        text = str(doc.text or "").strip()
        if text:
            snippets.append(f"[{doc.name}]\n{text}")
    return snippets


def _strip_reason_fields(response: dict[str, Any]) -> dict[str, Any]:
    response.pop("reason", None)
    for patch in response.get("patches", []) or []:
        if isinstance(patch, dict):
            patch.pop("reason", None)
    for uncertain in response.get("uncertain_terms", []) or []:
        if isinstance(uncertain, dict):
            uncertain.pop("reason", None)
    return response


def _apply_patch(
    items: list[SubtitleItem],
    patch: dict[str, Any],
    block: dict[str, float],
    patch_limit: int,
) -> tuple[bool, list[SubtitleItem], str]:
    op = str(patch.get("op", "")).strip()
    indexes = _patch_indexes(patch)
    if op not in {"replace", "merge_replace", "split_replace", "mark_uncertain"}:
        return False, items, f"unsupported op: {op}"
    if op == "mark_uncertain":
        return True, items, "marked uncertain"
    if not indexes:
        return False, items, "missing indexes"

    selected = [item for item in items if item.index in indexes]
    if len(selected) != len(set(indexes)):
        return False, items, "some indexes not found"
    selected.sort(key=lambda item: item.index)
    if not _is_patch_in_core(selected, block):
        return False, items, "patch outside core range"

    old_text = str(patch.get("old_text", "")).strip()
    current_old_text = " ".join(item.text for item in selected)
    if not old_text or _norm_patch_text(old_text) != _norm_patch_text(current_old_text):
        return False, items, "old_text mismatch"

    if op in {"replace", "merge_replace"}:
        new_text = sanitize_subtitle_text(str(patch.get("new_text", "")))
        if not new_text:
            return False, items, "empty new_text"
        if display_len(new_text) > patch_limit:
            new_texts = _auto_split_patch_text(new_text, selected, patch_limit)
            if len(new_texts) <= 1:
                return False, items, f"new_text exceeds limit {patch_limit}"
            too_long = [text for text in new_texts if display_len(text) > patch_limit]
            if too_long:
                return False, items, f"auto split still exceeds limit {patch_limit}"
            replacements = _split_replacements(selected[0], selected[-1], new_texts)
            return True, _replace_range(items, selected, replacements), "auto split applied"
        replacement = SubtitleItem(index=selected[0].index, start=selected[0].start, end=selected[-1].end, text=new_text)
        return True, _replace_range(items, selected, [replacement]), "applied"

    if op == "split_replace":
        raw_texts = patch.get("new_texts", [])
        if not isinstance(raw_texts, list) or not raw_texts:
            return False, items, "missing new_texts"
        new_texts = [sanitize_subtitle_text(str(text)) for text in raw_texts]
        new_texts = [text for text in new_texts if text]
        if not new_texts:
            return False, items, "empty new_texts"
        too_long = [text for text in new_texts if display_len(text) > patch_limit]
        if too_long:
            return False, items, f"new_texts exceed limit {patch_limit}"
        replacements = _split_replacements(selected[0], selected[-1], new_texts)
        return True, _replace_range(items, selected, replacements), "applied"

    return False, items, "unhandled patch"


def _auto_split_patch_text(text: str, selected: list[SubtitleItem], patch_limit: int) -> list[str]:
    text_len = display_len(text)
    if text_len <= patch_limit:
        return [text]
    target_count = max(2, min(len(selected), (text_len + patch_limit - 1) // patch_limit))
    balanced_limit = min(patch_limit, max(1, (text_len + target_count - 1) // target_count))
    pieces = split_text(text, max_chars=balanced_limit)
    if len(pieces) <= 1 or any(display_len(piece) > patch_limit for piece in pieces):
        pieces = split_text(text, max_chars=patch_limit)
    return pieces


def _apply_uncertain_suggestion(
    items: list[SubtitleItem],
    uncertain: dict[str, Any],
    block: dict[str, float],
    patch_limit: int,
) -> tuple[bool, list[SubtitleItem], str]:
    raw_text = sanitize_subtitle_text(str(uncertain.get("raw_asr_text", uncertain.get("old_text", ""))))
    suggested_text = sanitize_subtitle_text(str(uncertain.get("suggested_text", uncertain.get("new_text", ""))))
    if not raw_text or not suggested_text:
        return False, items, "missing raw_asr_text or suggested_text"

    indexes = _patch_indexes(uncertain)
    candidates = [item for item in items if item.index in indexes] if indexes else [
        item for item in items if block["core_start"] <= _center(item) < block["core_end"]
    ]
    if not candidates:
        return False, items, "no candidate subtitles"

    raw_norm = _norm_patch_text(raw_text)
    updated: list[SubtitleItem] = []
    applied = False
    for item in items:
        if item not in candidates:
            updated.append(item)
            continue
        new_text = _replace_term_text(item.text, raw_text, suggested_text)
        if new_text == item.text and raw_norm not in _norm_patch_text(item.text):
            updated.append(item)
            continue
        if new_text == item.text:
            new_text = suggested_text
        if display_len(new_text) > patch_limit:
            updated.append(item)
            continue
        updated.append(SubtitleItem(index=item.index, start=item.start, end=item.end, text=new_text))
        applied = True

    if not applied:
        if not indexes:
            return _apply_uncertain_merge_suggestion(items, candidates, raw_text, suggested_text, raw_norm, patch_limit)
        return False, items, "suggested_text not found or exceeds length limit"
    return True, _renumber(updated), "applied suggested_text"


def _apply_uncertain_merge_suggestion(
    items: list[SubtitleItem],
    candidates: list[SubtitleItem],
    raw_text: str,
    suggested_text: str,
    raw_norm: str,
    patch_limit: int,
) -> tuple[bool, list[SubtitleItem], str]:
    for size in range(2, min(4, len(candidates)) + 1):
        for start in range(0, len(candidates) - size + 1):
            selected = candidates[start : start + size]
            if any(selected[idx].index + 1 != selected[idx + 1].index for idx in range(len(selected) - 1)):
                continue
            joined = " ".join(item.text for item in selected)
            if raw_norm not in _norm_patch_text(joined):
                continue
            new_text = _replace_term_text(joined, raw_text, suggested_text)
            if display_len(new_text) > patch_limit:
                continue
            replacement = SubtitleItem(index=selected[0].index, start=selected[0].start, end=selected[-1].end, text=new_text)
            return True, _replace_range(items, selected, [replacement]), "merged and applied suggested_text"
    return False, items, "no merge candidate"


def _replace_term_text(text: str, raw_text: str, suggested_text: str) -> str:
    text = sanitize_subtitle_text(text)
    raw_text = sanitize_subtitle_text(raw_text)
    suggested_text = sanitize_subtitle_text(suggested_text)
    if raw_text in text:
        return sanitize_subtitle_text(text.replace(raw_text, suggested_text))
    compact_text = _norm_patch_text(text)
    compact_raw = _norm_patch_text(raw_text)
    if compact_raw and compact_text == compact_raw:
        return suggested_text
    return text


def _sorted_patches(patches: list[Any]) -> list[Any]:
    def sort_key(patch: Any) -> int:
        if not isinstance(patch, dict):
            return -1
        indexes = _patch_indexes(patch)
        return max(indexes) if indexes else -1

    return sorted(patches, key=sort_key, reverse=True)


def _split_replacements(first: SubtitleItem, last: SubtitleItem, texts: list[str]) -> list[SubtitleItem]:
    start = float(first.start)
    end = max(float(last.end), start + 0.1)
    weights = [max(display_len(text), 1) for text in texts]
    total = sum(weights)
    cursor = start
    replacements: list[SubtitleItem] = []
    for idx, text in enumerate(texts):
        next_time = end if idx == len(texts) - 1 else cursor + (end - start) * (weights[idx] / total)
        replacements.append(SubtitleItem(index=first.index + idx, start=cursor, end=max(next_time, cursor + 0.05), text=text))
        cursor = replacements[-1].end
    return replacements


def _replace_range(items: list[SubtitleItem], selected: list[SubtitleItem], replacements: list[SubtitleItem]) -> list[SubtitleItem]:
    selected_ids = {item.index for item in selected}
    output: list[SubtitleItem] = []
    inserted = False
    first_id = selected[0].index
    for item in items:
        if item.index == first_id and not inserted:
            output.extend(replacements)
            inserted = True
        if item.index not in selected_ids:
            output.append(item)
    return _renumber(output)


def _build_blocks(items: list[SubtitleItem], window_seconds: float, overlap_seconds: float) -> list[dict[str, float]]:
    if not items:
        return []
    max_end = max(float(item.end) for item in items)
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


def _items_in_range(items: list[SubtitleItem], start: float, end: float) -> list[SubtitleItem]:
    return [item for item in items if start <= _center(item) < end]


def _raw_text_for_range(segments: list[CleanedSegment], start: float, end: float) -> str:
    texts = []
    for segment in segments:
        center = (float(segment.start) + float(segment.end)) / 2
        if start <= center < end:
            texts.append(segment.raw_text)
    return "\n".join(texts)


def _is_patch_in_core(selected: list[SubtitleItem], block: dict[str, float]) -> bool:
    center = (float(selected[0].start) + float(selected[-1].end)) / 2
    return block["core_start"] <= center < block["core_end"]


def _patch_indexes(patch: dict[str, Any]) -> list[int]:
    value = patch.get("indexes")
    if isinstance(value, list):
        result = []
        for item in value:
            try:
                result.append(int(item))
            except (TypeError, ValueError):
                pass
        return result
    if value is None and "index" in patch:
        value = patch.get("index")
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []


def _normalize_uncertain(item: dict[str, Any], block: dict[str, float]) -> dict[str, Any]:
    return {
        "start_time": block["core_start"],
        "end_time": block["core_end"],
        "raw_asr_text": item.get("raw_asr_text", ""),
        "suggested_text": item.get("suggested_text", ""),
        "reason": "",
        "source": item.get("source", "llm_patch"),
        "confidence": item.get("confidence", ""),
    }


def _report_row(
    status: str,
    op: str,
    indexes: Any,
    message: str,
    block: dict[str, float],
    old_text: Any = "",
    new_text: Any = "",
) -> dict[str, Any]:
    return {
        "stage": PATCH_STAGE,
        "status": status,
        "op": op,
        "indexes": json.dumps(indexes, ensure_ascii=False),
        "old_text": old_text,
        "new_text": json.dumps(new_text, ensure_ascii=False) if isinstance(new_text, list) else new_text,
        "message": message,
        "core_start": block["core_start"],
        "core_end": block["core_end"],
        "input_start": block["input_start"],
        "input_end": block["input_end"],
    }


def _norm_patch_text(text: str) -> str:
    return re.sub(r"\s+", "", sanitize_subtitle_text(text))


def _center(item: SubtitleItem) -> float:
    return (float(item.start) + float(item.end)) / 2


def _copy_item(item: SubtitleItem) -> SubtitleItem:
    return SubtitleItem(index=item.index, start=item.start, end=item.end, text=item.text)


def _renumber(items: list[SubtitleItem]) -> list[SubtitleItem]:
    return [SubtitleItem(index=idx, start=item.start, end=item.end, text=item.text) for idx, item in enumerate(items, start=1)]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 2)
