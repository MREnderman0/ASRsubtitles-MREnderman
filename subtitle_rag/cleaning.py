from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import json_repair

from core.utils.ask_gpt import ask_gpt
from core.utils.config_utils import load_key
from subtitle_rag.rag import RagContext, apply_glossary, glossary_prompt, reference_snippets


FILLERS = (
    "呃",
    "呃呃",
    "啊",
    "嗯",
    "嗯嗯",
    "那个",
    "这个",
    "就是",
    "然后呃",
    "对吧",
    "是不是",
)

_LLM_DISABLED_REASON: str | None = None


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    tokens: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CleanedSegment:
    id: int
    start: float
    end: float
    raw_text: str
    cleaned_text: str
    tokens: list[dict[str, Any]] = field(default_factory=list)
    uncertain_terms: list[dict[str, Any]] = field(default_factory=list)


def deterministic_clean(text: str) -> str:
    cleaned = str(text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([，。！？；：,.!?;:])", r"\1", cleaned)
    return cleaned.strip()


def clean_segments(segments: list[Segment], rag_context: RagContext, batch_size: int = 8) -> list[CleanedSegment]:
    global _LLM_DISABLED_REASON
    _LLM_DISABLED_REASON = None

    cleaned: list[CleanedSegment] = []
    for idx in range(0, len(segments), batch_size):
        batch = segments[idx : idx + batch_size]
        cleaned.extend(_clean_batch(batch, rag_context))
    return cleaned


def _clean_batch(segments: list[Segment], rag_context: RagContext) -> list[CleanedSegment]:
    pre_corrected: list[tuple[Segment, str, list[dict[str, Any]], list[str]]] = []
    for seg in segments:
        corrected, hits = apply_glossary(seg.text, rag_context.glossary)
        snippets = reference_snippets(corrected, rag_context.references)
        uncertain = [
            {
                "start_time": seg.start,
                "end_time": seg.end,
                "raw_asr_text": hit.raw,
                "suggested_text": hit.replacement,
                "reason": hit.reason,
                "source": hit.source,
                "confidence": hit.confidence,
            }
            for hit in hits
        ]
        pre_corrected.append((seg, corrected, uncertain, snippets))

    global _LLM_DISABLED_REASON
    try:
        by_id = {} if _LLM_DISABLED_REASON else _ask_llm_batch(pre_corrected, rag_context)
    except Exception as exc:
        _LLM_DISABLED_REASON = str(exc)
        by_id = {}

    results: list[CleanedSegment] = []
    for seg, corrected, glossary_uncertain, _ in pre_corrected:
        item = by_id.get(seg.id)
        if item:
            text = str(item.get("cleaned_text") or corrected).strip()
            uncertain = item.get("uncertain_terms") or []
            if not isinstance(uncertain, list):
                uncertain = []
            normalized_uncertain = []
            for uncertain_item in uncertain:
                normalized = _normalize_uncertain(seg, uncertain_item)
                if normalized:
                    normalized_uncertain.append(normalized)
            results.append(
                CleanedSegment(
                    id=seg.id,
                    start=seg.start,
                    end=seg.end,
                    raw_text=seg.text,
                    cleaned_text=text or deterministic_clean(corrected),
                    tokens=seg.tokens,
                    uncertain_terms=glossary_uncertain + normalized_uncertain,
                )
            )
        else:
            results.append(
                CleanedSegment(
                    id=seg.id,
                    start=seg.start,
                    end=seg.end,
                    raw_text=seg.text,
                    cleaned_text=deterministic_clean(corrected),
                    tokens=seg.tokens,
                    uncertain_terms=glossary_uncertain,
                )
            )
    return results


def _ask_llm_batch(pre_corrected: list[tuple[Segment, str, list[dict[str, Any]], list[str]]], rag_context: RagContext) -> dict[int, dict[str, Any]]:
    if not _llm_configured():
        raise RuntimeError("LLM is not configured.")

    segment_payload = []
    reference_payload: list[str] = []
    for seg, corrected, _, snippets in pre_corrected:
        segment_payload.append(
            {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "raw_text": seg.text,
                "glossary_corrected_text": corrected,
            }
        )
        reference_payload.extend(snippets)

    prompt = f"""
你是一个中文听写字幕清洗助手。请基于自定义词汇表、参考资料片段和上下文清洗 ASR 文本。

必须执行：
1. 删除无意义语气词，例如：呃、啊、嗯、那个、这个等。
2. 删除口吃、重复词和明显口语冗余。
3. 结合上下文修正同音字、近音字和专业术语，但不要使用固定纠错表臆断。
4. 自定义词汇表优先级最高，其次参考资料，最后才使用你的常识判断。
5. 如果可以明确判断正确写法，直接写入 cleaned_text，不要放入 uncertain_terms。
6. 只有无法明确确认的疑似误识别词、专业词或近音词，才写入 uncertain_terms，并标注原因、候选和置信度。

自定义词汇表：
{glossary_prompt(rag_context.glossary)}

参考资料片段：
{chr(10).join(reference_payload[:12])}

待处理片段 JSON：
{json.dumps(segment_payload, ensure_ascii=False, indent=2)}

只返回 JSON，格式：
{{
  "segments": [
    {{
      "id": 1,
      "cleaned_text": "清洗后的文本",
      "uncertain_terms": [
        {{
          "raw_asr_text": "原始模糊词",
          "suggested_text": "候选结果",
          "reason": "为什么仍不确定",
          "source": "llm_knowledge",
          "confidence": 0.65
        }}
      ]
    }}
  ]
}}
"""
    response = ask_gpt(prompt, resp_type="json", log_title="subtitle_rag_cleaning")
    if isinstance(response, str):
        response = json_repair.loads(response)
    items = response.get("segments", []) if isinstance(response, dict) else []
    return {int(item["id"]): item for item in items if "id" in item}


def _llm_configured() -> bool:
    try:
        key = str(load_key("api.key")).strip()
        model = str(load_key("api.model")).strip()
    except Exception:
        return False
    placeholders = {"", "your-api-key", "your_api_key", "YOUR_API_KEY"}
    return key not in placeholders and model not in placeholders


def _normalize_uncertain(seg: Segment, item: dict[str, Any]) -> dict[str, Any] | None:
    raw_text = str(item.get("raw_asr_text", "")).strip()
    suggested_text = str(item.get("suggested_text", "")).strip()
    reason = str(item.get("reason", "")).strip()
    confidence = item.get("confidence", "")

    if not raw_text and not suggested_text:
        return None
    if not reason and _confidence_as_float(confidence) >= 0.85:
        return None

    return {
        "start_time": item.get("start_time", seg.start),
        "end_time": item.get("end_time", seg.end),
        "raw_asr_text": raw_text,
        "suggested_text": suggested_text,
        "reason": reason,
        "source": item.get("source", "llm_knowledge"),
        "confidence": confidence,
    }


def _confidence_as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
