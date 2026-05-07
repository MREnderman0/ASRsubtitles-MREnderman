from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import json_repair
import pandas as pd

from core.utils.ask_gpt import ask_gpt
from subtitle_rag.parsers import ReferenceDoc
from subtitle_rag.rag import RagContext, glossary_prompt


SYNTHETIC_REFERENCE_NAME = "global_asr_analysis.md"


def analyze_global_transcript(
    words: pd.DataFrame,
    rag_context: RagContext,
    run_dir: Path,
    enabled: bool = True,
) -> tuple[RagContext, dict[str, Any]]:
    md_path = run_dir / "global_asr_analysis.md"
    json_path = run_dir / "global_asr_analysis.json"
    stats: dict[str, Any] = {
        "global_asr_analysis_enabled": bool(enabled),
        "global_asr_analysis_success": False,
        "global_asr_analysis_error": "",
        "global_asr_analysis_md": str(md_path),
        "global_asr_analysis_json": str(json_path),
        "global_asr_analysis_input_chars": 0,
        "global_asr_analysis_output_chars": 0,
        "global_asr_analysis_estimated_input_tokens": 0,
        "global_asr_analysis_estimated_output_tokens": 0,
    }
    if not enabled:
        return rag_context, stats

    transcript = _join_word_text(words)
    stats["global_asr_analysis_input_chars"] = len(transcript)
    if not transcript.strip():
        stats["global_asr_analysis_error"] = "empty transcript"
        _write_fallback_files(md_path, json_path, stats["global_asr_analysis_error"])
        return rag_context, stats

    try:
        response, prompt = _ask_global_analysis_llm(transcript, rag_context)
        md_text = _analysis_to_markdown(response)
        md_path.write_text(md_text, encoding="utf-8")
        json_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
        stats["global_asr_analysis_success"] = True
        stats["global_asr_analysis_output_chars"] = len(md_text)
        stats["global_asr_analysis_estimated_input_tokens"] = _estimate_tokens(prompt)
        stats["global_asr_analysis_estimated_output_tokens"] = _estimate_tokens(json.dumps(response, ensure_ascii=False))
        return _append_analysis_reference(rag_context, md_text), stats
    except Exception as exc:
        stats["global_asr_analysis_error"] = str(exc)
        _write_fallback_files(md_path, json_path, str(exc))
        return rag_context, stats


def _ask_global_analysis_llm(transcript: str, rag_context: RagContext) -> tuple[dict[str, Any], str]:
    references = _all_reference_texts(rag_context)
    prompt = f"""
你是中文 ASR 转录诊断专家。请阅读完整转录文本、自定义词汇表和参考资料，整理一份供后续字幕内容校对使用的诊断资料。

目标：
1. 判断音频主题、人物、机构、专业领域和核心概念。
2. 找出语音转文字中出现的所有明确问题；即使只出现一次，只要能判断为错误，也要列出。重点关注近音误识别、术语误写、人名误写、机构名误写、英文名大小写问题。
3. 优先使用自定义词汇表，其次参考资料，最后再根据全文上下文判断。
4. 只整理资料，不直接改写全文，不输出完整字幕。
5. 对可以明确判断的术语给出推荐写法；无法明确判断的内容标为 uncertain。
6. 输出要服务后续字幕清洗：简洁、可执行、不要泛泛而谈。

自定义词汇表：
{glossary_prompt(rag_context.glossary)}

参考资料：
{chr(10).join(references)}

完整 ASR 转录文本：
{transcript}

只返回 JSON：
{{
  "topic_summary": "一句话概括音频主题",
  "domain": ["领域或场景"],
  "confirmed_terms": [
    {{
      "term": "推荐写法",
      "aliases_or_misrecognitions": ["可能的误识别写法"],
      "basis": "glossary/reference/context"
    }}
  ],
  "suspected_error_patterns": [
    {{
      "pattern": "错误模式",
      "recommended_action": "后续校对应如何处理"
    }}
  ],
  "uncertain_terms": [
    {{
      "raw_asr_text": "存疑写法",
      "candidate": "候选写法",
      "confidence": 0.6
    }}
  ],
  "cleaning_guidance": [
    "给后续字幕校对模型的具体注意事项"
  ]
}}
"""
    response = ask_gpt(prompt, resp_type="json", log_title="subtitle_rag_global_asr_analysis")
    if isinstance(response, str):
        response = json_repair.loads(response)
    if not isinstance(response, dict):
        raise ValueError("global analysis LLM returned non-object JSON")
    return response, prompt


def _analysis_to_markdown(response: dict[str, Any]) -> str:
    lines = ["# Global ASR Analysis", ""]
    topic = str(response.get("topic_summary", "")).strip()
    if topic:
        lines.extend(["## Topic", topic, ""])

    domain = response.get("domain", [])
    if isinstance(domain, list) and domain:
        lines.extend(["## Domain", ", ".join(str(item) for item in domain if str(item).strip()), ""])

    lines.extend(["## Confirmed Terms"])
    for item in response.get("confirmed_terms", []) or []:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term", "")).strip()
        aliases = item.get("aliases_or_misrecognitions", []) or []
        basis = str(item.get("basis", "")).strip()
        if term:
            lines.append(f"- {term}; possible ASR forms: {', '.join(str(alias) for alias in aliases)}; basis: {basis}")
    lines.append("")

    lines.extend(["## Suspected Error Patterns"])
    for item in response.get("suspected_error_patterns", []) or []:
        if not isinstance(item, dict):
            continue
        pattern = str(item.get("pattern", "")).strip()
        action = str(item.get("recommended_action", "")).strip()
        if pattern or action:
            lines.append(f"- {pattern}: {action}")
    lines.append("")

    lines.extend(["## Uncertain Terms"])
    for item in response.get("uncertain_terms", []) or []:
        if not isinstance(item, dict):
            continue
        raw = str(item.get("raw_asr_text", "")).strip()
        candidate = str(item.get("candidate", "")).strip()
        confidence = item.get("confidence", "")
        if raw or candidate:
            lines.append(f"- {raw} => {candidate}; confidence: {confidence}")
    lines.append("")

    lines.extend(["## Cleaning Guidance"])
    for item in response.get("cleaning_guidance", []) or []:
        text = str(item).strip()
        if text:
            lines.append(f"- {text}")
    lines.append("")
    return "\n".join(lines)


def _append_analysis_reference(rag_context: RagContext, md_text: str) -> RagContext:
    references = list(rag_context.references)
    if md_text.strip():
        references.append(ReferenceDoc(name=SYNTHETIC_REFERENCE_NAME, text=md_text.strip()))
    return RagContext(glossary=list(rag_context.glossary), references=references)


def _write_fallback_files(md_path: Path, json_path: Path, error: str) -> None:
    payload = {"error": error}
    md_path.write_text(f"# Global ASR Analysis\n\nAnalysis failed: {error}\n", encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _all_reference_texts(rag_context: RagContext) -> list[str]:
    texts: list[str] = []
    for doc in rag_context.references:
        text = str(doc.text or "").strip()
        if text:
            texts.append(f"[{doc.name}]\n{text}")
    return texts


def _join_word_text(words: pd.DataFrame) -> str:
    text = ""
    for value in words["text"].astype(str).tolist():
        word = value.strip()
        if not word:
            continue
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-']*", word):
            text = f"{text} {word}".strip()
        else:
            text += word
    return re.sub(r"\s+([，。！？；：,.!?;:])", r"\1", text).strip()


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(str(text)) / 3))
