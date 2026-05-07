from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("RICH_NO_LEGACY_WINDOWS", "1")
os.environ.setdefault("TERM", "xterm-256color")

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from core.utils.config_utils import load_key, update_key
from subtitle_rag.pipeline import EXTENSION_ROOT, SubtitleRagError, process_media


UPLOAD_DIR = EXTENSION_ROOT / "uploads"
RESULT_STATE_KEY = "subtitle_rag_last_result"
SETTINGS_DIALOG_KEY = "subtitle_rag_settings_open"
PROGRESS_STEPS_KEY = "subtitle_rag_progress_steps"
PROGRESS_LABEL_KEY = "subtitle_rag_progress_label"
PROGRESS_VALUE_KEY = "subtitle_rag_progress_value"
RUNNING_STATE_KEY = "subtitle_rag_running"


def main() -> None:
    st.set_page_config(page_title="ASR-MREnderman", page_icon="docs/logo.svg", layout="wide")
    _inject_styles()

    settings = _load_settings()
    allowed_types = _safe_load_key("allowed_video_formats", []) + _safe_load_key("allowed_audio_formats", [])

    _render_header(settings)
    if st.session_state.get(SETTINGS_DIALOG_KEY):
        _settings_dialog(settings)

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown('<div class="section-label">输入</div>', unsafe_allow_html=True)
        st.subheader("上传素材")
        media_file = st.file_uploader("音频或视频文件", type=allowed_types, label_visibility="collapsed")
        glossary_files = st.file_uploader(
            "自定义词汇表",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            help="CSV/XLSX，可多选。专业词会优先用于纠错和保护。",
        )
        reference_files = st.file_uploader(
            "参考资料",
            type=["txt", "md", "srt", "vtt", "csv", "xlsx", "xls", "pdf", "docx"],
            accept_multiple_files=True,
            help="TXT/MD/SRT/VTT/CSV/XLSX/PDF/DOCX，可多选。用于 LLM 内容校对。",
        )

    with right:
        st.markdown('<div class="section-label">运行</div>', unsafe_allow_html=True)
        st.subheader("处理参数")
        st.markdown(
            f"""
            <div class="param-grid">
              <div><span>字幕长度</span><strong>{settings['max_chars']}</strong></div>
              <div><span>窗口</span><strong>{settings['window_seconds']}s</strong></div>
              <div><span>交叠</span><strong>{settings['overlap_seconds']}s</strong></div>
              <div><span>LLM 并发</span><strong>{settings['max_concurrent_llm_tasks']}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("参数从右上角设置写入 config.yaml。")
        start = st.button(
            "开始生成字幕",
            type="primary",
            use_container_width=True,
            disabled=media_file is None or bool(st.session_state.get(RUNNING_STATE_KEY)),
        )
        progress_box = st.container()

    if start:
        _run_job(media_file, glossary_files, reference_files, settings, progress_box)
    elif _has_progress_state():
        _render_progress_panel(progress_box)

    _render_result_downloads(st.session_state.get(RESULT_STATE_KEY))


def _render_header(settings: dict[str, object]) -> None:
    asr_backend = _safe_load_key("asr.backend", "unknown")
    model = _safe_load_key("api.model", "")
    base_url = _safe_load_key("api.base_url", "")
    json_mode = "开启" if settings["llm_support_json"] else "关闭"
    st.markdown(
        f"""
        <div class="hero">
          <div class="brand-block">
            <div class="eyebrow">ASR WORKBENCH</div>
            <h1>ASR-MREnderman</h1>
            <p>本地识别、LLM 分句校对、原文 SRT 生成。</p>
          </div>
          <div class="status-grid">
            <div class="status-card"><span>ASR</span><strong>{asr_backend}</strong></div>
            <div class="status-card"><span>LLM</span><strong>{model}</strong></div>
            <div class="status-card"><span>JSON</span><strong>{json_mode}</strong></div>
            <div class="status-card wide"><span>Base URL</span><strong>{base_url}</strong></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    top_left, top_right = st.columns([1, 0.18])
    with top_right:
        if st.button("设置", key="open_settings", use_container_width=True):
            st.session_state[SETTINGS_DIALOG_KEY] = True
            st.rerun()


@st.dialog("设置")
def _settings_dialog(settings: dict[str, object]) -> None:
    with st.form("subtitle_rag_settings_form"):
        st.markdown('<div class="dialog-hint">保存后写入 config.yaml，下一次处理生效。</div>', unsafe_allow_html=True)
        st.markdown("#### LLM")
        base_url = st.text_input("Base URL", value=str(settings["base_url"]))
        api_key = st.text_input("API Key", value=str(settings["api_key"]), type="password")
        model = st.text_input("模型", value=str(settings["model"]))

        with st.expander("高级设置", expanded=False):
            st.caption("仅当当前 API 兼容 OpenAI JSON response_format 时开启。此前默认是关闭。")
            llm_support_json = st.toggle("强制 JSON 输出", value=bool(settings["llm_support_json"]))

        st.markdown("#### 字幕")
        c1, c2, c3, c4 = st.columns(4)
        max_chars = c1.number_input("最大长度", min_value=1, max_value=80, value=int(settings["max_chars"]), step=1)
        window_seconds = c2.number_input("窗口秒数", min_value=30, max_value=7200, value=int(settings["window_seconds"]), step=30)
        overlap_seconds = c3.number_input("交叠秒数", min_value=0, max_value=600, value=int(settings["overlap_seconds"]), step=5)
        max_concurrent_llm_tasks = c4.number_input(
            "最大并发",
            min_value=1,
            max_value=16,
            value=int(settings["max_concurrent_llm_tasks"]),
            step=1,
            help="用于 LLM 分词规划和 LLM 内容校对。当前模型支持 3 个并发任务时建议填 3。",
        )

        save, cancel = st.columns(2)
        save_clicked = save.form_submit_button("保存", type="primary", use_container_width=True)
        cancel_clicked = cancel.form_submit_button("取消", use_container_width=True)

    if save_clicked:
        updates = {
            "api.base_url": base_url.strip(),
            "api.key": api_key.strip(),
            "api.model": model.strip(),
            "api.llm_support_json": bool(llm_support_json),
            "subtitle_rag.max_chars": int(max_chars),
            "subtitle_rag.window_seconds": int(window_seconds),
            "subtitle_rag.overlap_seconds": int(overlap_seconds),
            "subtitle_rag.max_concurrent_llm_tasks": int(max_concurrent_llm_tasks),
        }
        for key, value in updates.items():
            update_key(key, value)
        st.session_state[SETTINGS_DIALOG_KEY] = False
        st.toast("设置已保存", icon="✓")
        st.rerun()
    if cancel_clicked:
        st.session_state[SETTINGS_DIALOG_KEY] = False
        st.rerun()


def _run_job(media_file, glossary_files, reference_files, settings: dict[str, object], progress_box) -> None:
    if media_file is None:
        st.warning("请先上传音频或视频文件。")
        return

    st.session_state.pop(RESULT_STATE_KEY, None)
    st.session_state[RUNNING_STATE_KEY] = True
    st.session_state[PROGRESS_STEPS_KEY] = []
    st.session_state[PROGRESS_LABEL_KEY] = "准备开始"
    st.session_state[PROGRESS_VALUE_KEY] = 0.0
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    input_path = _save_upload(media_file, UPLOAD_DIR / "media")
    glossary_paths = [_save_upload(file, UPLOAD_DIR / "glossary") for file in glossary_files]
    reference_paths = [_save_upload(file, UPLOAD_DIR / "references") for file in reference_files]

    progress_bar, status, step_list = _render_progress_panel(progress_box)

    def on_progress(label: str, value: float) -> None:
        display_label = _progress_label(label)
        progress_steps = st.session_state.setdefault(PROGRESS_STEPS_KEY, [])
        _update_progress_steps(progress_steps, label, value)
        st.session_state[PROGRESS_STEPS_KEY] = progress_steps
        st.session_state[PROGRESS_LABEL_KEY] = display_label
        st.session_state[PROGRESS_VALUE_KEY] = float(value)
        status.write(display_label)
        step_list.markdown(_progress_steps_markdown(progress_steps), unsafe_allow_html=True)
        progress_bar.progress(min(max(value, 0.0), 1.0))

    try:
        result = process_media(
            input_path=input_path,
            glossary_paths=glossary_paths,
            reference_paths=reference_paths,
            max_chars=int(settings["max_chars"]),
            window_seconds=float(settings["window_seconds"]),
            overlap_seconds=float(settings["overlap_seconds"]),
            max_concurrent_llm_tasks=int(settings["max_concurrent_llm_tasks"]),
            progress=on_progress,
        )
    except SubtitleRagError as exc:
        st.session_state[RUNNING_STATE_KEY] = False
        _mark_current_step_failed(str(exc))
        st.error(str(exc))
        return
    except Exception as exc:
        st.session_state[RUNNING_STATE_KEY] = False
        _mark_current_step_failed(f"处理失败：{exc}")
        st.error(f"处理失败：{exc}")
        raise

    st.session_state[RESULT_STATE_KEY] = result
    st.session_state[RUNNING_STATE_KEY] = False


def _has_progress_state() -> bool:
    return bool(
        st.session_state.get(RUNNING_STATE_KEY)
        or st.session_state.get(PROGRESS_STEPS_KEY)
        or st.session_state.get(PROGRESS_LABEL_KEY)
    )


def _render_progress_panel(container):
    with container:
        st.markdown('<div class="section-label">进度</div>', unsafe_allow_html=True)
        st.subheader("处理状态")
        value = float(st.session_state.get(PROGRESS_VALUE_KEY, 0.0) or 0.0)
        label = str(st.session_state.get(PROGRESS_LABEL_KEY, "等待开始") or "等待开始")
        steps = st.session_state.get(PROGRESS_STEPS_KEY, [])
        progress_bar = st.progress(min(max(value, 0.0), 1.0))
        status = st.empty()
        step_list = st.empty()
        status.write(label)
        if steps:
            step_list.markdown(_progress_steps_markdown(steps), unsafe_allow_html=True)
        elif st.session_state.get(RUNNING_STATE_KEY):
            step_list.markdown('<div class="progress-scroll"><div class="step-row"><span>...</span><p>任务正在运行</p></div></div>', unsafe_allow_html=True)
        return progress_bar, status, step_list


def _load_settings() -> dict[str, object]:
    return {
        "base_url": _safe_load_key("api.base_url", ""),
        "api_key": _safe_load_key("api.key", ""),
        "model": _safe_load_key("api.model", ""),
        "llm_support_json": bool(_safe_load_key("api.llm_support_json", False)),
        "max_chars": int(_safe_load_key("subtitle_rag.max_chars", 17)),
        "window_seconds": int(_safe_load_key("subtitle_rag.window_seconds", 600)),
        "overlap_seconds": int(_safe_load_key("subtitle_rag.overlap_seconds", 30)),
        "max_concurrent_llm_tasks": int(_safe_load_key("subtitle_rag.max_concurrent_llm_tasks", 3)),
    }


def _safe_load_key(key: str, fallback):
    try:
        return load_key(key)
    except Exception:
        return fallback


def _save_upload(uploaded_file, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / uploaded_file.name.replace(" ", "_")
    path.write_bytes(uploaded_file.getbuffer())
    return path


def _progress_label(label: str) -> str:
    if label.startswith("FAIL:"):
        return label.removeprefix("FAIL:").strip()
    mapping = {
        "Preparing input": "准备输入文件",
        "Running local ASR word-level transcription": "本地 ASR 词级时间戳识别",
        "Running WhisperX word-level transcription": "本地 ASR 词级时间戳识别",
        "Loading word-level timestamps": "读取词级时间戳",
        "Loading glossary and reference materials": "读取词汇表和参考资料",
        "Analyzing full ASR transcript with LLM": "LLM 全局转录诊断",
        "LLM 全局转录诊断完成": "LLM 全局转录诊断完成",
        "Planning subtitle boundaries with LLM": "LLM 规划字幕边界",
        "Generating draft transcript": "生成初稿文本",
        "Generating draft SRT subtitles": "生成 draft.srt",
        "Reviewing draft with LLM patches": "LLM 校对字幕内容",
        "Done": "处理完成",
    }
    if label.startswith("LLM content review block"):
        return label.replace("LLM content review block", "LLM 内容校对分块")
    if label.startswith("LLM patch review block"):
        return label.replace("LLM patch review block", "LLM 校对分块")
    if label.startswith("LLM 分词 block"):
        return label
    return mapping.get(label, label)


def _update_progress_steps(steps: list[dict[str, object]], label: str, value: float) -> None:
    failed = label.startswith("FAIL:")
    if failed:
        label = _progress_label(label)
        for step in reversed(steps):
            if step.get("label") == label or str(label).startswith(str(step.get("label", ""))):
                step["done"] = False
                step["failed"] = True
                step["message"] = label
                return
        steps.append({"label": label, "done": False, "failed": True, "message": label})
        return

    label = _progress_label(label)
    if not steps:
        steps.append({"label": label, "done": value >= 1.0, "failed": False})
    elif steps[-1]["label"] != label:
        if not steps[-1].get("failed"):
            steps[-1]["done"] = True
        steps.append({"label": label, "done": value >= 1.0, "failed": False})
    elif value >= 1.0:
        if not steps[-1].get("failed"):
            steps[-1]["done"] = True

    if value >= 1.0:
        for step in steps:
            if not step.get("failed"):
                step["done"] = True


def _progress_steps_markdown(steps: list[dict[str, object]]) -> str:
    rows = ['<div class="progress-scroll">']
    for step in steps:
        if step.get("failed"):
            icon = "x"
            state_class = "failed"
            label = step.get("message") or step.get("label")
        else:
            icon = "ok" if step.get("done") else "..."
            state_class = "done" if step.get("done") else "running"
            label = step.get("label")
        rows.append(f'<div class="step-row {state_class}"><span>{icon}</span><p>{label}</p></div>')
    rows.append("</div>")
    return "\n".join(rows)


def _mark_current_step_failed(message: str) -> None:
    steps = st.session_state.setdefault(PROGRESS_STEPS_KEY, [])
    label = str(st.session_state.get(PROGRESS_LABEL_KEY, "处理失败") or "处理失败")
    if steps:
        steps[-1]["done"] = False
        steps[-1]["failed"] = True
        steps[-1]["message"] = message or label
    else:
        steps.append({"label": label, "done": False, "failed": True, "message": message or label})
    st.session_state[PROGRESS_STEPS_KEY] = steps
    st.session_state[PROGRESS_LABEL_KEY] = message or label


def _render_result_downloads(result: dict | None) -> None:
    if not result:
        return

    required_paths = [result.get("final_srt"), result.get("uncertain_terms"), result.get("zip")]
    if any(not path or not Path(path).exists() for path in required_paths):
        st.warning("上一轮结果文件已经不存在，请重新处理。")
        st.session_state.pop(RESULT_STATE_KEY, None)
        return

    st.markdown('<div class="section-label">输出</div>', unsafe_allow_html=True)
    st.subheader("处理结果")
    st.success("字幕生成完成")

    downloads = [
        ("final.srt", result["final_srt"], "text/plain", "download_final_srt"),
        ("uncertain_terms.csv", result["uncertain_terms"], "text/csv", "download_uncertain_terms"),
        ("完整结果 zip", result["zip"], "application/zip", "download_result_zip"),
    ]
    if result.get("draft_srt") and Path(result["draft_srt"]).exists():
        downloads.append(("draft.srt", result["draft_srt"], "text/plain", "download_draft_srt"))
    patch_report = Path(result["run_dir"]) / "patch_report.csv"
    if patch_report.exists():
        downloads.append(("patch_report.csv", str(patch_report), "text/csv", "download_patch_report"))
    global_analysis_md = Path(result["run_dir"]) / "global_asr_analysis.md"
    if global_analysis_md.exists():
        downloads.append(("global_asr_analysis.md", str(global_analysis_md), "text/markdown", "download_global_asr_analysis_md"))
    global_analysis_json = Path(result["run_dir"]) / "global_asr_analysis.json"
    if global_analysis_json.exists():
        downloads.append(("global_asr_analysis.json", str(global_analysis_json), "application/json", "download_global_asr_analysis_json"))

    cols = st.columns(3)
    for idx, (label, path, mime, key) in enumerate(downloads):
        with cols[idx % 3]:
            _download_button(label, path, mime, key)

    st.caption("Run directory")
    st.code(result["run_dir"], language="text")


def _download_button(label: str, path: str, mime: str, key: str) -> None:
    file_path = Path(path)
    st.download_button(
        label=label,
        data=file_path.read_bytes(),
        file_name=file_path.name,
        mime=mime,
        key=key,
        use_container_width=True,
    )


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg: #070b12;
          --panel: #0f1520;
          --panel-2: #121a27;
          --ink: #eef4ff;
          --muted: #91a0b8;
          --line: #223049;
          --accent: #34d399;
          --accent-2: #60a5fa;
          --warn: #fbbf24;
        }
        .stApp {
          background:
            radial-gradient(circle at 20% 0%, rgba(52, 211, 153, .10), transparent 28%),
            linear-gradient(180deg, #070b12 0%, #0a1019 46%, #070b12 100%);
          color: var(--ink);
        }
        .block-container { padding-top: 2rem; max-width: 1180px; }
        #MainMenu, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
          display: none !important;
          visibility: hidden !important;
        }
        .stMarkdown a[href^="#"] {
          display: none !important;
          visibility: hidden !important;
        }
        h1, h2, h3, p, label, span { letter-spacing: 0; }
        .hero {
          display: grid;
          grid-template-columns: minmax(0, 1.15fr) minmax(360px, .85fr);
          gap: 16px;
          align-items: stretch;
          margin-bottom: 12px;
        }
        .brand-block {
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 22px 24px;
          background: linear-gradient(135deg, rgba(18, 26, 39, .95), rgba(12, 18, 28, .96));
          box-shadow: 0 20px 45px rgba(0,0,0,.32);
        }
        .brand-block h1 { margin: 0; font-size: 2.35rem; color: var(--ink); }
        .brand-block p { margin: 8px 0 0 0; color: var(--muted); }
        .eyebrow, .section-label {
          color: var(--accent);
          font-size: .75rem;
          font-weight: 800;
          letter-spacing: .1em;
          text-transform: uppercase;
          margin-bottom: 8px;
        }
        .status-grid {
          display: grid;
          grid-template-columns: 1fr 1fr 1fr;
          gap: 10px;
        }
        .status-card {
          background: rgba(15, 21, 32, .94);
          border: 1px solid var(--line);
          border-radius: 8px;
          box-shadow: 0 20px 45px rgba(0,0,0,.28);
        }
        .status-card { padding: 14px 16px; min-width: 0; }
        .status-card.wide { grid-column: span 3; }
        .status-card span {
          display: block;
          color: var(--muted);
          font-size: .72rem;
          margin-bottom: 5px;
        }
        .status-card strong {
          display: block;
          color: var(--ink);
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          font-size: .94rem;
        }
        [data-testid="column"] > div {
          background: rgba(15, 21, 32, .72);
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 18px;
          box-shadow: 0 20px 45px rgba(0,0,0,.18);
        }
        [data-testid="stHorizontalBlock"] [data-testid="column"] [data-testid="column"] > div {
          background: var(--panel-2);
          padding: 0;
          box-shadow: none;
        }
        [data-testid="column"] h3 { margin-top: 0; color: var(--ink); }
        .param-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 10px;
          margin: 8px 0 12px;
        }
        .param-grid div {
          border: 1px solid var(--line);
          background: var(--panel-2);
          border-radius: 8px;
          padding: 12px;
        }
        .param-grid span {
          display: block;
          color: var(--muted);
          font-size: .76rem;
        }
        .param-grid strong {
          display: block;
          color: var(--ink);
          font-size: 1.25rem;
          margin-top: 3px;
        }
        .dialog-hint {
          color: var(--muted);
          font-size: .86rem;
          margin-bottom: 12px;
        }
        .stButton > button {
          border-radius: 8px;
          background: #101827;
          color: var(--ink);
          border-color: var(--line);
        }
        .stButton > button[kind="primary"] {
          background: linear-gradient(90deg, #10b981, #0ea5e9);
          color: #041018;
          border: 0;
          font-weight: 800;
        }
        .stDownloadButton button {
          border-radius: 8px;
          background: #101827;
          color: var(--ink);
          border-color: var(--line);
        }
        [data-testid="stFileUploader"] {
          border: 1px dashed #334764;
          border-radius: 8px;
          padding: 8px;
          background: rgba(18, 26, 39, .75);
        }
        [data-testid="stFileUploader"] section {
          background: transparent;
          border: 0;
        }
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input {
          background: #0b111b;
          color: var(--ink);
          border-color: var(--line);
          border-radius: 8px;
        }
        [data-testid="stToggle"] {
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 10px 12px;
          background: #0b111b;
        }
        .stProgress > div > div > div > div {
          background: linear-gradient(90deg, #10b981, #60a5fa);
        }
        .progress-card {
          display: none;
        }
        .progress-scroll {
          max-height: 260px;
          overflow-y: auto;
          padding: 10px 14px;
          margin-top: 14px;
          border: 1px solid var(--line);
          border-radius: 8px;
          background: rgba(15, 21, 32, .88);
        }
        .step-row {
          display: grid;
          grid-template-columns: 32px minmax(0, 1fr);
          gap: 8px;
          align-items: start;
          border-bottom: 1px solid rgba(34, 48, 73, .72);
          padding: 8px 0;
        }
        .step-row:last-child { border-bottom: 0; }
        .step-row span {
          display: inline-grid;
          place-items: center;
          width: 24px;
          height: 24px;
          border-radius: 999px;
          background: #172033;
          color: var(--muted);
          font-size: .72rem;
          font-weight: 800;
        }
        .step-row.done span {
          background: rgba(16, 185, 129, .16);
          color: var(--accent);
        }
        .step-row.failed span {
          background: rgba(248, 113, 113, .16);
          color: #f87171;
        }
        .step-row.running span {
          background: rgba(96, 165, 250, .15);
          color: var(--accent-2);
        }
        .step-row p {
          margin: 2px 0 0;
          color: var(--ink);
          line-height: 1.35;
          word-break: break-word;
        }
        code {
          color: #dbeafe !important;
          background: #0b111b !important;
        }
        @media (max-width: 900px) {
          .hero { grid-template-columns: 1fr; }
          .status-grid { grid-template-columns: 1fr; }
          .status-card.wide { grid-column: span 1; }
          .param-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
