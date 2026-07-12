from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import cv2
import fitz
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from config import (
    DEFAULT_LANG,
    ENABLE_RUSSIAN_CORRECTION,
    SHOW_DEBUG_PANEL,
    SHOW_GPU_STATUS,
    SHOW_PAGE_PROGRESS,
    SHOW_PROCESSING_TIMER,
    SHOW_ETA,
)
from preprocessing import crop_document, deskew, enhance_document, upscale_document
from ocr.ensemble import ocr_ensemble
from language import correct_text
from export import export_to_docx, export_to_pdf
from export.layout_html_export import export_layout_html
from layout.page_layout import (
    extract_page_layout,
    extract_layout_from_image,
    extract_tables_from_page,
    extract_tables_from_image,
    render_page_bgr,
)
from utils.confidence import format_review_report, is_low_confidence
from utils.gpu import gpu_available, gpu_name, gpu_memory_total_gb, gpu_memory_allocated_gb
from utils.device_status import get_actual_device

try:
    from paddleocr import PPStructure
except Exception:
    PPStructure = None


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines: List[str] = []
    for line in text.split("\n"):
        lines.append(re.sub(r"[ \t]+", " ", line).strip())
    out: List[str] = []
    empty = 0
    for line in lines:
        if line:
            empty = 0
            out.append(line)
        else:
            empty += 1
            if empty <= 1:
                out.append("")
    return "\n".join(out).strip()


def format_hms(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}h : {minutes:02d}m : {secs:02d}s"


def image_bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Could not decode the image.")
    return img


def bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    if len(image_bgr.shape) == 2:
        return Image.fromarray(image_bgr)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""

    def esc(v: Any) -> str:
        if pd.isna(v):
            return ""
        s = str(v)
        return s.replace("\n", " ").replace("|", "\\|").strip()

    headers = [esc(c) for c in df.columns]
    rows = [[esc(v) for v in row] for row in df.fillna("").values.tolist()]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, sep_line] + data_lines)


def maybe_correct_text(text: str, apply_correction: bool) -> str:
    text = normalize_text(text)
    if apply_correction and ENABLE_RUSSIAN_CORRECTION:
        text = correct_text(text)
        text = normalize_text(text)
    return text.strip()


def build_page_block(
    file_name: str,
    page_number: int,
    engine: str,
    score: float,
    text: str,
    tables: List[Dict[str, Any]],
) -> str:
    parts: List[str] = []
    parts.append(f"===== {file_name} | Page {page_number} =====")
    parts.append(f"[Engine: {engine} | Score: {score:.2f}]")

    cleaned_text = text.strip()
    if cleaned_text:
        parts.append("")
        parts.append(cleaned_text)

    if tables:
        parts.append("")
        parts.append("[TABLES]")
        for tbl in tables:
            parts.append(f"--- Table {tbl.get('table_index', '')} ---")
            md = tbl.get("markdown", "")
            if md:
                parts.append(md)
            else:
                html = tbl.get("html", "")
                if html:
                    parts.append(html)

    return "\n".join(parts).strip()


def process_image_page(
    image_bytes: bytes,
    file_name: str,
    page_number: int,
    lang: str,
    extract_tables: bool,
    task_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    original = image_bytes_to_bgr(image_bytes)
    processed = preprocess_image(original)
    ocr_result = ocr_ensemble(processed, lang=lang)

    raw_text = str(ocr_result.get("text", ""))
    score = float(ocr_result.get("score", 0.0))
    engine = str(ocr_result.get("engine", "None"))
    layout = extract_layout_from_image(original)

    tables = []
    if extract_tables:
        tables = extract_tables_from_image(original)

    if task_state is not None:
        task_state["current_page"] = page_number
        task_state["page_total"] = 1
        task_state["status"] = f"Processing image: {file_name}"

    return {
        "file_name": file_name,
        "page_number": page_number,
        "engine": engine,
        "score": score,
        "raw_text": raw_text,
        "processed_image": processed,
        "tables": tables,
        "layout": layout,
        "report": format_review_report(raw_text, score, engine),
    }


def process_pdf_file(
    pdf_bytes: bytes,
    file_name: str,
    lang: str,
    extract_tables: bool,
    page_mode: str,
    start_page: int,
    end_page: int,
    prefer_text_layer: bool,
    zoom: float = 5.0,
    task_state: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    results: List[Dict[str, Any]] = []

    try:
        total_pages = len(doc)
        if page_mode == "Page range":
            first_page = max(1, start_page)
            last_page = min(end_page, total_pages)
            if first_page > last_page:
                first_page, last_page = last_page, first_page
        else:
            first_page = 1
            last_page = total_pages

        for page_number in range(first_page, last_page + 1):
            if task_state is not None:
                task_state["current_page"] = page_number
                task_state["page_total"] = last_page
                task_state["status"] = f"Processing PDF page {page_number}/{last_page} : {file_name}"

            page = doc.load_page(page_number - 1)
            page_bgr = render_page_bgr(page, zoom=zoom)
            processed = preprocess_image(page_bgr)
            layout = extract_page_layout(page, zoom=zoom)

            page_text = ""
            if prefer_text_layer:
                try:
                    page_text = (page.get_text("text") or "").strip()
                except Exception:
                    page_text = ""

            if page_text:
                raw_text = page_text
                score = 1.0
                engine = "PDF text layer"
            else:
                ocr_result = ocr_ensemble(processed, lang=lang)
                raw_text = str(ocr_result.get("text", ""))
                score = float(ocr_result.get("score", 0.0))
                engine = str(ocr_result.get("engine", "None"))
                print(ocr_result)

            tables = []
            if extract_tables:
                tables = extract_tables_from_page(page, page_bgr, zoom=zoom)

            results.append(
                {
                    "file_name": file_name,
                    "page_number": page_number,
                    "engine": engine,
                    "score": score,
                    "raw_text": raw_text,
                    "processed_image": processed,
                    "tables": tables,
                    "layout": layout,
                    "report": format_review_report(raw_text, score, engine),
                }
            )
    finally:
        doc.close()

    return results


def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None:
        raise ValueError("Empty image.")
    img = crop_document(image_bgr)
    img = deskew(img)
    img = upscale_document(img, factor=4)
    enhanced = enhance_document(img)
    if len(enhanced.shape) == 2:
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced


def ocr_worker(files_payload: List[Dict[str, Any]], params: Dict[str, Any], task_state: Dict[str, Any]) -> None:
    try:
        all_blocks: List[str] = []
        failed_files: List[Tuple[str, str]] = []
        pages_out: List[Dict[str, Any]] = []
        layout_pages: List[Dict[str, Any]] = []

        total_tasks = len(files_payload)
        task_state["total_files"] = total_tasks
        task_state["progress"] = 0.0
        task_state["status"] = "Starting OCR..."
        task_state["done"] = False
        task_state["error"] = None
        task_state["result"] = None

        for idx, item in enumerate(files_payload, start=1):
            name = item["name"]
            data = item["bytes"]
            ext = Path(name).suffix.lower()

            task_state["file_index"] = idx
            task_state["current_file"] = name
            task_state["status"] = f"Processing file {idx}/{total_tasks}: {name}"

            if ext == ".pdf":
                page_results = process_pdf_file(
                    pdf_bytes=data,
                    file_name=name,
                    lang=params["lang"],
                    extract_tables=params["extract_tables"],
                    page_mode=params["page_mode"],
                    start_page=params["start_page"],
                    end_page=params["end_page"],
                    prefer_text_layer=params["prefer_text_layer"],
                    zoom=5.0,
                    task_state=task_state,
                )
            else:
                page_results = [
                    process_image_page(
                        image_bytes=data,
                        file_name=name,
                        page_number=1,
                        lang=params["lang"],
                        extract_tables=params["extract_tables"],
                        task_state=task_state,
                    )
                ]

            for page in page_results:
                raw_text = page["raw_text"]

                if params["preserve_exact"]:
                    final_text = raw_text.strip()
                    if params["normalize_ws"]:
                        final_text = normalize_text(final_text)
                else:
                    final_text = maybe_correct_text(raw_text, params["apply_language_correction"])
                    if not params["normalize_ws"]:
                        final_text = final_text.strip()

                page["final_text"] = final_text
                block = build_page_block(
                    file_name=page["file_name"],
                    page_number=page["page_number"],
                    engine=page["engine"],
                    score=page["score"],
                    text=final_text,
                    tables=page["tables"],
                )
                all_blocks.append(block)
                pages_out.append(page)
                layout_pages.append(page["layout"])

            task_state["progress"] = idx / total_tasks
            task_state["status"] = f"Completed file {idx}/{total_tasks}: {name}"

        final_report = "\n\n".join(all_blocks).strip()
        task_state["result"] = {
            "final_report": final_report,
            "pages_out": pages_out,
            "layout_pages": layout_pages,
            "failed_files": failed_files,
        }
        task_state["done"] = True

    except Exception as exc:
        task_state["error"] = str(exc)
        task_state["done"] = True


st.set_page_config(page_title="Hard Scan OCR Extractor", layout="wide")
st.title("Hard Scan OCR Extractor")
st.caption("GPU-enabled OCR with page range support, layout export, and low-memory live timer.")

with st.sidebar:
    st.header("Settings")
    lang = st.selectbox("OCR language", ["rus+eng", "rus", "eng"], index=0)
    page_mode = st.radio("Pages to process", ["All pages", "Page range"], index=0)
    start_page = st.number_input("Start page", min_value=1, value=1, step=1)
    end_page = st.number_input("End page", min_value=1, value=1, step=1)
    extract_tables = st.checkbox("Extract tables", value=True)
    prefer_text_layer = st.checkbox("Prefer PDF text layer when available", value=True)
    preserve_exact = st.checkbox("Preserve exact OCR text", value=True)
    apply_language_correction = st.checkbox("Apply text correction", value=False)
    normalize_ws = st.checkbox("Normalize whitespace", value=True)
    generate_layout_html = st.checkbox("Generate layout-preserving HTML", value=True)
    show_preview = st.checkbox("Show processed preview", value=True)
    show_diagnostics = st.checkbox("Show diagnostics", value=True)

    st.markdown("---")
    st.subheader("GPU Status")
    if SHOW_GPU_STATUS:
        if gpu_available():
            st.success("CUDA Enabled")
            st.write(f"GPU: {gpu_name()}")
            st.write(f"VRAM: {gpu_memory_total_gb()} GB")
            st.write(f"Allocated: {gpu_memory_allocated_gb()} GB")
        else:
            st.warning("Running on CPU")

    st.markdown("---")
    st.write("For exact transcription, keep correction off.")
    st.write("Timer updates while OCR is running.")

uploaded_files = st.file_uploader(
    "Upload one or more images or PDFs",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp", "pdf"],
    accept_multiple_files=True,
)

run = st.button("Run OCR", type="primary", use_container_width=True)

if run:
    if not uploaded_files:
        st.warning("Please upload at least one file.")
        st.stop()

    if page_mode == "Page range" and start_page > end_page:
        st.warning("Start page is greater than end page. The values will be swapped automatically.")

    files_payload = [{"name": f.name, "bytes": f.getvalue()} for f in uploaded_files]
    params = {
        "lang": lang,
        "page_mode": page_mode,
        "start_page": int(start_page),
        "end_page": int(end_page),
        "extract_tables": extract_tables,
        "prefer_text_layer": prefer_text_layer,
        "preserve_exact": preserve_exact,
        "apply_language_correction": apply_language_correction,
        "normalize_ws": normalize_ws,
    }

    task_state: Dict[str, Any] = {
        "done": False,
        "error": None,
        "result": None,
        "progress": 0.0,
        "status": "Starting OCR...",
        "current_file": "",
        "current_page": 0,
        "page_total": 0,
        "total_files": len(files_payload),
        "file_index": 0,
        "start_time": time.perf_counter(),
    }

    worker = threading.Thread(
        target=ocr_worker,
        args=(files_payload, params, task_state),
        daemon=True,
    )
    worker.start()

    status_placeholder = st.empty()
    timer_col, device_col = st.columns(2)
    timer_placeholder = timer_col.empty()
    device_placeholder = device_col.empty()
    progress_placeholder = st.progress(0)
    details_placeholder = st.empty()

    while not task_state["done"]:

        elapsed = time.perf_counter() - task_state["start_time"]

        status_placeholder.info(
            task_state.get("status", "OCR processing...")
        )

        if SHOW_PROCESSING_TIMER:
            timer_placeholder.metric(
                "⏱ Processing Time",
                format_hms(elapsed)
            )

        # =====================================
        # DEVICE STATUS
        # =====================================

        device_info = get_actual_device()

        current_engine = task_state.get(
            "engine",
            "Unknown"
        )

        if device_info["device"] == "GPU":

            device_placeholder.success(
                f"""
    🚀 GPU ACTIVE

    Engine:
    {current_engine}

    GPU:
    {device_info['name']}

    VRAM:
    {device_info['memory']} GB
    """
            )

        else:

            device_placeholder.warning(
                f"""
    🖥 CPU ACTIVE

    Engine:
    {current_engine}
    """
            )

        progress_value = float(
            task_state.get("progress", 0.0)
        )

        progress_value = max(
            0.0,
            min(1.0, progress_value)
        )

        progress_placeholder.progress(
            progress_value
        )

        current_file = task_state.get(
            "current_file",
            ""
        )

        current_page = int(
            task_state.get(
                "current_page",
                0
            )
        )

        page_total = int(
            task_state.get(
                "page_total",
                0
            )
        )

        time.sleep(0.5)

    elapsed = time.perf_counter() - task_state["start_time"]
    elapsed_text = format_hms(elapsed)

    if SHOW_PROCESSING_TIMER:
        timer_placeholder.metric("⏱ Processing Time", elapsed_text)
    progress_placeholder.progress(1.0)

    if task_state.get("error"):
        status_placeholder.error(f"OCR failed: {task_state['error']}")
        st.stop()

    result = task_state.get("result") or {}
    final_report = result.get("final_report", "")
    pages_out = result.get("pages_out", [])
    layout_pages = result.get("layout_pages", [])
    failed_files = result.get("failed_files", [])

    status_placeholder.success(f"OCR completed in {elapsed_text}")

    device_info = get_actual_device()

    last_engine = task_state.get(
        "engine",
        "Unknown"
    )

    if device_info["device"] == "GPU":

        device_placeholder.success(
            f"""
    ✅ OCR COMPLETED

    Device:
    GPU

    Engine:
    {last_engine}

    GPU:
    {device_info['name']}

    VRAM Used:
    {device_info['memory']} GB
    """
        )

    else:

        device_placeholder.warning(
            f"""
    ✅ OCR COMPLETED

    Device:
    CPU

    Engine:
    {last_engine}
    """
        )

    st.subheader("Processing time")
    st.info(f"⏱ Total processing time: {elapsed_text}")

    st.subheader("Extracted text")
    st.text_area("OCR output", final_report, height=420)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "Download Word (.docx)",
            data=export_to_docx(final_report),
            file_name="ocr_extracted_text.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Download PDF (.pdf)",
            data=export_to_pdf(final_report),
            file_name="ocr_extracted_text.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with col3:
        if generate_layout_html and layout_pages:
            st.download_button(
                "Download Layout HTML",
                data=export_layout_html(layout_pages, title="OCR Layout Report"),
                file_name="ocr_layout_report.html",
                mime="text/html",
                use_container_width=True,
            )

    if show_preview:
        st.subheader("Processed previews")
        for page in pages_out:
            with st.expander(f"{page['file_name']} — Page {page['page_number']} ({page['engine']}, {page['score']:.2f})"):
                st.image(bgr_to_pil(page["processed_image"]), use_container_width=True)
                st.text(page.get("final_text", ""))
                if page["tables"]:
                    st.markdown("### Tables")
                    for tbl in page["tables"]:
                        st.markdown(f"**Table {tbl.get('table_index', '')}**")
                        df = tbl.get("df")
                        if df is not None and not df.empty:
                            st.dataframe(df, use_container_width=True, hide_index=True)
                        elif tbl.get("html"):
                            st.markdown(tbl["html"], unsafe_allow_html=True)
                        else:
                            st.info("Table detected, but no structured HTML was returned.")

    if show_diagnostics:
        st.subheader("Diagnostics")
        for page in pages_out:
            rep = page.get("report", {})
            with st.expander(f"{page['file_name']} — Page {page['page_number']} diagnostics"):
                st.write(f"Engine: {rep.get('engine', page.get('engine', 'None'))}")
                st.write(f"Score: {rep.get('score', page.get('score', 0.0)):.2f}")
                st.write(f"Quality: {rep.get('quality', 0.0):.2f}")
                st.write(f"Lines: {rep.get('line_count', 0)}")
                st.write(f"Weak lines: {rep.get('weak_line_count', 0)}")
                if rep.get("weak_lines"):
                    st.warning("Weak OCR lines detected")
                    st.text("\n".join(rep["weak_lines"]))
                if is_low_confidence(page.get("final_text", ""), page.get("score", 0.0)):
                    st.error("This page should be reviewed manually.")

    if failed_files:
        st.error("Some files failed.")
        for name, err in failed_files:
            st.write(f"- {name}: {err}")

st.markdown("---")
st.markdown(
    "Trying best..."
)
