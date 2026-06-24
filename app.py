import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import fitz
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from preprocessing import crop_document, deskew, enhance_document, upscale_document
from ocr.ensemble import ocr_ensemble
from language import correct_text
from export import export_to_docx, export_to_pdf
from utils.confidence import format_review_report, is_low_confidence

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
        s = s.replace("\n", " ").replace("|", "\\|")
        return s.strip()

    headers = [esc(c) for c in df.columns]
    rows = [[esc(v) for v in row] for row in df.fillna("").values.tolist()]

    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, sep_line] + data_lines)


@st.cache_resource(show_spinner=False)
def get_table_engine():
    if PPStructure is None:
        return None
    try:
        return PPStructure(show_log=False)
    except Exception:
        return None


def crop_and_prepare_for_table(image_bgr: np.ndarray) -> np.ndarray:
    img = crop_document(image_bgr)
    img = deskew(img)
    img = upscale_document(img, factor=2)
    return img


def extract_tables_from_image(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    engine = get_table_engine()
    if engine is None:
        return []

    try:
        table_img = crop_and_prepare_for_table(image_bgr)
        result = engine(table_img)
    except Exception:
        return []

    tables: List[Dict[str, Any]] = []
    if not result:
        return tables

    for idx, item in enumerate(result, start=1):
        try:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "table":
                continue

            bbox = item.get("bbox")
            res = item.get("res")
            html = ""

            if isinstance(res, dict):
                html = res.get("html", "") or ""
            elif isinstance(res, str):
                html = res

            df = None
            if html:
                try:
                    parsed = pd.read_html(html)
                    if parsed:
                        df = parsed[0]
                except Exception:
                    df = None

            tables.append(
                {
                    "table_index": idx,
                    "bbox": bbox,
                    "html": html,
                    "df": df,
                    "markdown": dataframe_to_markdown(df) if df is not None else "",
                }
            )
        except Exception:
            continue

    return tables


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


def process_image_page(
    image_bytes: bytes,
    file_name: str,
    page_number: int,
    lang: str,
    extract_tables: bool,
) -> Dict[str, Any]:
    original = image_bytes_to_bgr(image_bytes)
    processed = preprocess_image(original)
    ocr_result = ocr_ensemble(processed, lang=lang)

    raw_text = str(ocr_result.get("text", ""))
    score = float(ocr_result.get("score", 0.0))
    engine = str(ocr_result.get("engine", "None"))
    tables = extract_tables_from_image(original) if extract_tables else []

    return {
        "file_name": file_name,
        "page_number": page_number,
        "engine": engine,
        "score": score,
        "raw_text": raw_text,
        "processed_image": processed,
        "tables": tables,
        "report": format_review_report(raw_text, score, engine),
    }


def render_pdf_page_to_bgr(page, zoom: float = 5.0) -> np.ndarray:
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8)

    if pix.n == 4:
        img = img.reshape(pix.height, pix.width, 4)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = img.reshape(pix.height, pix.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = img.reshape(pix.height, pix.width)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return 0


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
) -> List[Dict[str, Any]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    results: List[Dict[str, Any]] = []

    try:
        total_pages = len(doc)
        if total_pages <= 0:
            return results

        if page_mode == "Page range":
            first_page = max(1, start_page)
            last_page = min(end_page, total_pages)
            if first_page > last_page:
                first_page, last_page = last_page, first_page
        else:
            first_page = 1
            last_page = total_pages

        for page_number in range(first_page, last_page + 1):
            page = doc.load_page(page_number - 1)

            page_text = ""
            if prefer_text_layer:
                try:
                    page_text = (page.get_text("text") or "").strip()
                except Exception:
                    page_text = ""

            page_bgr = render_pdf_page_to_bgr(page, zoom=zoom)
            processed = preprocess_image(page_bgr)

            if page_text:
                raw_text = page_text
                score = 1.0
                engine = "PDF text layer"
            else:
                ocr_result = ocr_ensemble(processed, lang=lang)
                raw_text = str(ocr_result.get("text", ""))
                score = float(ocr_result.get("score", 0.0))
                engine = str(ocr_result.get("engine", "None"))

            tables = extract_tables_from_image(page_bgr) if extract_tables else []

            results.append(
                {
                    "file_name": file_name,
                    "page_number": page_number,
                    "engine": engine,
                    "score": score,
                    "raw_text": raw_text,
                    "processed_image": processed,
                    "tables": tables,
                    "report": format_review_report(raw_text, score, engine),
                }
            )
    finally:
        doc.close()

    return results


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


def maybe_correct_text(text: str, apply_correction: bool) -> str:
    text = normalize_text(text)
    if apply_correction:
        text = correct_text(text)
        text = normalize_text(text)
    return text.strip()


st.set_page_config(page_title="Hard Scan OCR Extractor", layout="wide")
st.title("Hard Scan OCR Extractor")
st.caption("Russian/English OCR with page range support, page numbering, and table extraction.")

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
    show_preview = st.checkbox("Show processed preview", value=True)
    show_diagnostics = st.checkbox("Show diagnostics", value=True)
    st.markdown("---")
    st.write("For range mode, the same page range is applied to every uploaded PDF.")
    st.write("If you need exact transcription, keep correction off.")

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

    all_blocks: List[str] = []
    failed_files: List[Tuple[str, str]] = []
    pages_out: List[Dict[str, Any]] = []

    progress = st.progress(0)
    total_tasks = len(uploaded_files)

    for idx, file in enumerate(uploaded_files, start=1):
        try:
            name = file.name
            ext = Path(name).suffix.lower()
            data = file.getvalue()

            if ext == ".pdf":
                page_results = process_pdf_file(
                    pdf_bytes=data,
                    file_name=name,
                    lang=lang,
                    extract_tables=extract_tables,
                    page_mode=page_mode,
                    start_page=int(start_page),
                    end_page=int(end_page),
                    prefer_text_layer=prefer_text_layer,
                    zoom=5.0,
                )
            else:
                page_results = [
                    process_image_page(
                        image_bytes=data,
                        file_name=name,
                        page_number=1,
                        lang=lang,
                        extract_tables=extract_tables,
                    )
                ]

            for page in page_results:
                raw_text = page["raw_text"]

                if preserve_exact:
                    final_text = raw_text.strip()
                    if normalize_ws:
                        final_text = normalize_text(final_text)
                else:
                    final_text = maybe_correct_text(raw_text, apply_language_correction)
                    if not normalize_ws:
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

        except Exception as exc:
            failed_files.append((file.name, str(exc)))

        progress.progress(idx / total_tasks)

    final_report = "\n\n".join(all_blocks).strip()

    st.subheader("Extracted text")
    st.text_area("OCR output", final_report, height=420)

    col1, col2 = st.columns(2)
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

    if show_preview:
        st.subheader("Processed previews")
        for page in pages_out:
            with st.expander(
                f"{page['file_name']} — Page {page['page_number']} ({page['engine']}, {page['score']:.2f})"
            ):
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
    "best handled"
)