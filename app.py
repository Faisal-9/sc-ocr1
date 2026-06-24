"""Hard Scan OCR Extractor with layout-preserving export.

Features:
- PyMuPDF rendering (no Poppler)
- All pages or fixed page range
- Page numbering using original page numbers
- Conservative OCR by default (exact text preserved)
- Table extraction
- Layout-preserving HTML export
- Figure/image block extraction
- OCR overlay placement for scanned pages using PaddleOCR boxes

Run:
    streamlit run app.py
"""

import base64
import os
import re
from io import StringIO
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
from ocr.paddle_engine import load_paddle_engine
from language import correct_text
from export import export_to_docx, export_to_pdf
from utils.confidence import format_review_report, is_low_confidence

try:
    from paddleocr import PPStructure
except Exception:
    PPStructure = None


# ----------------------------
# General helpers
# ----------------------------

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


def safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except Exception:
        pass


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


def image_to_b64(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def bbox_to_px(bbox: Tuple[float, float, float, float], zoom: float) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (
        int(round(x0 * zoom)),
        int(round(y0 * zoom)),
        int(round(x1 * zoom)),
        int(round(y1 * zoom)),
    )


def crop_bgr(image_bgr: np.ndarray, bbox_px: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox_px
    h, w = image_bgr.shape[:2]
    x0 = max(0, min(w, x0))
    y0 = max(0, min(h, y0))
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return image_bgr
    return image_bgr[y0:y1, x0:x1].copy()


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


# ----------------------------
# Layout extraction helpers
# ----------------------------

@st.cache_resource(show_spinner=False)
def get_table_engine():
    if PPStructure is None:
        return None
    try:
        return PPStructure(show_log=False)
    except Exception:
        return None


def text_from_native_block(block: Dict[str, Any]) -> str:
    lines = []
    for line in block.get("lines", []):
        spans = line.get("spans", [])
        line_text = "".join(span.get("text", "") for span in spans).strip()
        if line_text:
            lines.append(line_text)
    return "\n".join(lines).strip()


def extract_native_text_blocks(page: fitz.Page, zoom: float) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    try:
        data = page.get_text("dict")
    except Exception:
        return items

    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        text = text_from_native_block(block)
        bbox = block.get("bbox")
        if not text or not bbox:
            continue
        items.append(
            {
                "kind": "text",
                "bbox_px": bbox_to_px(tuple(bbox), zoom),
                "text": text,
                "conf": 1.0,
                "source": "native",
            }
        )
    return items


def extract_image_blocks_from_native(page: fitz.Page, page_bgr: np.ndarray, zoom: float) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    try:
        data = page.get_text("dict")
    except Exception:
        return items

    for block in data.get("blocks", []):
        if block.get("type") != 1:
            continue
        bbox = block.get("bbox")
        if not bbox:
            continue
        bbox_px = bbox_to_px(tuple(bbox), zoom)
        crop = crop_bgr(page_bgr, bbox_px)
        items.append(
            {
                "kind": "image",
                "bbox_px": bbox_px,
                "image_b64": image_to_b64(crop),
            }
        )
    return items


def extract_tables_from_page(page: fitz.Page, page_bgr: np.ndarray, zoom: float) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []
    if not hasattr(page, "find_tables"):
        return tables

    try:
        finder = page.find_tables()
    except Exception:
        return tables

    found = getattr(finder, "tables", finder)
    if not found:
        return tables

    for idx, table in enumerate(found, start=1):
        try:
            bbox = getattr(table, "bbox", None)
            if bbox is None and isinstance(table, dict):
                bbox = table.get("bbox")
            if bbox is None:
                continue

            bbox_px = bbox_to_px(tuple(bbox), zoom)
            crop = crop_bgr(page_bgr, bbox_px)
            rows = []
            if hasattr(table, "extract"):
                try:
                    rows = table.extract()
                except Exception:
                    rows = []

            tables.append(
                {
                    "kind": "table",
                    "table_index": idx,
                    "bbox_px": bbox_px,
                    "image_b64": image_to_b64(crop),
                    "rows": rows,
                }
            )
        except Exception:
            continue
    return tables


def extract_text_boxes_with_paddle(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    engine = load_paddle_engine()
    if engine is None:
        return []

    try:
        result = engine.ocr(image_bgr, cls=True)
    except Exception:
        return []

    items: List[Dict[str, Any]] = []
    if not result:
        return items

    for page_result in result:
        if not page_result:
            continue
        for det in page_result:
            try:
                box = det[0]
                text = det[1][0]
                conf = float(det[1][1])
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                bbox_px = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                items.append(
                    {
                        "kind": "text",
                        "bbox_px": bbox_px,
                        "text": text,
                        "conf": conf,
                        "source": "paddleocr",
                    }
                )
            except Exception:
                continue
    return items


def render_page_bgr(page: fitz.Page, zoom: float = 5.0) -> np.ndarray:
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    if pix.n == 4:
        img = arr.reshape(pix.height, pix.width, 4)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if pix.n == 3:
        img = arr.reshape(pix.height, pix.width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = arr.reshape(pix.height, pix.width)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


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
            if not isinstance(item, dict) or item.get("type") != "table":
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
                    parsed = pd.read_html(StringIO(html))
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


def extract_page_layout(page: fitz.Page, zoom: float = 5.0) -> Dict[str, Any]:
    page_bgr = render_page_bgr(page, zoom=zoom)
    width_px, height_px = page_bgr.shape[1], page_bgr.shape[0]

    native_text = extract_native_text_blocks(page, zoom=zoom)
    images = extract_image_blocks_from_native(page, page_bgr, zoom=zoom)
    tables = extract_tables_from_page(page, page_bgr, zoom=zoom)

    text_items = native_text if native_text else extract_text_boxes_with_paddle(page_bgr)
    elements = text_items + images + tables
    elements.sort(key=lambda item: (item["bbox_px"][1], item["bbox_px"][0]))

    return {
        "page_width_px": width_px,
        "page_height_px": height_px,
        "page_b64": image_to_b64(page_bgr),
        "elements": elements,
        "text_items": text_items,
        "image_items": images,
        "table_items": tables,
    }


def extract_layout_from_image(image_bgr: np.ndarray) -> Dict[str, Any]:
    height_px, width_px = image_bgr.shape[:2]
    text_items = extract_text_boxes_with_paddle(image_bgr)
    return {
        "page_width_px": width_px,
        "page_height_px": height_px,
        "page_b64": image_to_b64(image_bgr),
        "elements": text_items,
        "text_items": text_items,
        "image_items": [],
        "table_items": [],
    }


def table_rows_to_html(rows: List[List[Any]]) -> str:
    if not rows:
        return ""
    parts = ["<table class='table-box'>"]
    for r, row in enumerate(rows):
        parts.append("<tr>")
        for cell in row:
            tag = "th" if r == 0 else "td"
            cell_text = "" if cell is None else str(cell)
            parts.append(f"<{tag}>{cell_text}</{tag}>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def export_layout_html(pages: List[Dict[str, Any]], title: str = "OCR Layout Report") -> bytes:
    css = """
    <style>
      body { font-family: Arial, sans-serif; margin: 0; background: #f4f4f4; }
      .wrap { padding: 20px; }
      .page {
        position: relative;
        margin: 0 auto 28px auto;
        background-repeat: no-repeat;
        background-size: contain;
        background-position: top left;
        box-shadow: 0 2px 16px rgba(0,0,0,.12);
        border: 1px solid #ddd;
        background-color: white;
      }
      .overlay-text {
        position: absolute;
        color: rgba(0,0,0,0.01);
        white-space: pre-wrap;
        line-height: 1.15;
        user-select: text;
      }
      .section-title {
        margin: 12px 0 8px;
        font-size: 14px;
        font-weight: 700;
      }
      .asset-row { display: flex; gap: 12px; flex-wrap: wrap; }
      .asset-card {
        background: white;
        border: 1px solid #ddd;
        padding: 10px;
        margin-bottom: 10px;
      }
      .asset-img { max-width: 100%; display: block; }
      .table-box { border-collapse: collapse; width: 100%; margin-top: 8px; background: white; }
      .table-box td, .table-box th { border: 1px solid #777; padding: 4px 6px; font-size: 12px; vertical-align: top; }
      .meta { font-size: 12px; color: #666; margin-bottom: 6px; }
    </style>
    """

    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        css,
        "</head><body><div class='wrap'>",
        f"<h1>{title}</h1>",
    ]

    for i, page in enumerate(pages, start=1):
        w = int(page["page_width_px"])
        h = int(page["page_height_px"])
        page_b64 = page["page_b64"]
        parts.append(
            f"<div class='page' style='width:{w}px;height:{h}px;background-image:url(data:image/png;base64,{page_b64});'>"
        )

        for item in page.get("elements", []):
            if item.get("kind") != "text":
                continue
            x0, y0, x1, y1 = item["bbox_px"]
            left = max(0, x0)
            top = max(0, y0)
            width = max(1, x1 - x0)
            height = max(1, y1 - y0)
            txt = str(item.get("text", ""))
            parts.append(
                f"<div class='overlay-text' style='left:{left}px;top:{top}px;width:{width}px;height:{height}px;'>"
                f"{txt}</div>"
            )

        parts.append("</div>")

        images = page.get("image_items", [])
        tables = page.get("table_items", [])
        if images or tables:
            parts.append(f"<div class='section-title'>Page {i} extracted assets</div>")

        if images:
            parts.append("<div class='asset-row'>")
            for img in images:
                img_b64 = img.get("image_b64", "")
                if not img_b64:
                    continue
                parts.append(
                    "<div class='asset-card'>"
                    "<div class='meta'>Extracted image / figure</div>"
                    f"<img class='asset-img' src='data:image/png;base64,{img_b64}' />"
                    "</div>"
                )
            parts.append("</div>")

        if tables:
            for t in tables:
                rows = t.get("rows", [])
                parts.append("<div class='asset-card'>")
                parts.append(f"<div class='meta'>Table {t.get('table_index', '')}</div>")
                if rows:
                    parts.append(table_rows_to_html(rows))
                img_b64 = t.get("image_b64", "")
                if img_b64:
                    parts.append("<div class='meta' style='margin-top:8px;'>Detected table region</div>")
                    parts.append(f"<img class='asset-img' src='data:image/png;base64,{img_b64}' />")
                parts.append("</div>")

    parts.append("</div></body></html>")
    return "\n".join(parts).encode("utf-8")


# ----------------------------
# OCR processing
# ----------------------------

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
    layout = extract_layout_from_image(original)

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
                # Keep the OCR box layout for scanned pages.
                layout["text_items"] = extract_text_boxes_with_paddle(page_bgr)
                layout["elements"] = layout["text_items"] + layout.get("image_items", []) + layout.get("table_items", [])
                layout["elements"].sort(key=lambda item: (item["bbox_px"][1], item["bbox_px"][0]))

            if not extract_tables:
                layout["table_items"] = []

            tables = extract_tables_from_page(page, page_bgr, zoom=zoom) if extract_tables else []

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


# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title="Hard Scan OCR Extractor", layout="wide")
st.title("Hard Scan OCR Extractor")
st.caption("Russian/English OCR with page range support, page numbering, and layout-preserving HTML export.")

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
    st.write("For exact transcription, keep correction off.")
    st.write("Layout HTML is the closest to the original page structure.")

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
    layout_pages: List[Dict[str, Any]] = []

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
                layout_pages.append(page["layout"])

        except Exception as exc:
            failed_files.append((file.name, str(exc)))

        progress.progress(idx / total_tasks)

    final_report = "\n\n".join(all_blocks).strip()

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
    "Trying best for ocr with its structure"
)
