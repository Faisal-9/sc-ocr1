import base64
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import cv2
import fitz
import numpy as np

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None


@lru_cache(maxsize=1)
def get_box_ocr():
    if PaddleOCR is None:
        return None
    try:
        return PaddleOCR(use_angle_cls=True, lang="ru", show_log=False)
    except Exception:
        return None


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


def text_from_native_block(block: Dict[str, Any]) -> str:
    lines = []
    for line in block.get("lines", []):
        spans = line.get("spans", [])
        line_text = "".join(span.get("text", "") for span in spans).strip()
        if line_text:
            lines.append(line_text)
    return "\n".join(lines).strip()


def extract_text_boxes_with_paddle(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    engine = get_box_ocr()
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
                    }
                )
            except Exception:
                continue

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
        if not text:
            continue

        bbox = block.get("bbox")
        if not bbox:
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


def extract_page_layout(page: fitz.Page, zoom: float = 5.0) -> Dict[str, Any]:
    page_bgr = render_page_bgr(page, zoom=zoom)
    width_px, height_px = page_bgr.shape[1], page_bgr.shape[0]

    native_text = extract_native_text_blocks(page, zoom=zoom)
    images = extract_image_blocks_from_native(page, page_bgr, zoom=zoom)
    tables = extract_tables_from_page(page, page_bgr, zoom=zoom)

    # If the PDF has no usable text layer, use OCR boxes.
    if native_text:
        text_items = native_text
    else:
        text_items = extract_text_boxes_with_paddle(page_bgr)

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