from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

from ocr.paddle_engine import load_paddle_engine


@dataclass
class OCRWord:
    text: str
    conf: float
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def w(self) -> int:
        return max(1, self.x1 - self.x0)

    @property
    def h(self) -> int:
        return max(1, self.y1 - self.y0)


def _normalize_image(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None:
        raise ValueError("Empty image passed to table reconstruction.")
    if len(image_bgr.shape) == 2:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    return image_bgr


def _is_numeric_token(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    return any(ch.isdigit() for ch in text)


def _is_short_token(text: str) -> bool:
    t = text.strip()
    return len(t) <= 4


def _words_from_paddle(image_bgr: np.ndarray, min_conf: float = 0.20) -> List[OCRWord]:
    engine = load_paddle_engine()
    if engine is None:
        return []

    image_bgr = _normalize_image(image_bgr)

    try:
        result = engine.ocr(image_bgr, cls=True)
    except Exception:
        return []

    words: List[OCRWord] = []
    if not result:
        return words

    for page_result in result:
        if not page_result:
            continue
        for det in page_result:
            try:
                box = det[0]
                text = str(det[1][0]).strip()
                conf = float(det[1][1])

                if not text or conf < min_conf:
                    continue

                xs = [int(p[0]) for p in box]
                ys = [int(p[1]) for p in box]
                words.append(
                    OCRWord(
                        text=text,
                        conf=conf,
                        x0=min(xs),
                        y0=min(ys),
                        x1=max(xs),
                        y1=max(ys),
                    )
                )
            except Exception:
                continue

    words.sort(key=lambda w: (w.cy, w.cx))
    return words


def cluster_words_into_rows(words: Sequence[OCRWord]) -> List[Dict[str, Any]]:
    if not words:
        return []

    heights = [w.h for w in words]
    ys = sorted([w.cy for w in words])
    med_h = max(8, int(median(heights))) if heights else 12
    y_tol = max(8, int(med_h * 0.75))

    rows: List[Dict[str, Any]] = []
    current: List[OCRWord] = []
    current_y: Optional[float] = None

    for word in sorted(words, key=lambda w: (w.cy, w.cx)):
        if current_y is None:
            current = [word]
            current_y = word.cy
            continue

        if abs(word.cy - current_y) <= y_tol:
            current.append(word)
            current_y = sum(w.cy for w in current) / len(current)
        else:
            rows.append(_build_row(current))
            current = [word]
            current_y = word.cy

    if current:
        rows.append(_build_row(current))

    return rows


def _build_row(words: Sequence[OCRWord]) -> Dict[str, Any]:
    x0 = min(w.x0 for w in words)
    y0 = min(w.y0 for w in words)
    x1 = max(w.x1 for w in words)
    y1 = max(w.y1 for w in words)

    row_width = max(1, x1 - x0)
    token_count = len(words)
    numeric_ratio = sum(1 for w in words if _is_numeric_token(w.text)) / max(token_count, 1)
    short_ratio = sum(1 for w in words if _is_short_token(w.text)) / max(token_count, 1)

    return {
        "words": sorted(list(words), key=lambda w: w.cx),
        "bbox": (x0, y0, x1, y1),
        "center_y": (y0 + y1) / 2.0,
        "row_width": row_width,
        "token_count": token_count,
        "numeric_ratio": numeric_ratio,
        "short_ratio": short_ratio,
    }


def _cluster_x_centers(words: Sequence[OCRWord], tol: Optional[int] = None) -> List[Dict[str, Any]]:
    if not words:
        return []

    widths = [w.w for w in words]
    if tol is None:
        tol = max(12, int(median(widths) * 1.25)) if widths else 18

    clusters: List[Dict[str, Any]] = []
    for word in sorted(words, key=lambda w: w.cx):
        placed = False
        for c in clusters:
            if abs(word.cx - c["center"]) <= tol:
                c["members"].append(word)
                c["center"] = sum(w.cx for w in c["members"]) / len(c["members"])
                placed = True
                break
        if not placed:
            clusters.append({"center": word.cx, "members": [word]})

    clusters.sort(key=lambda c: c["center"])
    return clusters


def _assign_words_to_clusters(
    rows: Sequence[Dict[str, Any]],
    clusters: Sequence[Dict[str, Any]],
    tol: Optional[int] = None,
) -> Tuple[List[List[str]], List[float]]:
    if not rows or not clusters:
        return [], []

    all_widths = []
    for row in rows:
        for w in row["words"]:
            all_widths.append(w.w)

    if tol is None:
        tol = max(12, int(median(all_widths) * 1.4)) if all_widths else 18

    cell_rows: List[List[str]] = []
    cluster_presence = [0 for _ in clusters]

    for row in rows:
        cells = [""] * len(clusters)
        for word in sorted(row["words"], key=lambda w: w.cx):
            best_idx = None
            best_dist = None
            for i, c in enumerate(clusters):
                dist = abs(word.cx - c["center"])
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx is None:
                continue

            if best_dist is not None and best_dist <= tol * 1.5:
                if cells[best_idx]:
                    cells[best_idx] = f"{cells[best_idx]} {word.text}".strip()
                else:
                    cells[best_idx] = word.text
            else:
                # Still place it into the nearest column so rare odd spacing does not break the table.
                if cells[best_idx]:
                    cells[best_idx] = f"{cells[best_idx]} {word.text}".strip()
                else:
                    cells[best_idx] = word.text

        for i, cell in enumerate(cells):
            if cell.strip():
                cluster_presence[i] += 1

        cell_rows.append(cells)

    presence_ratio = [p / max(len(rows), 1) for p in cluster_presence]
    return cell_rows, presence_ratio


def _row_gap(prev_row: Dict[str, Any], next_row: Dict[str, Any]) -> int:
    return int(max(0, next_row["bbox"][1] - prev_row["bbox"][3]))


def _table_block_score(rows: Sequence[Dict[str, Any]], presence_ratio: Sequence[float]) -> float:
    if not rows:
        return 0.0

    row_count = len(rows)
    avg_tokens = sum(r["token_count"] for r in rows) / row_count
    avg_numeric = sum(r["numeric_ratio"] for r in rows) / row_count
    avg_short = sum(r["short_ratio"] for r in rows) / row_count
    widths = [r["row_width"] for r in rows]
    width_cv = (float(np.std(widths)) / max(float(np.mean(widths)), 1.0)) if len(widths) > 1 else 0.0

    stable_cols = sum(1 for p in presence_ratio if p >= 0.35)
    stable_bonus = max(0.0, stable_cols - 1)

    # Borderless tables usually have repeated columns, short/numeric tokens, and similar row widths.
    score = (
        row_count * 0.9
        + avg_tokens * 0.3
        + avg_numeric * 2.0
        + avg_short * 0.6
        + stable_bonus * 2.0
    )
    score = score / (1.0 + width_cv)
    return float(score)


def _is_table_like(rows: Sequence[Dict[str, Any]], presence_ratio: Sequence[float]) -> bool:
    if len(rows) < 2:
        return False

    stable_cols = sum(1 for p in presence_ratio if p >= 0.35)
    if stable_cols < 2:
        return False

    avg_numeric = sum(r["numeric_ratio"] for r in rows) / max(len(rows), 1)
    avg_tokens = sum(r["token_count"] for r in rows) / max(len(rows), 1)

    if avg_tokens < 2:
        return False

    # Stronger if there are numbers and short technical tokens.
    if avg_numeric >= 0.15:
        return True

    # Otherwise the column stability must be very good.
    return stable_cols >= 3 and avg_tokens >= 3


def _rows_to_bbox(rows: Sequence[Dict[str, Any]]) -> Tuple[int, int, int, int]:
    x0 = min(r["bbox"][0] for r in rows)
    y0 = min(r["bbox"][1] for r in rows)
    x1 = max(r["bbox"][2] for r in rows)
    y1 = max(r["bbox"][3] for r in rows)
    return int(x0), int(y0), int(x1), int(y1)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""

    def esc(v: Any) -> str:
        if pd.isna(v):
            return ""
        return str(v).replace("\n", " ").replace("|", "\\|").strip()

    headers = [esc(c) for c in df.columns]
    rows = [[esc(v) for v in row] for row in df.fillna("").values.tolist()]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, sep_line] + body)


def _build_table_result(rows: Sequence[Dict[str, Any]], page_bgr: np.ndarray, table_index: int) -> Optional[Dict[str, Any]]:
    if not rows:
        return None

    all_words = [w for row in rows for w in row["words"]]
    clusters = _cluster_x_centers(all_words)
    if len(clusters) < 2:
        return None

    cell_rows, presence_ratio = _assign_words_to_clusters(rows, clusters)
    if not _is_table_like(rows, presence_ratio):
        return None

    keep_cols = [i for i, p in enumerate(presence_ratio) if p >= 0.20]
    if len(keep_cols) < 2:
        return None

    filtered_rows = [[r[i].strip() for i in keep_cols] for r in cell_rows]

    # Remove rows that are completely empty after assignment.
    filtered_rows = [r for r in filtered_rows if any(cell.strip() for cell in r)]
    if len(filtered_rows) < 2:
        return None

    df = pd.DataFrame(filtered_rows)
    bbox = _rows_to_bbox(rows)

    return {
        "kind": "table",
        "table_index": table_index,
        "bbox_px": bbox,
        "rows": filtered_rows,
        "df": df,
        "html": df.to_html(index=False, header=False, escape=False) if df is not None else "",
        "markdown": dataframe_to_markdown(df),
        "score": _table_block_score(rows, presence_ratio),
    }


def _split_into_table_blocks(rows: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    if not rows:
        return []

    dense_flags = []
    for r in rows:
        dense = r["token_count"] >= 2 and (r["row_width"] >= 120)
        dense_flags.append(dense)

    blocks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        if dense_flags[idx]:
            if not current:
                current = [row]
            else:
                gap = _row_gap(current[-1], row)
                # Allow a small gap inside a table, but split on large paragraph-like spacing.
                if gap <= max(18, int(median([r["bbox"][3] - r["bbox"][1] for r in current]) * 1.2)):
                    current.append(row)
                else:
                    if len(current) >= 2:
                        blocks.append(current)
                    current = [row]
        else:
            if len(current) >= 2:
                blocks.append(current)
            current = []

    if len(current) >= 2:
        blocks.append(current)

    return blocks


def extract_borderless_tables(image_bgr: np.ndarray, max_tables: int = 3) -> List[Dict[str, Any]]:
    """
    Reconstruct tables from OCR word positions even when there are no border lines.
    Best for scanned pages like the uploaded sample.
    """
    image_bgr = _normalize_image(image_bgr)
    words = _words_from_paddle(image_bgr)
    if not words:
        return []

    rows = cluster_words_into_rows(words)
    if len(rows) < 2:
        return []

    blocks = _split_into_table_blocks(rows)
    if not blocks:
        # As a fallback, evaluate the whole page as one block.
        blocks = [rows]

    candidates: List[Dict[str, Any]] = []
    for idx, block_rows in enumerate(blocks, start=1):
        table = _build_table_result(block_rows, image_bgr, idx)
        if table is not None:
            candidates.append(table)

    candidates.sort(key=lambda t: t.get("score", 0.0), reverse=True)

    # Remove near-duplicate overlapping candidates.
    selected: List[Dict[str, Any]] = []
    seen_boxes: List[Tuple[int, int, int, int]] = []

    for cand in candidates:
        if len(selected) >= max_tables:
            break

        x0, y0, x1, y1 = cand["bbox_px"]
        area = max(1, (x1 - x0) * (y1 - y0))
        overlap = False

        for sx0, sy0, sx1, sy1 in seen_boxes:
            ix0 = max(x0, sx0)
            iy0 = max(y0, sy0)
            ix1 = min(x1, sx1)
            iy1 = min(y1, sy1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            inter = (ix1 - ix0) * (iy1 - iy0)
            other = max(1, (sx1 - sx0) * (sy1 - sy0))
            if inter / min(area, other) > 0.65:
                overlap = True
                break

        if not overlap:
            selected.append(cand)
            seen_boxes.append(cand["bbox_px"])

    return selected


def extract_word_layout(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """
    Returns OCR words with boxes. Useful if you later want to draw the exact text position.
    """
    image_bgr = _normalize_image(image_bgr)
    words = _words_from_paddle(image_bgr)
    return [
        {
            "text": w.text,
            "conf": w.conf,
            "bbox_px": (w.x0, w.y0, w.x1, w.y1),
        }
        for w in words
    ]