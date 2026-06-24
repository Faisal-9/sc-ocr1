import re
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .paddle_engine import ocr_paddle
from .easyocr_engine import ocr_easyocr
from .tesseract_engine import ocr_tesseract


def _normalize_bgr(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Empty image passed to ensemble OCR.")
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _light_variants(image_bgr: np.ndarray) -> List[np.ndarray]:
    image_bgr = _normalize_bgr(image_bgr)

    variants: List[np.ndarray] = []
    variants.append(image_bgr)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    variants.append(gray)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    variants.append(blur)

    try:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(blur)
        variants.append(enhanced)
    except Exception:
        pass

    try:
        up2 = cv2.resize(image_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        variants.append(up2)
        up2_gray = cv2.cvtColor(up2, cv2.COLOR_BGR2GRAY)
        variants.append(up2_gray)
    except Exception:
        pass

    try:
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)
        variants.append(cv2.bitwise_not(otsu))
    except Exception:
        pass

    uniq = []
    seen = set()
    for v in variants:
        key = (v.shape, int(np.mean(v)), int(np.std(v)))
        if key not in seen:
            seen.add(key)
            uniq.append(v)

    return uniq


def _score_text(text: str, conf: float) -> float:
    if not text or not text.strip():
        return -1.0

    words = re.findall(r"\w+", text, flags=re.UNICODE)
    length_bonus = min(len(text) / 6000.0, 0.5)
    word_bonus = min(len(words) / 120.0, 0.35)
    return (conf * 1.5) + length_bonus + word_bonus


def ocr_ensemble(image_bgr: np.ndarray, lang: str = "rus+eng") -> Dict[str, object]:
    variants = _light_variants(image_bgr)

    best_text = ""
    best_score = -1.0
    best_engine = ""
    best_variant_index = 0

    for i, variant in enumerate(variants):
        candidates: List[Tuple[str, Tuple[str, float]]] = []

        try:
            pt, pc = ocr_paddle(variant)
            candidates.append(("PaddleOCR", (pt, pc)))
        except Exception:
            candidates.append(("PaddleOCR", ("", 0.0)))

        try:
            et, ec = ocr_easyocr(variant)
            candidates.append(("EasyOCR", (et, ec)))
        except Exception:
            candidates.append(("EasyOCR", ("", 0.0)))

        try:
            tt, tc = ocr_tesseract(variant, lang=lang, psm=6)
            candidates.append(("Tesseract", (tt, tc)))
        except Exception:
            candidates.append(("Tesseract", ("", 0.0)))

        for engine_name, (text, conf) in candidates:
            score = _score_text(text, conf)
            if score > best_score:
                best_score = score
                best_text = text.strip()
                best_engine = engine_name
                best_variant_index = i

    return {
        "text": best_text,
        "score": max(best_score, 0.0),
        "engine": best_engine or "None",
        "variant_index": best_variant_index,
    }