from __future__ import annotations

import re
from typing import Dict, List, Tuple

import cv2
import numpy as np

from config import (
    ENABLE_ENSEMBLE,
    PRIMARY_OCR_ENGINE,
    USE_EASYOCR_FALLBACK,
    USE_TESSERACT_FALLBACK,
)
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
    variants: List[np.ndarray] = [image_bgr]

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


def _try_engine(name: str, image: np.ndarray, lang: str) -> Tuple[str, float]:
    if name == "paddle":
        return ocr_paddle(image)
    if name == "easyocr":
        return ocr_easyocr(image)
    return ocr_tesseract(image, lang=lang, psm=6)


def ocr_ensemble(image_bgr: np.ndarray, lang: str = "rus+eng") -> Dict[str, object]:
    variants = _light_variants(image_bgr)

    if not ENABLE_ENSEMBLE:
        engine_order = [PRIMARY_OCR_ENGINE]
    else:
        engine_order = [PRIMARY_OCR_ENGINE]
        if USE_EASYOCR_FALLBACK and PRIMARY_OCR_ENGINE != "easyocr":
            engine_order.append("easyocr")
        if USE_TESSERACT_FALLBACK:
            engine_order.append("tesseract")

    # dedupe while preserving order
    seen_engines = set()
    engine_order = [e for e in engine_order if not (e in seen_engines or seen_engines.add(e))]

    best_text = ""
    best_score = -1.0
    best_engine = ""
    best_variant_index = 0

    for i, variant in enumerate(variants):
        for engine_name in engine_order:
            try:
                text, conf = _try_engine(engine_name, variant, lang)
            except Exception:
                text, conf = "", 0.0

            score = _score_text(text, conf)
            if score > best_score:
                best_score = score
                best_text = text.strip()
                best_engine = engine_name
                best_variant_index = i


    print("Selected OCR Engine:", best_engine)
    return {
        "text": best_text,
        "score": max(best_score, 0.0),
        "engine": best_engine or "None",
        "variant_index": best_variant_index,
    }