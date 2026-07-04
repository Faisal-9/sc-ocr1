from functools import lru_cache
from typing import List, Tuple

import cv2
import numpy as np

from config import USE_GPU

try:
    import torch
except Exception:
    torch = None

try:
    import easyocr
except Exception:
    easyocr = None


@lru_cache(maxsize=1)
def load_easyocr_engine():
    if easyocr is None:
        return None
    try:
        gpu_ok = bool(USE_GPU and torch is not None and torch.cuda.is_available())
        return easyocr.Reader(["ru", "en"], gpu=gpu_ok)
    except Exception:
        return None


def _normalize_bgr(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Empty image passed to EasyOCR.")
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def ocr_easyocr(image: np.ndarray) -> Tuple[str, float]:
    engine = load_easyocr_engine()
    if engine is None:
        return "", 0.0

    image = _normalize_bgr(image)

    try:
        result = engine.readtext(image)
    except Exception:
        return "", 0.0

    texts: List[str] = []
    confs: List[float] = []

    for item in result:
        try:
            txt = item[1]
            conf = float(item[2])
            if txt:
                texts.append(txt)
                confs.append(conf)
        except Exception:
            continue

    text = "\n".join(texts).strip()
    score = float(np.mean(confs)) if confs else 0.0
    return text, score