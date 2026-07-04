from functools import lru_cache
from typing import List, Tuple

import cv2
import numpy as np

from config import USE_GPU
from utils.gpu import gpu_available

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None


@lru_cache(maxsize=1)
def load_paddle_engine():
    if PaddleOCR is None:
        return None

    try:
        return PaddleOCR(
            use_angle_cls=True,
            lang="ru",
            use_gpu=bool(USE_GPU and gpu_available()),
            show_log=False,
        )
    except Exception:
        return None


def _normalize_bgr(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Empty image passed to PaddleOCR.")
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def ocr_paddle(image: np.ndarray) -> Tuple[str, float]:
    engine = load_paddle_engine()
    if engine is None:
        return "", 0.0

    image = _normalize_bgr(image)

    try:
        result = engine.ocr(image, cls=True)
    except Exception:
        return "", 0.0

    texts: List[str] = []
    confs: List[float] = []

    if result:
        for page in result:
            if not page:
                continue
            for item in page:
                try:
                    txt = item[1][0]
                    conf = float(item[1][1])
                    if txt:
                        texts.append(txt)
                        confs.append(conf)
                except Exception:
                    continue

    text = "\n".join(texts).strip()
    score = float(np.mean(confs)) if confs else 0.0
    return text, score