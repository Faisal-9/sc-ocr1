from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import pytesseract
except Exception:
    pytesseract = None


def _normalize_bgr(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Empty image passed to Tesseract.")
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _to_pil(image: np.ndarray) -> Image.Image:
    image = _normalize_bgr(image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _mean_confidence_from_data(data) -> float:
    confs: List[float] = []
    try:
        for c in data.get("conf", []):
            try:
                c = float(c)
                if c >= 0:
                    confs.append(c / 100.0)
            except Exception:
                continue
    except Exception:
        return 0.0
    return float(np.mean(confs)) if confs else 0.0


def ocr_tesseract(image: np.ndarray, lang: str = "rus+eng", psm: int = 6) -> Tuple[str, float]:
    if pytesseract is None:
        return "", 0.0

    pil_img = _to_pil(image)
    config = f"--oem 3 --psm {psm}"

    try:
        text = pytesseract.image_to_string(pil_img, lang=lang, config=config).strip()
    except Exception:
        return "", 0.0

    score = 0.0
    try:
        data = pytesseract.image_to_data(
            pil_img,
            lang=lang,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
        score = _mean_confidence_from_data(data)
    except Exception:
        score = 0.0

    return text, score