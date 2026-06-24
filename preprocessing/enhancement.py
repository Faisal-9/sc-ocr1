import cv2
import numpy as np


def enhance_document(image):
    if image is None:
        raise ValueError("Empty image passed to enhance_document.")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return enhanced