import cv2
import numpy as np


def crop_document(image):
    if image is None:
        raise ValueError("Empty image passed to crop_document.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    best_box = None
    best_area = 0
    thresholds = [5, 8, 12, 15, 20, 25, 30, 40, 50]

    for t in thresholds:
        _, mask = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 10 or bh < 10:
                continue
            if area > best_area:
                best_area = area
                best_box = (x, y, bw, bh)

    if best_box is None:
        return image

    x, y, bw, bh = best_box
    pad_x = max(10, int(bw * 0.08))
    pad_y = max(10, int(bh * 0.08))

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return image
    return cropped