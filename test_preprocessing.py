import cv2

from preprocessing import (
    crop_document,
    deskew,
    enhance_document,
    upscale_document
)

img = cv2.imread("C:/Users/AFS/Pictures/Screenshots/abbc23.png")
img = crop_document(img)
img = deskew(img)
img = enhance_document(img)
img = upscale_document(img)
cv2.imwrite(
    "result.png",
    img
)