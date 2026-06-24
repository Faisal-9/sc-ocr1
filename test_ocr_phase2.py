import cv2
from preprocessing import crop_document, deskew, enhance_document, upscale_document
from ocr.ensemble import ocr_ensemble

IMAGE_PATH = "C:/Users/AFS/Pictures/Screenshots/aaaaa11.png"

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

img = crop_document(img)
img = deskew(img)
img = enhance_document(img)
img = upscale_document(img, factor=4)

result = ocr_ensemble(img, lang="rus+eng")

print("ENGINE:", result["engine"])
print("SCORE :", result["score"])
print("TEXT   :")
print(result["text"])