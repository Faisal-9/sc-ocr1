import cv2


def upscale_document(image, factor=4):
    if image is None:
        raise ValueError("Empty image passed to upscale_document.")

    if factor <= 1:
        return image

    h, w = image.shape[:2]
    return cv2.resize(
        image,
        (w * factor, h * factor),
        interpolation=cv2.INTER_CUBIC,
    )