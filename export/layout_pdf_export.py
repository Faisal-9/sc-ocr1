import base64
from typing import Any, Dict, List

import fitz


def _b64_to_bytes(b64: str) -> bytes:
    return base64.b64decode(b64.encode("utf-8"))


def export_layout_pdf(pages: List[Dict[str, Any]], title: str = "OCR Layout Report") -> bytes:
    """
    Create a reconstructed PDF where each page uses the original page image
    as the background. This preserves the original page layout visually.

    Each page dictionary should contain:
      - page_width_px
      - page_height_px
      - page_b64
      - text_items (optional, for hidden/searchable overlay)
    """
    doc = fitz.open()

    for page_data in pages:
        width = float(page_data["page_width_px"])
        height = float(page_data["page_height_px"])
        page = doc.new_page(width=width, height=height)

        # Insert the original page image as the full-page background.
        img_bytes = _b64_to_bytes(page_data["page_b64"])
        page.insert_image(page.rect, stream=img_bytes, keep_proportion=False)

        # Optional hidden OCR text overlay for search/copy.
        # If anything fails here, the visual page still stays correct.
        try:
            for item in page_data.get("text_items", []):
                txt = str(item.get("text", "")).strip()
                bbox = item.get("bbox_px")
                if not txt or not bbox:
                    continue

                x0, y0, x1, y1 = bbox
                rect = fitz.Rect(float(x0), float(y0), float(x1), float(y1))

                try:
                    page.insert_textbox(
                        rect,
                        txt,
                        fontsize=max(4.0, min(10.0, rect.height * 0.6)),
                        fontname="helv",
                        color=(0, 0, 0),
                        render_mode=3,  # invisible text, if supported
                    )
                except Exception:
                    # If render_mode is not supported in this environment,
                    # keep the page as image-only rather than failing.
                    pass
        except Exception:
            pass

    return doc.tobytes(deflate=True, garbage=4)