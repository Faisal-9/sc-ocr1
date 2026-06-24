import io
import os
from typing import Optional, List

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


def _register_unicode_font() -> str:
    font_name = "Helvetica"
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/DejaVuSans.ttf",
    ]

    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont("CustomFont", font_path))
                return "CustomFont"
            except Exception:
                pass

    return font_name


def _wrap_line(text: str, font_name: str, font_size: int, max_width: float) -> List[str]:
    if not text:
        return [""]

    words = text.split()
    if not words:
        return [""]

    lines = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
        if stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


def export_to_pdf(text: str, title: Optional[str] = "OCR Extracted Text") -> bytes:
    bio = io.BytesIO()
    c = canvas.Canvas(bio, pagesize=A4)

    width, height = A4
    left = 40
    top = height - 40
    bottom = 40
    font_size = 10
    line_height = 13

    font_name = _register_unicode_font()
    c.setFont(font_name, font_size)

    y = top

    def draw_line(line: str):
        nonlocal y
        if y < bottom:
            c.showPage()
            c.setFont(font_name, font_size)
            y = top
        c.drawString(left, y, line)
        y -= line_height

    if title:
        draw_line(title)
        draw_line("")

    for paragraph in text.split("\n"):
        if paragraph.strip() == "":
            draw_line("")
            continue

        wrapped = _wrap_line(paragraph, font_name, font_size, width - 2 * left)
        for line in wrapped:
            draw_line(line)

    c.save()
    bio.seek(0)
    return bio.read()