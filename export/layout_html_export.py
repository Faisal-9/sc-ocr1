import html
from typing import Any, Dict, List


def _table_rows_to_html(rows: List[List[Any]]) -> str:
    if not rows:
        return ""

    parts = ["<table class='table-box'>"]
    for r, row in enumerate(rows):
        parts.append("<tr>")
        for cell in row:
            tag = "th" if r == 0 else "td"
            cell_text = html.escape("" if cell is None else str(cell))
            parts.append(f"<{tag}>{cell_text}</{tag}>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def export_layout_html(pages: List[Dict[str, Any]], title: str = "OCR Layout Report") -> bytes:
    css = """
    <style>
      body { font-family: Arial, sans-serif; margin: 0; background: #f4f4f4; }
      .wrap { padding: 20px; }
      .page {
        position: relative;
        margin: 0 auto 28px auto;
        background-repeat: no-repeat;
        background-size: contain;
        background-position: top left;
        box-shadow: 0 2px 16px rgba(0,0,0,.12);
        border: 1px solid #ddd;
        background-color: white;
      }
      .overlay-text {
        position: absolute;
        color: transparent;
        white-space: pre-wrap;
        line-height: 1.15;
        user-select: text;
      }
      .section-title {
        margin: 12px 0 8px;
        font-size: 14px;
        font-weight: 700;
      }
      .asset-row {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
      }
      .asset-card {
        background: white;
        border: 1px solid #ddd;
        padding: 10px;
        margin-bottom: 10px;
      }
      .asset-img {
        max-width: 100%;
        display: block;
      }
      .table-box {
        border-collapse: collapse;
        width: 100%;
        margin-top: 8px;
        background: white;
      }
      .table-box td, .table-box th {
        border: 1px solid #777;
        padding: 4px 6px;
        font-size: 12px;
        vertical-align: top;
      }
      .meta {
        font-size: 12px;
        color: #666;
        margin-bottom: 6px;
      }
    </style>
    """

    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        css,
        "</head><body><div class='wrap'>",
        f"<h1>{html.escape(title)}</h1>",
    ]

    for i, page in enumerate(pages, start=1):
        w = int(page["page_width_px"])
        h = int(page["page_height_px"])
        page_b64 = page["page_b64"]

        parts.append(
            f"<div class='page' style='width:{w}px;height:{h}px;background-image:url(data:image/png;base64,{page_b64});'>"
        )

        for item in page.get("elements", []):
            kind = item.get("kind")
            x0, y0, x1, y1 = item["bbox_px"]
            left = max(0, x0)
            top = max(0, y0)
            width = max(1, x1 - x0)
            height = max(1, y1 - y0)

            if kind == "text":
                txt = html.escape(str(item.get("text", "")))
                parts.append(
                    f"<div class='overlay-text' style='left:{left}px;top:{top}px;width:{width}px;height:{height}px;'>"
                    f"{txt}</div>"
                )

        parts.append("</div>")

        images = page.get("image_items", [])
        tables = page.get("table_items", [])

        if images or tables:
            parts.append(f"<div class='section-title'>Page {i} extracted assets</div>")

        if images:
            parts.append("<div class='asset-row'>")
            for img in images:
                img_b64 = img.get("image_b64", "")
                if not img_b64:
                    continue
                parts.append(
                    "<div class='asset-card'>"
                    "<div class='meta'>Extracted image / figure</div>"
                    f"<img class='asset-img' src='data:image/png;base64,{img_b64}' />"
                    "</div>"
                )
            parts.append("</div>")

        if tables:
            for t in tables:
                rows = t.get("rows", [])
                parts.append("<div class='asset-card'>")
                parts.append(f"<div class='meta'>Table {t.get('table_index', '')}</div>")
                if rows:
                    parts.append(_table_rows_to_html(rows))
                img_b64 = t.get("image_b64", "")
                if img_b64:
                    parts.append("<div class='meta' style='margin-top:8px;'>Detected table region</div>")
                    parts.append(f"<img class='asset-img' src='data:image/png;base64,{img_b64}' />")
                parts.append("</div>")

    parts.append("</div></body></html>")
    return "\n".join(parts).encode("utf-8")