"""OCR for scanned PDF pages: render the page to an image (pypdfium2), then read it (RapidOCR, ONNX Runtime)."""

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

RENDER_SCALE = 2.0  # 72 dpi x 2 = 144 dpi: sharp enough for body text, still fast


@lru_cache(maxsize=1)
def _engine():
    from rapidocr import RapidOCR

    # Uses the PP-OCR models bundled with the package (no download). The library logs every step at
    # INFO level by default, so only warnings are kept.
    return RapidOCR(params={"Global.log_level": "warning"})


def ocr_image(image) -> str:
    """Text lines found in the image, top to bottom."""
    result = _engine()(image)
    return "\n".join(result.txts or ()) if result is not None else ""


def ocr_pdf_pages(data: bytes, page_numbers: list[int]) -> dict[int, str]:
    """OCR the given 0-based pages of a PDF. Returns {page number: text}."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    try:
        texts = {}
        for n in page_numbers:
            image = pdf[n].render(scale=RENDER_SCALE).to_pil()
            texts[n] = ocr_image(image)
            logger.info("OCR page %d: %d characters", n + 1, len(texts[n]))
        return texts
    finally:
        pdf.close()
