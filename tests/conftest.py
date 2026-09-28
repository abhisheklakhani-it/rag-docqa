import os
import zlib

import numpy as np
import pytest

# Tests never call the real LLM API.
os.environ["RAGQA_LLM"] = "none"
os.environ["RAGQA_RETRIEVER"] = "hybrid"
os.environ["RAGQA_MIN_DENSE_SCORE"] = "0"  # the floor is calibrated for the real model, not the test embedder


class HashingEmbedder:
    """Stands in for the real embedding model: hashed bag of words, no download, instant."""

    def __init__(self, *args, dim: int = 512, **kwargs):
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        from ragqa.retriever import tokenize

        v = np.zeros(self.dim, dtype="float32")
        for word in tokenize(text):
            v[zlib.crc32(word.encode()) % self.dim] += 1
        return v / (np.linalg.norm(v) or 1)

    def embed_documents(self, texts):
        return np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.dim), "float32")

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture(autouse=True)
def no_model_download(monkeypatch):
    """Every test that builds the app gets the hashing embedder instead of downloading a model."""
    import ragqa.retriever

    monkeypatch.setattr(ragqa.retriever, "FastEmbedder", HashingEmbedder)


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal valid PDF with one line of text per page (no extra dependencies)."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "", "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {len(objs)} 0 R "
            "/Resources << /Font << /F1 3 0 R >> >> >>"
        )
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"

    out, offsets = b"%PDF-1.4\n", []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


@pytest.fixture
def pdf_bytes() -> bytes:
    return make_pdf([
        "ONNX is an open format for machine learning models.",
        "Int8 quantization makes CPU inference about 3x faster.",
    ])


def make_scanned_pdf(lines: list[str]) -> bytes:
    """A PDF page that is only an image of text (like a scan): no text layer, slightly rotated and blurred."""
    import io

    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    image = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=36)
    for i, line in enumerate(lines):
        draw.text((100, 150 + i * 70), line, fill="black", font=font)
    image = image.rotate(0.7, fillcolor="white").filter(ImageFilter.GaussianBlur(0.8))
    out = io.BytesIO()
    image.save(out, "PDF", resolution=150)
    return out.getvalue()


@pytest.fixture(scope="session")
def scanned_pdf() -> bytes:
    return make_scanned_pdf(["Quarterly Report 2026", "Int8 quantization made CPU inference",
                             "about three times faster on our servers."])
