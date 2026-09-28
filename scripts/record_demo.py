"""Record the README demo GIF: upload a scanned PDF (OCR), ask a question, watch the answer stream in.

Start the app first (docker compose up, or uvicorn), then:
  python scripts/record_demo.py --url http://localhost:8000
Needs: pip install playwright imageio-ffmpeg (uses your installed Google Chrome).
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from conftest import make_scanned_pdf  # noqa: E402

SIZE = {"width": 1100, "height": 900}


def record(url: str, workdir: Path) -> Path:
    scan = workdir / "scanned-report.pdf"
    scan.write_bytes(make_scanned_pdf([
        "Quarterly Engineering Report", "Int8 quantization made CPU inference",
        "about three times faster on our servers.", "Model size dropped from 420 MB to 110 MB.",
    ]))
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        context = browser.new_context(viewport=SIZE, record_video_dir=str(workdir), record_video_size=SIZE)
        page = context.new_page()
        page.goto(url)
        page.wait_for_timeout(1200)

        page.set_input_files("#file", str(scan))
        page.wait_for_timeout(600)
        page.click("#upload")
        page.get_by_text("scanned-report.pdf").wait_for(timeout=60_000)
        page.wait_for_timeout(1200)

        page.locator("#question").press_sequentially("How much faster did int8 quantization make CPU inference?", delay=35)
        page.wait_for_timeout(400)
        page.click("#go")
        page.locator("#mode").wait_for(state="visible", timeout=120_000)
        page.wait_for_timeout(4000)

        video = Path(page.video.path())
        context.close()
        browser.close()
    return video


def to_gif(video: Path, out: Path, width: int = 900, fps: int = 8) -> None:
    filters = f"fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video), "-filter_complex",
                    f"{filters},split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5",
                    str(out)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--out", default=str(ROOT / "docs" / "demo.gif"))
    args = parser.parse_args()
    Path(args.out).parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        to_gif(record(args.url, Path(tmp)), Path(args.out))
    print(f"Saved {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
