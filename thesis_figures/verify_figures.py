"""Render PDFs and check hashes, page bounds and nonblank output; requires PyMuPDF and Pillow."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "generation_manifest.json").read_text(encoding="utf-8"))
    output = ROOT / "qa"
    output.mkdir(exist_ok=True)
    previews = []
    report = []
    for stem, digests in manifest["figures"].items():
        for extension, expected in digests.items():
            assert hashlib.sha256((ROOT / f"{stem}.{extension}").read_bytes()).hexdigest() == expected
        with pymupdf.open(ROOT / f"{stem}.pdf") as doc:
            assert len(doc) == 1, stem
            page = doc[0]
            text = page.get_text()
            assert len(text.strip()) > 50, f"Missing searchable text: {stem}"
            outside = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        rect = pymupdf.Rect(span["bbox"])
                        if not (page.rect + (-1, -1, 1, 1)).contains(rect):
                            outside.append(span["text"])
            assert not outside, f"Text outside page: {stem}: {outside}"
            pix = page.get_pixmap(matrix=pymupdf.Matrix(1.6, 1.6), alpha=False)
            pix.save(output / f"{stem}.png")
            im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pixels = np.asarray(im)
            nonwhite = float(np.mean(np.any(pixels < 240, axis=2)))
            assert .01 < nonwhite < .8, f"Blank or flooded page: {stem}"
            report.append({"stem": stem, "page_count": 1, "searchable_text": True,
                           "out_of_page_text": outside, "nonwhite_fraction": nonwhite})
            im.thumbnail((670, 460))
            previews.append((stem, im.copy()))
    sheet = Image.new("RGB", (1400, 7 * 505), "#ECEFF1")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("DejaVuSans.ttf", 17) if Path("DejaVuSans.ttf").exists() else ImageFont.load_default(size=18)
    for index, (stem, im) in enumerate(previews):
        x, y = 15 + (index % 2) * 695, 15 + (index // 2) * 505
        draw.text((x, y), stem, fill="#303840", font=font)
        sheet.paste(im, (x + (670-im.width)//2, y+32))
    sheet.save(ROOT / "00_contact_sheet.png")
    (output / "validation_report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(f"PASS: {len(report)} PDFs rendered; hashes, page bounds and nonblank checks passed.")


if __name__ == "__main__":
    main()
