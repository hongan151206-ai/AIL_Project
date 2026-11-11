"""Render Fig. 7 by screenshotting the real operator tool (app/index.html).
Requires: playwright (pip install playwright && playwright install chromium) and Pillow.
Run from code/:  python shoot_app.py
"""
import pathlib
from playwright.sync_api import sync_playwright
from PIL import Image

APP = pathlib.Path(__file__).resolve().parents[1] / "app" / "index.html"
OUT_PNG = pathlib.Path(__file__).resolve().parents[1] / "figures" / "fig_tool.png"
OUT_PDF = pathlib.Path(__file__).resolve().parents[1] / "figures" / "fig_tool.pdf"
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(device_scale_factor=2, viewport={"width": 1120, "height": 700})
    pg.goto(APP.as_uri())
    pg.wait_for_timeout(400)
    pg.query_selector(".app").screenshot(path=str(OUT_PNG))
    b.close()

Image.open(OUT_PNG).convert("RGB").save(str(OUT_PDF), "PDF", resolution=300.0)
print("wrote", OUT_PDF)
