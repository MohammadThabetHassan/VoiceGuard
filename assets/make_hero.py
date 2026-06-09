"""Generate the README hero banner (assets/hero.png). Reproducible — rerun after a
headline number changes. Design: dark gradient, VoiceGuard wordmark, tagline, 4 stats.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 640
FD = "/usr/share/fonts/truetype/dejavu"
bold = lambda s: ImageFont.truetype(f"{FD}/DejaVuSans-Bold.ttf", s)  # noqa: E731
reg = lambda s: ImageFont.truetype(f"{FD}/DejaVuSans.ttf", s)  # noqa: E731

CYAN = (77, 208, 225)
WHITE = (240, 244, 255)
GRAY = (150, 167, 196)
PURPLE = (124, 92, 255)

# ── diagonal gradient background with a soft glow ──────────────────────────────
tl = np.array([10, 14, 28], dtype=float)
br = np.array([16, 26, 50], dtype=float)
yy, xx = np.mgrid[0:H, 0:W]
t = (xx / W * 0.5 + yy / H * 0.5)
bg = (tl[None, None] * (1 - t[..., None]) + br[None, None] * t[..., None])
glow = np.exp(-(((xx - W * 0.18) ** 2 + (yy - H * 0.32) ** 2) / (2 * (W * 0.33) ** 2)))
bg += glow[..., None] * np.array([22, 30, 60])
img = Image.fromarray(np.clip(bg, 0, 255).astype("uint8"), "RGB")
d = ImageDraw.Draw(img)


def rrect(box, r, **kw):
    d.rounded_rectangle(box, radius=r, **kw)


# ── logo tile "VG" ─────────────────────────────────────────────────────────────
lx, ly, ls = 56, 70, 84
tile = Image.new("RGB", (ls, ls))
ty = np.linspace(0, 1, ls)
tile_arr = (np.array([124, 92, 255])[None, None] * (1 - ty[:, None, None])
            + np.array([155, 108, 255])[None, None] * ty[:, None, None])
tile = Image.fromarray(np.clip(np.broadcast_to(tile_arr, (ls, ls, 3)), 0, 255).astype("uint8"))
mask = Image.new("L", (ls, ls), 0)
ImageDraw.Draw(mask).rounded_rectangle([0, 0, ls - 1, ls - 1], radius=20, fill=255)
img.paste(tile, (lx, ly), mask)
vgf = bold(40)
d.text((lx + ls / 2, ly + ls / 2), "VG", font=vgf, fill=WHITE, anchor="mm")

# ── wordmark ───────────────────────────────────────────────────────────────────
wx = lx + ls + 28
wf = bold(76)
d.text((wx, ly + 6), "Voice", font=wf, fill=WHITE)
vw = d.textlength("Voice", font=wf)
d.text((wx + vw, ly + 6), "Guard", font=wf, fill=CYAN)

# ── tagline ──────────────────────────────────────────────────────────────────--
tf, tfb = reg(23), bold(23)
ty0 = 210
def line(segs, y):
    x = 56
    for txt, f, c in segs:
        d.text((x, y), txt, font=f, fill=c)
        x += d.textlength(txt, font=f)

line([("Real-time voice ", tf, GRAY), ("deepfake detection", tfb, WHITE),
      (", synthesis with voice cloning, and", tf, GRAY)], ty0)
line([("vishing defence", tfb, WHITE),
      (" — catches clones + premium TTS, explainable, watermarked, edge-ready.", tf, GRAY)], ty0 + 34)

# ── stat cards ───────────────────────────────────────────────────────────────--
stats = [
    ("2.84%", "EER · ASVSPOOF 2021 LA", CYAN),
    ("96–100%", "CLONE + PREMIUM DETECTION", WHITE),
    ("0.62 MB", "INT8 EDGE MODEL", WHITE),
    ("30 ms", "CPU INFERENCE", WHITE),
]
cw, ch, gap, x0, y0 = 270, 96, 22, 56, 430
nf, lf = bold(38), reg(14)
for i, (num, lab, col) in enumerate(stats):
    x = x0 + i * (cw + gap)
    rrect([x, y0, x + cw, y0 + ch], 14, fill=(20, 28, 50), outline=(48, 64, 104), width=1)
    d.text((x + 22, y0 + 20), num, font=nf, fill=col)
    d.text((x + 22, y0 + 66), lab, font=lf, fill=GRAY)

img.save("/srv/thabet/VoiceGuard/assets/hero.png")
print("wrote assets/hero.png", img.size)
