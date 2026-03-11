import io
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from PIL import Image, ImageFilter
except Exception:
    Image = None  # type: ignore
    ImageFilter = None  # type: ignore


def _open_image(path_or_url: str):
    if Image is None:
        raise RuntimeError("Pillow is required. Install with: pip install pillow")
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        req = urllib.request.Request(path_or_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    p = Path(path_or_url)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return Image.open(str(p)).convert("RGB")


def _dominant_colors(img, k: int = 5) -> List[str]:
    w, h = img.size
    s = min(96, max(32, min(w, h)))
    small = img.resize((s, s))
    colors = small.getcolors(maxcolors=s * s)
    if not colors:
        pal = small.convert("P", palette=Image.ADAPTIVE, colors=k)
        colors = pal.getcolors()
    colors = sorted(colors or [], key=lambda c: c[0], reverse=True)
    out = []
    for count, rgb in colors[:k]:
        r, g, b = rgb
        out.append(f"#{r:02x}{g:02x}{b:02x}")
    return out


def _grayscale(img):
    return img.convert("L")


def _edge_intensity(img) -> float:
    if ImageFilter is None:
        return 0.0
    edges = img.filter(ImageFilter.FIND_EDGES)
    w, h = edges.size
    small = edges.resize((64, 64))
    pix = list(small.getdata())
    total = sum(pix)
    return total / (64 * 64 * 255.0)


def _column_transitions(gray) -> float:
    g = gray.resize((64, 64))
    px = g.load()
    h = 64
    w = 64
    changes = 0
    total_pairs = 0
    for x in range(1, w):
        col_prev = sum(px[x - 1, y] for y in range(h)) / h
        col_curr = sum(px[x, y] for y in range(h)) / h
        diff = col_curr - col_prev
        if abs(diff) > 8:
            changes += 1
        total_pairs += 1
    return changes / max(1, total_pairs)


def _row_transitions(gray) -> float:
    g = gray.resize((64, 64))
    px = g.load()
    h = 64
    w = 64
    changes = 0
    total_pairs = 0
    for y in range(1, h):
        row_prev = sum(px[x, y - 1] for x in range(w)) / w
        row_curr = sum(px[x, y] for x in range(w)) / w
        diff = row_curr - row_prev
        if abs(diff) > 8:
            changes += 1
        total_pairs += 1
    return changes / max(1, total_pairs)


def _color_variance(img) -> float:
    small = img.resize((32, 32))
    px = list(small.getdata())
    if not px:
        return 0.0
    rs = [p[0] for p in px]
    gs = [p[1] for p in px]
    bs = [p[2] for p in px]
    def var(vals):
        m = sum(vals) / len(vals)
        return sum((v - m) ** 2 for v in vals) / len(vals)
    v = (var(rs) + var(gs) + var(bs)) / 3.0
    return v / (255.0 ** 2)


def analyze_image(path_or_url: str) -> Dict[str, object]:
    img = _open_image(path_or_url)
    colors = _dominant_colors(img, k=5)
    gray = _grayscale(img)
    edge = _edge_intensity(gray)
    ct = _column_transitions(gray)
    rt = _row_transitions(gray)
    varc = _color_variance(img)
    pattern = "solid"
    if ct > 0.35 and rt > 0.35:
        pattern = "checked"
    elif ct > 0.35:
        pattern = "striped_vertical"
    elif rt > 0.35:
        pattern = "striped_horizontal"
    elif varc > 0.04 and edge > 0.15:
        pattern = "print"
    texture = "smooth" if edge < 0.12 else "textured"
    return {
        "dominant_colors": colors[:3],
        "pattern": pattern,
        "texture": texture,
    }


def detect_item_type(path_or_url: str) -> str:
    img = _open_image(path_or_url)
    gray = _grayscale(img)
    w, h = gray.size
    aspect = h / max(1, w)
    # Build a coarse edge map intensity per region
    if ImageFilter is not None:
        edges = gray.filter(ImageFilter.FIND_EDGES)
    else:
        edges = gray
    small = edges.resize((64, 64))
    px = list(small.getdata())
    # Split into vertical thirds
    def region_energy(px, y0, y1):
        e = 0
        for y in range(y0, y1):
            for x in range(64):
                e += px[y * 64 + x]
        return e / ((y1 - y0) * 64 * 255.0)
    top_e = region_energy(px, 0, 21)
    mid_e = region_energy(px, 21, 43)
    bot_e = region_energy(px, 43, 64)
    # Heuristics:
    # - Shoes: wide images, strong edge energy near bottom
    # - Bottom: tall-ish, energy concentrated mid-lower
    # - Top: otherwise
    if aspect < 0.85 and bot_e > max(top_e, mid_e) * 1.15:
        return "shoes"
    if aspect > 1.15 and (mid_e + bot_e) > top_e * 1.4:
        return "bottom"
    return "top"
