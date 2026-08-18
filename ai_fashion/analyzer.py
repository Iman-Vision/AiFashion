import io
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from PIL import Image, ImageFilter
except Exception:
    Image = None  # type: ignore
    ImageFilter = None  # type: ignore

PATTERNS = ["solid", "checked", "striped_vertical", "striped_horizontal", "print"]
TEXTURES = ["smooth", "textured"]

def color_saturation_lightness(hex_color: str) -> Tuple[float, float]:
    """HSV saturation + lightness (0-1 each) of a hex color."""
    r, g, b = _hex_to_rgb01(hex_color)
    hi, lo = max(r, g, b), min(r, g, b)
    lightness = (hi + lo) / 2
    sat = 0.0 if hi == lo else (hi - lo) / (1 - abs(2 * lightness - 1))
    return sat, lightness


def neutral_counterpart(hex_color: str) -> str:
    """A versatile neutral that contrasts in lightness with a bold color —
    e.g. a bright pink/red top pairs well with black or charcoal, a dark
    saturated color with cream/white. Only meaningful for saturated colors;
    callers should check color_saturation_lightness() first."""
    _, lightness = color_saturation_lightness(hex_color)
    return "#f5f0e8" if lightness < 0.5 else "#1f2023"


def _hex_to_rgb01(hex_color: str) -> Tuple[float, float, float]:
    c = hex_color.lstrip("#")
    if len(c) != 6:
        return (0.6, 0.6, 0.6)
    r = int(c[0:2], 16) / 255.0
    g = int(c[2:4], 16) / 255.0
    b = int(c[4:6], 16) / 255.0
    return (r, g, b)


# Color is the dominant visual signal for matching; pattern/texture should
# only break ties between similarly-colored items. Without this weighting
# the one-hot pattern/texture bits (norm ~1.4) outweigh RGB (norm <1.7 but
# usually much smaller for muted colors), and since most real photos land
# on "solid"/"smooth" that shared match swamps cosine similarity — results
# barely move with color at all.
_COLOR_WEIGHT = 4.0
_PATTERN_WEIGHT = 0.5
_TEXTURE_WEIGHT = 0.5
# Deliberately strong: this is what stops "sneakers with a skirt, heels
# with jeans" — casual/dressy agreement between an outfit's pieces matters
# more to whether it reads as a coherent look than color alone does.
_FORMALITY_WEIGHT = 2.5


def formality_vec(formality: str) -> List[float]:
    if formality == "casual":
        return [1.0, 0.0]
    if formality == "dressy":
        return [0.0, 1.0]
    return [0.5, 0.5]  # neutral / unknown — doesn't pull matching either way


def feature_vector(color_hex: str, pattern: str = "solid", texture: str = "smooth", formality: str = "neutral") -> np.ndarray:
    """Handcrafted feature vector: weighted RGB + one-hot pattern + one-hot
    texture + casual/dressy formality."""
    r, g, b = _hex_to_rgb01(color_hex)
    pattern_vec = [1.0 if pattern == p else 0.0 for p in PATTERNS]
    texture_vec = [1.0 if texture == t else 0.0 for t in TEXTURES]
    return np.array(
        [r * _COLOR_WEIGHT, g * _COLOR_WEIGHT, b * _COLOR_WEIGHT]
        + [v * _PATTERN_WEIGHT for v in pattern_vec]
        + [v * _TEXTURE_WEIGHT for v in texture_vec]
        + [v * _FORMALITY_WEIGHT for v in formality_vec(formality)],
        dtype=np.float64,
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return -1.0
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return -1.0
    return float(np.dot(a, b) / (a_norm * b_norm))


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


def _is_studio_background(rgb) -> bool:
    """Near-white or near-flat-gray — the seamless backdrop most product
    photos are shot on, not the garment itself."""
    r, g, b = rgb
    if max(r, g, b) - min(r, g, b) > 12:
        return False  # has actual hue, not a neutral backdrop
    return r > 225 and g > 225 and b > 225


def _dominant_colors(img, k: int = 5) -> List[str]:
    w, h = img.size
    s = min(96, max(32, min(w, h)))
    small = img.resize((s, s))
    colors = small.getcolors(maxcolors=s * s)
    if not colors:
        pal = small.convert("P", palette=Image.ADAPTIVE, colors=k)
        colors = pal.getcolors()
    colors = sorted(colors or [], key=lambda c: c[0], reverse=True)

    # Product photos are almost always shot on a white/light-grey backdrop,
    # which getcolors() ranks first by sheer pixel count. Without skipping
    # it, nearly every item's "dominant color" comes out white regardless
    # of the actual garment, which collapses all cosine-similarity matching
    # toward the same handful of pale items. Prefer the most frequent color
    # that isn't the backdrop; only fall back to it if nothing else exists
    # (i.e. the garment genuinely is white).
    foreground = [c for c in colors if not _is_studio_background(c[1])]
    ranked = foreground if foreground else colors

    out = []
    for count, rgb in ranked[:k]:
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


def estimate_shoe_formality(img) -> str:
    """Coarse heel-vs-flat heuristic from silhouette alone (Re-PolyVore's
    shoes folder has no sneaker/heel labels to draw on). Heels are shot
    tall and narrow with weight concentrated under a thin heel post;
    sneakers/flats are wider and flat-soled, spreading mass across the
    whole bottom edge. Not precise, but far better than treating every
    shoe as formality-neutral."""
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES) if ImageFilter is not None else gray
    s = 64
    small = edges.resize((s, s))
    px = small.load()

    bottom_y0 = int(s * 0.85)
    cx0, cx1 = int(s * 0.4), int(s * 0.6)

    def region_avg(x0, x1, y0, y1):
        total = sum(px[x, y] for y in range(y0, y1) for x in range(x0, x1))
        area = max(1, (x1 - x0) * (y1 - y0))
        return total / area

    center_avg = region_avg(cx0, cx1, bottom_y0, s)
    left_avg = region_avg(0, cx0, bottom_y0, s)
    right_avg = region_avg(cx1, s, bottom_y0, s)
    outer_avg = (left_avg + right_avg) / 2.0

    w, h = img.size
    aspect = h / max(1, w)

    narrow_heel_signal = center_avg > outer_avg * 1.25
    tall_signal = aspect > 1.15
    return "dressy" if (narrow_heel_signal or tall_signal) else "casual"


def infer_formality(category: str, img=None) -> str:
    """Best-effort casual/dressy tag used to build the gallery index.
    'category' is the Re-PolyVore source folder name."""
    if category == "pants":
        return "casual"
    if category == "skirt":
        return "dressy"
    if category == "shoes" and img is not None:
        return estimate_shoe_formality(img)
    return "neutral"


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
