import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ai_fashion.analyzer import cosine_similarity, feature_vector, color_saturation_lightness, neutral_counterpart

BASE_DIR = Path(__file__).parent.parent
GALLERY_INDEX_PATH = BASE_DIR / "models" / "gallery_index.json"


Item = Dict[str, object]
Board = Dict[str, object]

# Curated fallback styles for the CLI's hint-only mode (no image, just text
# like "denim jacket"). Unrelated to the Re-PolyVore-backed ITEM_POOL below.
STYLE_CATALOG: Dict[str, Dict[str, object]] = {
    "smart_casual": {
        "bottoms": [
            {"name": "Slim Chinos", "image": "https://images.unsplash.com/photo-1582582494700-56c65d05548f?q=80&w=800&auto=format&fit=crop"},
            {"name": "Wool Trousers", "image": "https://images.unsplash.com/photo-1612423284934-285c4f6c90b1?q=80&w=800&auto=format&fit=crop"},
        ],
        "shoes": [
            {"name": "White Sneakers", "image": "https://images.unsplash.com/photo-1525966222134-fcfa99b8ae77?q=80&w=800&auto=format&fit=crop"},
            {"name": "Brown Loafers", "image": "https://images.unsplash.com/photo-1520412099551-62b6bafeb5bb?q=80&w=800&auto=format&fit=crop"},
        ],
        "palette": ["#f5f5f5", "#d9d9d9", "#a67c52"],
    },
    "minimal": {
        "bottoms": [
            {"name": "Black Trousers", "image": "https://images.unsplash.com/photo-1516259762381-22954d7d3ad2?q=80&w=800&auto=format&fit=crop"},
            {"name": "Straight Jeans", "image": "https://images.unsplash.com/photo-1512436991641-6745cdb1723f?q=80&w=800&auto=format&fit=crop"},
        ],
        "shoes": [
            {"name": "Black Derby", "image": "https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a?q=80&w=800&auto=format&fit=crop"},
            {"name": "Monochrome Sneaker", "image": "https://images.unsplash.com/photo-1519744792095-2f2205e87b6f?q=80&w=800&auto=format&fit=crop"},
        ],
        "palette": ["#111111", "#333333", "#f0f0f0"],
    },
    "streetwear": {
        "bottoms": [
            {"name": "Cargo Pants", "image": "https://images.unsplash.com/photo-1592878849125-968a9362878a?q=80&w=800&auto=format&fit=crop"},
            {"name": "Relaxed Jeans", "image": "https://images.unsplash.com/photo-1539533113208-f6df8cc8b543?q=80&w=800&auto=format&fit=crop"},
        ],
        "shoes": [
            {"name": "High-top Sneakers", "image": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?q=80&w=800&auto=format&fit=crop"},
            {"name": "Chunky Sneakers", "image": "https://images.unsplash.com/photo-1519744792095-2f2205e87b6f?q=80&w=800&auto=format&fit=crop"},
        ],
        "palette": ["#0f0f0f", "#222222", "#e0e0e0"],
    },
    "casual": {
        "bottoms": [
            {"name": "Blue Jeans", "image": "https://images.unsplash.com/photo-1512436991641-6745cdb1723f?q=80&w=800&auto=format&fit=crop"},
            {"name": "Beige Chinos", "image": "https://images.unsplash.com/photo-1582582494700-56c65d05548f?q=80&w=800&auto=format&fit=crop"},
        ],
        "shoes": [
            {"name": "Canvas Sneakers", "image": "https://images.unsplash.com/photo-1525966222134-fcfa99b8ae77?q=80&w=800&auto=format&fit=crop"},
            {"name": "Slip-ons", "image": "https://images.unsplash.com/photo-1539185441755-769473a23570?q=80&w=800&auto=format&fit=crop"},
        ],
        "palette": ["#f7f7f7", "#e6e6e6", "#3a6ea5"],
    },
    "sporty": {
        "bottoms": [
            {"name": "Track Pants", "image": "https://images.unsplash.com/photo-1592878849125-968a9362878a?q=80&w=800&auto=format&fit=crop"},
            {"name": "Athletic Shorts", "image": "https://images.unsplash.com/photo-1593032457861-3ea957f64b79?q=80&w=800&auto=format&fit=crop"},
        ],
        "shoes": [
            {"name": "Running Shoes", "image": "https://images.unsplash.com/photo-1519744792095-2f2205e87b6f?q=80&w=800&auto=format&fit=crop"},
            {"name": "Trainers", "image": "https://images.unsplash.com/photo-1525966222134-fcfa99b8ae77?q=80&w=800&auto=format&fit=crop"},
        ],
        "palette": ["#121212", "#2e7d32", "#eeeeee"],
    },
}

TYPE_TO_STYLES = {
    "top": ["smart_casual", "minimal"],
    "outwear": ["streetwear", "minimal"],
    "bottom": ["casual", "smart_casual"],
    "shoes": ["casual", "sporty"],
}


def recommend_for_top(top_type: str) -> List[Board]:
    """CLI hint-only mode (no image): pick a couple of curated style
    catalog looks for the given item type."""
    styles = TYPE_TO_STYLES.get(top_type, ["minimal", "casual"])
    boards: List[Board] = []
    for style in styles[:2]:
        catalog = STYLE_CATALOG.get(style)
        if not catalog:
            continue
        boards.append({
            "style": style,
            "bottom": catalog["bottoms"][0],
            "shoes": catalog["shoes"][0],
            "palette": catalog["palette"],
            "title": style.replace("_", " ").title(),
        })
    return boards[:2]


# Which board slot each detected/uploaded item type occupies. "outwear" gets
# its own slot — it's a layering piece, not interchangeable with "top" — and
# is intentionally left out of ESSENTIAL_SLOTS below, so it's never
# auto-filled from the pool: it only shows up on a board when the caller
# actually uploaded one.
UPLOADED_SLOT = {"top": "top", "outwear": "outwear", "bottom": "bottom", "shoes": "shoes"}

# slot name (as used on a Board) -> ITEM_POOL key
_SLOT_TO_POOL_KEY = {"top": "tops", "bottom": "bottoms", "shoes": "shoes", "outwear": "outwear"}

# Slots every board must have, always filled (from the pool if not supplied).
ESSENTIAL_SLOTS = ("top", "bottom", "shoes")

_item_vectors: Dict[str, np.ndarray] = {}


def _item_vector(item: Item) -> np.ndarray:
    key = item["image"]
    if key not in _item_vectors:
        _item_vectors[key] = np.array(item["vector"], dtype=np.float64)
    return _item_vectors[key]


def _find_best_items(target_vector: np.ndarray, candidates: List[Item], count: int = 2) -> List[Item]:
    scored = [(cosine_similarity(target_vector, _item_vector(item)), item) for item in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:count]]


# Used only if models/gallery_index.json hasn't been built yet
# (run `python build_gallery_index.py` to generate it from Re-PolyVore).
_FALLBACK_ITEM_POOL = {
    "tops": [
        {"name": "White Shirt", "image": "https://images.unsplash.com/photo-1520975698519-59c03d604b8b?q=80&w=800&auto=format&fit=crop", "vector": feature_vector("#ffffff", "solid", "smooth").tolist()},
        {"name": "Black Tee", "image": "https://images.unsplash.com/photo-1534759846116-579bd379706c?q=80&w=800&auto=format&fit=crop", "vector": feature_vector("#111111", "solid", "smooth").tolist()},
        {"name": "Grey Sweater", "image": "https://images.unsplash.com/photo-1540577162971-3d8c67a74288?q=80&w=800&auto=format&fit=crop", "vector": feature_vector("#808080", "solid", "textured").tolist()},
    ],
    "outwear": [
        {"name": "Denim Jacket", "image": "https://images.unsplash.com/photo-1551028719-00167b16eac5?q=80&w=800&auto=format&fit=crop", "vector": feature_vector("#3a6ea5", "solid", "textured").tolist()},
        {"name": "Black Blazer", "image": "https://images.unsplash.com/photo-1507679799987-c73779587ccf?q=80&w=800&auto=format&fit=crop", "vector": feature_vector("#111111", "solid", "smooth").tolist()},
        {"name": "Beige Trench", "image": "https://images.unsplash.com/photo-1544923246-77307dd270da?q=80&w=800&auto=format&fit=crop", "vector": feature_vector("#e8dcc8", "solid", "smooth").tolist()},
    ],
    "bottoms": [
        {"name": "Black Trousers", "image": "https://images.unsplash.com/photo-1516259762381-22954d7d3ad2?q=80&w=800&auto=format&fit=crop", "vector": feature_vector("#111111", "solid", "smooth").tolist()},
        {"name": "Indigo Jeans", "image": "https://images.unsplash.com/photo-1512436991641-6745cdb1723f?q=80&w=800&auto=format&fit=crop", "vector": feature_vector("#3a6ea5", "solid", "textured").tolist()},
        {"name": "Khaki Chinos", "image": "https://images.unsplash.com/photo-1582582494700-56c65d05548f?q=80&w=800&auto=format&fit=crop", "vector": feature_vector("#c3b091", "solid", "smooth").tolist()},
    ],
    "shoes": [
        {"name": "White Sneakers", "image": "https://images.unsplash.com/photo-1525966222134-fcfa99b8ae77?q=80&w=800&auto=format&fit=crop", "vector": feature_vector("#ffffff", "solid", "smooth").tolist()},
        {"name": "Black Derby", "image": "https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a?q=80&w=800&auto=format&fit=crop", "vector": feature_vector("#111111", "solid", "smooth").tolist()},
        {"name": "Chelsea Boots", "image": "https://images.unsplash.com/photo-1544441892-70fd5649411c?q=80&w=800&auto=format&fit=crop", "vector": feature_vector("#6f4e37", "solid", "smooth").tolist()},
    ],
    "accessories": [
        {"name": "Olive Tote", "category": "bag", "image": "https://images.unsplash.com/photo-1607453996400-6fc0604b88a8?q=80&w=800&auto=format&fit=crop", "vector": feature_vector("#708238").tolist()},
        {"name": "Beige Sunglasses", "category": "glasses", "image": "https://images.unsplash.com/photo-1511406367277-603a27cdb69e?q=80&w=800&auto=format&fit=crop", "vector": feature_vector("#e8dcc8").tolist()},
        {"name": "Silver Earrings", "category": "jewelry", "image": "https://images.unsplash.com/photo-1617038260897-1a2d3fd408a7?q=80&w=800&auto=format&fit=crop", "vector": feature_vector("#c0c0c0").tolist()},
        {"name": "Gold Necklace", "category": "jewelry", "image": "https://images.unsplash.com/photo-1599643478518-a784e5dc4c8f?q=80&w=800&auto=format&fit=crop", "vector": feature_vector("#d4af37").tolist()},
        {"name": "Leather Watch", "category": "watch", "image": "https://images.unsplash.com/photo-1524805444758-089113d48a6d?q=80&w=800&auto=format&fit=crop", "vector": feature_vector("#6f4e37").tolist()},
    ],
}


def _load_item_pool() -> Dict[str, List[Item]]:
    if GALLERY_INDEX_PATH.exists():
        try:
            with open(GALLERY_INDEX_PATH, "r", encoding="utf-8") as f:
                pool = json.load(f)
            if all(pool.get(slot) for slot in ("tops", "bottoms", "shoes", "accessories")):
                return pool
        except Exception:
            pass
    return _FALLBACK_ITEM_POOL


ITEM_POOL = _load_item_pool()


# Below this cosine-similarity-to-the-actual-board-pieces score, an optional
# addition (outwear especially) reads as "doesn't really go" rather than
# "completes the look" — used to gate whether it's allowed onto the board
# itself or only offered as a loose suggestion.
FIT_THRESHOLD = 0.55


def suggest_additions(category: str, context_vectors: List[np.ndarray], count: int = 8) -> List[Dict[str, object]]:
    """Rank optional add-ons (an outwear layer, or an accessory category)
    against the pieces already on the board — not just the single uploaded
    item — so "fits" reflects the whole look, not one piece of it.
    `category` is '' / 'all' for every accessory, an accessory sub-category
    (e.g. 'jewelry'), or 'outwear'."""
    if category == "outwear":
        candidates = ITEM_POOL["outwear"]
    else:
        candidates = ITEM_POOL["accessories"]
        if category and category != "all":
            candidates = [c for c in candidates if c.get("category") == category]

    results = []
    for item in candidates:
        vec = _item_vector(item)
        score = (
            sum(cosine_similarity(vec, cv) for cv in context_vectors) / len(context_vectors)
            if context_vectors else 0.0
        )
        results.append({
            "name": item["name"],
            "image": item["image"],
            "category": item.get("category", "outwear" if category == "outwear" else ""),
            "score": round(score, 3),
            "fits": score >= FIT_THRESHOLD,
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:count]


def _combo_score(target_vector: np.ndarray, items: List[Item]) -> float:
    """How well a set of items works as one outfit: each item's similarity
    to the source look, plus every pair's similarity to each other (so the
    picked pieces actually go with one another, not just independently with
    the uploaded item)."""
    vectors = [_item_vector(it) for it in items]
    to_target = sum(cosine_similarity(target_vector, v) for v in vectors) / len(vectors)
    if len(vectors) < 2:
        return to_target
    pair_scores = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            pair_scores.append(cosine_similarity(vectors[i], vectors[j]))
    coherence = sum(pair_scores) / len(pair_scores)
    return 0.6 * to_target + 0.4 * coherence


def _ranked_combos(known: Dict[str, Item], missing: List[str], target_vector: np.ndarray, pool_size: int) -> List[Dict[str, Item]]:
    candidates = {
        slot: _find_best_items(target_vector, ITEM_POOL[_SLOT_TO_POOL_KEY[slot]], count=pool_size)
        for slot in missing
    }
    combos = [
        dict(zip(missing, picks))
        for picks in itertools.product(*(candidates[slot] for slot in missing))
    ]
    combos.sort(
        key=lambda combo: _combo_score(target_vector, list(known.values()) + list(combo.values())),
        reverse=True,
    )
    return combos


def build_boards(known: Dict[str, Item], targets: List[np.ndarray], palette: List[str], count: int = 3) -> List[Board]:
    """Build up to `count` style-coherent boards. `known` holds the slots the
    caller already has real photos for (top/bottom/shoes -> Item); every
    other essential slot is filled from the pool.

    `targets` is a list of one-or-more target vectors, cycled round-robin
    across the returned boards (repeating the last one if `count` exceeds
    its length). A single target only ever ranks candidates by closeness to
    one color, which reliably clusters same-hue results across every board
    (a saturated pink top gets pink bottoms on every single board); passing
    e.g. [true_color, neutral_counterpart] guarantees at least one board
    offers a genuinely different, still-coherent pairing instead of relying
    on incidental pool diversity to happen to break the tie."""
    missing = [s for s in ESSENTIAL_SLOTS if s not in known]

    if not missing:
        board = dict(known)
        board.update(style="your_look", title="Your Look", palette=palette)
        return [board]

    # Fewer candidates per slot as more slots are missing, so the combo
    # count (candidates ** missing_slots) stays cheap regardless of how
    # many essential slots need filling.
    pool_size = max(count * 3, 6) if len(missing) <= 2 else count + 3

    boards: List[Board] = []
    used_images = {item["image"] for item in known.values()}
    for i in range(count):
        target = targets[min(i, len(targets) - 1)]
        combos = _ranked_combos(known, missing, target, pool_size)
        for combo in combos:
            if any(item["image"] in used_images for item in combo.values()):
                continue
            board = dict(known)
            board.update(combo)
            board["style"] = f"style_match_{len(boards) + 1}"
            board["title"] = f"Style Match {len(boards) + 1}"
            board["palette"] = palette
            boards.append(board)
            used_images.update(item["image"] for item in combo.values())
            break

    return boards


def _targets_and_palette(features: Dict[str, object], formality: str = "neutral") -> tuple:
    dom = features.get("dominant_colors") or []
    base_hex = dom[0] if dom else "#cccccc"
    pattern = str(features.get("pattern", "solid"))
    texture = str(features.get("texture", "smooth"))
    target_vector = feature_vector(base_hex, pattern, texture, formality)
    palette = dom[:3] if dom else ["#ffffff", "#eeeeee", "#cccccc"]

    targets = [target_vector]
    saturation, _ = color_saturation_lightness(base_hex)
    if saturation >= 0.35:
        # A bold color (bright pink, red, etc.) reliably pulls same-hue
        # matches on every board if ranked by color-closeness alone — add
        # a neutral-paired variant so at least one board offers a genuinely
        # different, still-flattering option instead of pink-on-pink ×3.
        neutral_vector = feature_vector(neutral_counterpart(base_hex), pattern, texture, formality)
        targets.append(neutral_vector)
    return targets, target_vector, palette


def recommend_from_features(item_type: str, features: Dict[str, object], count: int = 3, formality: str = "neutral") -> List[Board]:
    """Detect-one-item flow: rank the pool by cosine similarity between the
    uploaded item's handcrafted feature vector (dominant color + pattern +
    texture + formality) and each candidate, then build `count`
    style-coherent boards around it. `formality` is the caller's best guess
    for the uploaded item itself (e.g. app.py infers "casual" for a plain
    top/outwear, or a silhouette-based guess for shoes) — without it every
    upload defaults formality-neutral, which does nothing to steer bottom
    candidates toward casual vs. dressy, so a same-color dressy skirt can
    win on looks alone and then drag shoe matching toward heels via the
    combo-coherence scoring even though the uploaded top never asked for
    that. The uploaded item's own slot is left for the caller to fill in
    with the real photo (see UPLOADED_SLOT)."""
    targets, target_vector, palette = _targets_and_palette(features, formality)

    slot = UPLOADED_SLOT.get(item_type, "top")
    # placeholder so `slot` counts as "known" and is excluded from candidates;
    # the caller (app.py) overwrites this with the real uploaded image/name.
    known = {slot: {"name": None, "image": None, "vector": target_vector.tolist()}}
    return build_boards(known, targets, palette, count=count)


def recommend_style_reference(features: Dict[str, object], count: int = 3) -> List[Board]:
    """'Mixed / full outfit' mode: the uploaded photo isn't confidently any
    single top/bottom/shoes (e.g. it shows a whole outfit, or the user just
    isn't sure) — use its color/pattern/texture purely as a style reference
    and fill *all three* essential slots from the pool, instead of forcing
    it into one slot the way recommend_from_features does."""
    targets, _, palette = _targets_and_palette(features)
    return build_boards({}, targets, palette, count=count)


def recommend_from_uploads(uploaded: Dict[str, Item], count: int = 3) -> List[Board]:
    """Multi-photo flow: the caller already places their own photos into
    slots ('top'/'bottom'/'shoes' -> {"name", "image", "vector"}). Any
    essential slot not supplied is filled from the pool, built into
    style-coherent boards the same way as the single-photo flow."""
    vectors = [item["vector"] for item in uploaded.values() if "vector" in item]
    target_vector = np.mean(vectors, axis=0) if vectors else feature_vector("#cccccc")
    palette = ["#111111", "#f5f5f5", "#cccccc"]
    return build_boards(uploaded, [target_vector], palette, count=count)
