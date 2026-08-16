import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ai_fashion.analyzer import cosine_similarity, feature_vector

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


# Which board slot each detected/uploaded item type occupies. The app only
# ever deals in these three essential slots — no "dress" category.
UPLOADED_SLOT = {"top": "top", "outwear": "top", "bottom": "bottom", "shoes": "shoes"}

# slot name (as used on a Board) -> ITEM_POOL key
_SLOT_TO_POOL_KEY = {"top": "tops", "bottom": "bottoms", "shoes": "shoes"}

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


def accessory_options(target_vector: np.ndarray, count: int = 6) -> List[Item]:
    """Rank accessories by cosine similarity to a look's target vector.
    Accessories are opt-in — this is only called when the user explicitly
    asks to see what jewelry/bags/watches would go with a board."""
    return _find_best_items(target_vector, ITEM_POOL["accessories"], count=count)


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


def build_boards(known: Dict[str, Item], target_vector: np.ndarray, palette: List[str], count: int = 3) -> List[Board]:
    """Build up to `count` style-coherent boards. `known` holds the slots the
    caller already has real photos for (top/bottom/shoes -> Item); every
    other essential slot is filled from the pool. Boards are ranked so the
    filled-in pieces work with the source look AND with each other, not just
    independently ranked lists zipped by index."""
    missing = [s for s in ESSENTIAL_SLOTS if s not in known]

    if not missing:
        board = dict(known)
        board.update(style="your_look", title="Your Look", palette=palette)
        return [board]

    candidate_pool = max(count * 3, 6)
    candidates = {
        slot: _find_best_items(target_vector, ITEM_POOL[_SLOT_TO_POOL_KEY[slot]], count=candidate_pool)
        for slot in missing
    }

    if len(missing) == 1:
        slot = missing[0]
        combos = [{slot: c} for c in candidates[slot]]
        combos.sort(key=lambda combo: _combo_score(target_vector, [known.get(s) or combo[slot] for s in ESSENTIAL_SLOTS]), reverse=True)
    else:
        combos = []
        for a in candidates[missing[0]]:
            for b in candidates[missing[1]]:
                combos.append({missing[0]: a, missing[1]: b})
        combos.sort(
            key=lambda combo: _combo_score(target_vector, list(known.values()) + list(combo.values())),
            reverse=True,
        )

    boards: List[Board] = []
    used_images = {item["image"] for item in known.values()}
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
        if len(boards) >= count:
            break

    return boards


def recommend_from_features(item_type: str, features: Dict[str, object], count: int = 3) -> List[Board]:
    """Detect-one-item flow: rank the pool by cosine similarity between the
    uploaded item's handcrafted feature vector (dominant color + pattern +
    texture) and each candidate, then build `count` style-coherent boards
    around it. The uploaded item's own slot is left for the caller to fill
    in with the real photo (see UPLOADED_SLOT)."""
    dom = features.get("dominant_colors") or []
    base_hex = dom[0] if dom else "#cccccc"
    pattern = str(features.get("pattern", "solid"))
    texture = str(features.get("texture", "smooth"))
    target_vector = feature_vector(base_hex, pattern, texture)
    palette = dom[:3] if dom else ["#ffffff", "#eeeeee", "#cccccc"]

    slot = UPLOADED_SLOT.get(item_type, "top")
    # placeholder so `slot` counts as "known" and is excluded from candidates;
    # the caller (app.py) overwrites this with the real uploaded image/name.
    known = {slot: {"name": None, "image": None, "vector": target_vector.tolist()}}
    return build_boards(known, target_vector, palette, count=count)


def recommend_from_uploads(uploaded: Dict[str, Item], count: int = 3) -> List[Board]:
    """Multi-photo flow: the caller already places their own photos into
    slots ('top'/'bottom'/'shoes' -> {"name", "image", "vector"}). Any
    essential slot not supplied is filled from the pool, built into
    style-coherent boards the same way as the single-photo flow."""
    vectors = [item["vector"] for item in uploaded.values() if "vector" in item]
    target_vector = np.mean(vectors, axis=0) if vectors else feature_vector("#cccccc")
    palette = ["#111111", "#f5f5f5", "#cccccc"]
    return build_boards(uploaded, target_vector, palette, count=count)
