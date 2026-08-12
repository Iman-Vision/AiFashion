from typing import Dict, List, Tuple

import numpy as np

from ai_fashion.analyzer import color_name_to_hex, cosine_similarity, feature_vector


Item = Dict[str, str]
Board = Dict[str, object]


STYLE_CATALOG: Dict[str, Dict[str, List[Item]]] = {
    "smart_casual": {
        "bottoms": [
            {"name": "Slim Chinos", "color": "khaki", "image": "https://images.unsplash.com/photo-1582582494700-56c65d05548f?q=80&w=800&auto=format&fit=crop"},
            {"name": "Wool Trousers", "color": "grey", "image": "https://images.unsplash.com/photo-1612423284934-285c4f6c90b1?q=80&w=800&auto=format&fit=crop"},
        ],
        "shoes": [
            {"name": "White Sneakers", "color": "white", "image": "https://images.unsplash.com/photo-1525966222134-fcfa99b8ae77?q=80&w=800&auto=format&fit=crop"},
            {"name": "Brown Loafers", "color": "brown", "image": "https://images.unsplash.com/photo-1520412099551-62b6bafeb5bb?q=80&w=800&auto=format&fit=crop"},
        ],
        "palette": ["#f5f5f5", "#d9d9d9", "#a67c52"],
    },
    "minimal": {
        "bottoms": [
            {"name": "Black Trousers", "color": "black", "image": "https://images.unsplash.com/photo-1516259762381-22954d7d3ad2?q=80&w=800&auto=format&fit=crop"},
            {"name": "Straight Jeans", "color": "indigo", "image": "https://images.unsplash.com/photo-1512436991641-6745cdb1723f?q=80&w=800&auto=format&fit=crop"},
        ],
        "shoes": [
            {"name": "Black Derby", "color": "black", "image": "https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a?q=80&w=800&auto=format&fit=crop"},
            {"name": "Monochrome Sneaker", "color": "black", "image": "https://images.unsplash.com/photo-1519744792095-2f2205e87b6f?q=80&w=800&auto=format&fit=crop"},
        ],
        "palette": ["#111111", "#333333", "#f0f0f0"],
    },
    "streetwear": {
        "bottoms": [
            {"name": "Cargo Pants", "color": "olive", "image": "https://images.unsplash.com/photo-1592878849125-968a9362878a?q=80&w=800&auto=format&fit=crop"},
            {"name": "Relaxed Jeans", "color": "blue", "image": "https://images.unsplash.com/photo-1539533113208-f6df8cc8b543?q=80&w=800&auto=format&fit=crop"},
        ],
        "shoes": [
            {"name": "High-top Sneakers", "color": "multi", "image": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?q=80&w=800&auto=format&fit=crop"},
            {"name": "Chunky Sneakers", "color": "white", "image": "https://images.unsplash.com/photo-1519744792095-2f2205e87b6f?q=80&w=800&auto=format&fit=crop"},
        ],
        "palette": ["#0f0f0f", "#222222", "#e0e0e0"],
    },
    "casual": {
        "bottoms": [
            {"name": "Blue Jeans", "color": "blue", "image": "https://images.unsplash.com/photo-1512436991641-6745cdb1723f?q=80&w=800&auto=format&fit=crop"},
            {"name": "Beige Chinos", "color": "beige", "image": "https://images.unsplash.com/photo-1582582494700-56c65d05548f?q=80&w=800&auto=format&fit=crop"},
        ],
        "shoes": [
            {"name": "Canvas Sneakers", "color": "white", "image": "https://images.unsplash.com/photo-1525966222134-fcfa99b8ae77?q=80&w=800&auto=format&fit=crop"},
            {"name": "Slip-ons", "color": "grey", "image": "https://images.unsplash.com/photo-1539185441755-769473a23570?q=80&w=800&auto=format&fit=crop"},
        ],
        "palette": ["#f7f7f7", "#e6e6e6", "#3a6ea5"],
    },
    "cozy": {
        "bottoms": [
            {"name": "Corduroy Pants", "color": "brown", "image": "https://images.unsplash.com/photo-1603570419989-4269a9300d25?q=80&w=800&auto=format&fit=crop"},
            {"name": "Wool Skirt", "color": "camel", "image": "https://images.unsplash.com/photo-1603570420084-8908033b4601?q=80&w=800&auto=format&fit=crop"},
        ],
        "shoes": [
            {"name": "Chelsea Boots", "color": "brown", "image": "https://images.unsplash.com/photo-1544441892-70fd5649411c?q=80&w=800&auto=format&fit=crop"},
            {"name": "Suede Loafers", "color": "tan", "image": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?q=80&w=800&auto=format&fit=crop"},
        ],
        "palette": ["#a67c52", "#e6d5c3", "#3b3b3b"],
    },
    "sporty": {
        "bottoms": [
            {"name": "Track Pants", "color": "black", "image": "https://images.unsplash.com/photo-1592878849125-968a9362878a?q=80&w=800&auto=format&fit=crop"},
            {"name": "Athletic Shorts", "color": "grey", "image": "https://images.unsplash.com/photo-1593032457861-3ea957f64b79?q=80&w=800&auto=format&fit=crop"},
        ],
        "shoes": [
            {"name": "Running Shoes", "color": "neon", "image": "https://images.unsplash.com/photo-1519744792095-2f2205e87b6f?q=80&w=800&auto=format&fit=crop"},
            {"name": "Trainers", "color": "white", "image": "https://images.unsplash.com/photo-1525966222134-fcfa99b8ae77?q=80&w=800&auto=format&fit=crop"},
        ],
        "palette": ["#121212", "#2e7d32", "#eeeeee"],
    },
    "chic": {
        "bottoms": [
            {"name": "Pleated Skirt", "color": "black", "image": "https://images.unsplash.com/photo-1512436991641-6745cdb1723f?q=80&w=800&auto=format&fit=crop"},
            {"name": "Tailored Trousers", "color": "cream", "image": "https://images.unsplash.com/photo-1612423284934-285c4f6c90b1?q=80&w=800&auto=format&fit=crop"},
        ],
        "shoes": [
            {"name": "Heeled Boots", "color": "black", "image": "https://images.unsplash.com/photo-1544441892-70fd5649411c?q=80&w=800&auto=format&fit=crop"},
            {"name": "Pointed Flats", "color": "nude", "image": "https://images.unsplash.com/photo-1539185441755-769473a23570?q=80&w=800&auto=format&fit=crop"},
        ],
        "palette": ["#0f0f0f", "#f0f0f0", "#b38b6d"],
    },
    "preppy": {
        "bottoms": [
            {"name": "Pleated Chinos", "color": "navy", "image": "https://images.unsplash.com/photo-1582582494700-56c65d05548f?q=80&w=800&auto=format&fit=crop"},
            {"name": "Light Jeans", "color": "lightblue", "image": "https://images.unsplash.com/photo-1512436991641-6745cdb1723f?q=80&w=800&auto=format&fit=crop"},
        ],
        "shoes": [
            {"name": "Boat Shoes", "color": "tan", "image": "https://images.unsplash.com/photo-1520412099551-62b6bafeb5bb?q=80&w=800&auto=format&fit=crop"},
            {"name": "Clean Sneakers", "color": "white", "image": "https://images.unsplash.com/photo-1525966222134-fcfa99b8ae77?q=80&w=800&auto=format&fit=crop"},
        ],
        "palette": ["#1d3557", "#f1faee", "#a8dadc"],
    },
}

# Map DeepFashion2 item types to styles
TYPE_TO_STYLES = {
    "top": ["smart_casual", "minimal"],
    "outwear": ["streetwear", "minimal"],
    "bottom": ["casual", "smart_casual"],
    "dress": ["chic", "minimal"],
    "shoes": ["casual", "sporty"],
}

_item_vectors: Dict[str, np.ndarray] = {}


def _item_vector(item: Item) -> np.ndarray:
    key = item["image"]
    if key not in _item_vectors:
        hex_color = color_name_to_hex(item.get("color", ""))
        _item_vectors[key] = feature_vector(
            hex_color,
            item.get("pattern", "solid"),
            item.get("texture", "smooth"),
        )
    return _item_vectors[key]


def _find_best_items(target_vector: np.ndarray, candidates: List[Item], count: int = 2) -> List[Item]:
    scored = [(cosine_similarity(target_vector, _item_vector(item)), item) for item in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:count]]


ITEM_POOL = {
    "tops": [
        {"name": "White Shirt", "color": "white", "image": "https://images.unsplash.com/photo-1520975698519-59c03d604b8b?q=80&w=800&auto=format&fit=crop", "pattern": "solid", "texture": "smooth"},
        {"name": "Black Tee", "color": "black", "image": "https://images.unsplash.com/photo-1534759846116-579bd379706c?q=80&w=800&auto=format&fit=crop", "pattern": "solid", "texture": "smooth"},
        {"name": "Grey Sweater", "color": "grey", "image": "https://images.unsplash.com/photo-1540577162971-3d8c67a74288?q=80&w=800&auto=format&fit=crop", "pattern": "solid", "texture": "textured"},
    ],
    "outwear": [
        {"name": "Denim Jacket", "color": "blue", "image": "https://images.unsplash.com/photo-1551028719-00167b16eac5?q=80&w=800&auto=format&fit=crop", "pattern": "solid", "texture": "textured"},
        {"name": "Black Blazer", "color": "black", "image": "https://images.unsplash.com/photo-1507679799987-c73779587ccf?q=80&w=800&auto=format&fit=crop", "pattern": "solid", "texture": "smooth"},
        {"name": "Beige Trench", "color": "beige", "image": "https://images.unsplash.com/photo-1544923246-77307dd270da?q=80&w=800&auto=format&fit=crop", "pattern": "solid", "texture": "smooth"},
    ],
    "bottoms": [
        {"name": "Black Trousers", "color": "black", "image": "https://images.unsplash.com/photo-1516259762381-22954d7d3ad2?q=80&w=800&auto=format&fit=crop", "pattern": "solid", "texture": "smooth"},
        {"name": "Indigo Jeans", "color": "blue", "image": "https://images.unsplash.com/photo-1512436991641-6745cdb1723f?q=80&w=800&auto=format&fit=crop", "pattern": "solid", "texture": "textured"},
        {"name": "Khaki Chinos", "color": "khaki", "image": "https://images.unsplash.com/photo-1582582494700-56c65d05548f?q=80&w=800&auto=format&fit=crop", "pattern": "solid", "texture": "smooth"},
    ],
    "dress": [
        {"name": "Little Black Dress", "color": "black", "image": "https://images.unsplash.com/photo-1595777457583-95e059d581b8?q=80&w=800&auto=format&fit=crop", "pattern": "solid", "texture": "smooth"},
        {"name": "Floral Dress", "color": "multi", "image": "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?q=80&w=800&auto=format&fit=crop", "pattern": "print", "texture": "smooth"},
    ],
    "shoes": [
        {"name": "White Sneakers", "color": "white", "image": "https://images.unsplash.com/photo-1525966222134-fcfa99b8ae77?q=80&w=800&auto=format&fit=crop", "pattern": "solid", "texture": "smooth"},
        {"name": "Black Derby", "color": "black", "image": "https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a?q=80&w=800&auto=format&fit=crop", "pattern": "solid", "texture": "smooth"},
        {"name": "Chelsea Boots", "color": "brown", "image": "https://images.unsplash.com/photo-1544441892-70fd5649411c?q=80&w=800&auto=format&fit=crop", "pattern": "solid", "texture": "smooth"},
    ],
    "accessories": [
        {"name": "Olive Tote", "color": "olive", "category": "bag", "image": "https://images.unsplash.com/photo-1607453996400-6fc0604b88a8?q=80&w=800&auto=format&fit=crop"},
        {"name": "Beige Sunglasses", "color": "beige", "category": "glasses", "image": "https://images.unsplash.com/photo-1511406367277-603a27cdb69e?q=80&w=800&auto=format&fit=crop"},
        {"name": "Silver Earrings", "color": "silver", "category": "jewelry", "image": "https://images.unsplash.com/photo-1617038260897-1a2d3fd408a7?q=80&w=800&auto=format&fit=crop"},
        {"name": "Neutral Nail Polish", "color": "nude", "category": "beauty", "image": "https://images.unsplash.com/photo-1541099649105-f69ad21f3246?q=80&w=800&auto=format&fit=crop"},
        {"name": "Gold Necklace", "color": "gold", "category": "jewelry", "image": "https://images.unsplash.com/photo-1599643478518-a784e5dc4c8f?q=80&w=800&auto=format&fit=crop"},
        {"name": "Leather Watch", "color": "brown", "category": "watch", "image": "https://images.unsplash.com/photo-1524805444758-089113d48a6d?q=80&w=800&auto=format&fit=crop"},
    ],
}


def recommend_for_top(top_type: str) -> List[Board]:
    styles = TYPE_TO_STYLES.get(top_type, ["minimal", "casual"])
    boards: List[Board] = []
    for style in styles[:2]:
        catalog = STYLE_CATALOG.get(style)
        if not catalog:
            continue
        bottom = catalog["bottoms"][0]
        shoes = catalog["shoes"][0]
        boards.append(
            {
                "style": style,
                "bottom": bottom,
                "shoes": shoes,
                "palette": catalog["palette"],
                "title": f"{style.replace('_', ' ').title()}",
            }
        )
    return boards[:2]


def _accessory_pairs(target_vector: np.ndarray) -> Tuple[List[Item], List[Item]]:
    """Rank all accessories by cosine similarity to the uploaded item and split
    the top four into two non-overlapping pairs, so jewelry/bags/glasses/beauty
    all get a chance to appear across the two boards."""
    ranked = _find_best_items(target_vector, ITEM_POOL["accessories"], count=4)
    return ranked[0:2], ranked[2:4]


def recommend_from_features(item_type: str, features: Dict[str, object]) -> List[Board]:
    """Rank the item pool by cosine similarity between the uploaded item's
    handcrafted feature vector (dominant color + pattern + texture) and each
    candidate's feature vector, then pair the two best-matching options."""
    dom = features.get("dominant_colors") or []
    base_hex = dom[0] if dom else "#cccccc"
    pattern = str(features.get("pattern", "solid"))
    texture = str(features.get("texture", "smooth"))
    target_vector = feature_vector(base_hex, pattern, texture)
    palette = (dom[:3] if dom else ["#ffffff", "#eeeeee", "#cccccc"])
    acc_pair_1, acc_pair_2 = _accessory_pairs(target_vector)

    boards: List[Board] = []

    if item_type in ("top", "outwear"):
        bottoms = _find_best_items(target_vector, ITEM_POOL["bottoms"], count=2)
        shoes = _find_best_items(target_vector, ITEM_POOL["shoes"], count=2)
        for i in range(min(len(bottoms), len(shoes))):
            boards.append({
                "style": f"similar_{i + 1}",
                "bottom": bottoms[i],
                "shoes": shoes[i],
                "palette": palette,
                "title": f"Style Match {i + 1}",
                "accessories": acc_pair_1 if i == 0 else acc_pair_2,
            })
    elif item_type == "bottom":
        tops = _find_best_items(target_vector, ITEM_POOL["tops"], count=2)
        shoes = _find_best_items(target_vector, ITEM_POOL["shoes"], count=2)
        for i in range(min(len(tops), len(shoes))):
            boards.append({
                "style": f"similar_{i + 1}",
                "top": tops[i],
                "shoes": shoes[i],
                "palette": palette,
                "title": f"Style Match {i + 1}",
                "accessories": acc_pair_1 if i == 0 else acc_pair_2,
            })
    elif item_type == "dress":
        outwear = _find_best_items(target_vector, ITEM_POOL["outwear"], count=2)
        shoes = _find_best_items(target_vector, ITEM_POOL["shoes"], count=2)
        for i in range(min(len(outwear), len(shoes))):
            boards.append({
                "style": f"similar_{i + 1}",
                "top": outwear[i],
                "shoes": shoes[i],
                "palette": palette,
                "title": f"Style Match {i + 1}",
                "accessories": acc_pair_1 if i == 0 else acc_pair_2,
            })
    elif item_type == "shoes":
        tops = _find_best_items(target_vector, ITEM_POOL["tops"], count=2)
        bottoms = _find_best_items(target_vector, ITEM_POOL["bottoms"], count=2)
        for i in range(min(len(tops), len(bottoms))):
            boards.append({
                "style": f"similar_{i + 1}",
                "top": tops[i],
                "bottom": bottoms[i],
                "palette": palette,
                "title": f"Style Match {i + 1}",
                "accessories": acc_pair_1 if i == 0 else acc_pair_2,
            })

    return boards[:2]
