from typing import Dict, List


Item = Dict[str, str]
Board = Dict[str, object]


TOP_TO_STYLES: Dict[str, List[str]] = {
    "shirt": ["smart_casual", "minimal"],
    "tshirt": ["streetwear", "casual"],
    "sweater": ["cozy", "smart_casual"],
    "hoodie": ["streetwear", "sporty"],
    "blouse": ["chic", "smart_casual"],
    "polo": ["preppy", "casual"],
    "jacket": ["streetwear", "minimal"],
    "coat": ["minimal", "chic"],
}


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


def recommend_for_top(top_type: str) -> List[Board]:
    styles = TOP_TO_STYLES.get(top_type, ["minimal", "casual"])
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


NEUTRALS = ["black", "white", "grey", "khaki", "beige", "navy"]

ITEM_POOL = {
    "tops": [
        {"name": "White Shirt", "color": "white", "image": "https://images.unsplash.com/photo-1520975698519-59c03d604b8b?q=80&w=800&auto=format&fit=crop", "pattern": "solid", "texture": "smooth"},
        {"name": "Black Tee", "color": "black", "image": "https://images.unsplash.com/photo-1534759846116-579bd379706c?q=80&w=800&auto=format&fit=crop", "pattern": "solid", "texture": "smooth"},
        {"name": "Grey Sweater", "color": "grey", "image": "https://images.unsplash.com/photo-1540577162971-3d8c67a74288?q=80&w=800&auto=format&fit=crop", "pattern": "solid", "texture": "textured"},
    ],
    "bottoms": [
        {"name": "Black Trousers", "color": "black", "image": "https://images.unsplash.com/photo-1516259762381-22954d7d3ad2?q=80&w=800&auto=format&fit=crop", "pattern": "solid", "texture": "smooth"},
        {"name": "Indigo Jeans", "color": "blue", "image": "https://images.unsplash.com/photo-1512436991641-6745cdb1723f?q=80&w=800&auto=format&fit=crop", "pattern": "solid", "texture": "textured"},
        {"name": "Khaki Chinos", "color": "khaki", "image": "https://images.unsplash.com/photo-1582582494700-56c65d05548f?q=80&w=800&auto=format&fit=crop", "pattern": "solid", "texture": "smooth"},
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
    ],
}


def _is_neutral_hex(hex_color: str) -> bool:
    c = hex_color.lower()
    if c in ["#000000", "#111111", "#222222", "#333333", "#444444"]:
        return True
    if c in ["#ffffff", "#f5f5f5", "#eeeeee", "#e0e0e0", "#d9d9d9"]:
        return True
    return False


def _choose_bottom_and_shoes_from_features(features: Dict[str, object]) -> List[Board]:
    dom = [x for x in features.get("dominant_colors", [])][:1]
    base_hex = dom[0] if dom else "#cccccc"
    pattern = str(features.get("pattern", "solid"))
    texture = str(features.get("texture", "smooth"))

    bottoms = ITEM_POOL["bottoms"]
    shoes = ITEM_POOL["shoes"]

    prefer_neutral = _is_neutral_hex(base_hex) is False or "striped" in pattern or pattern == "checked" or pattern == "print"

    candidate_bottoms = [b for b in bottoms if (b["color"] in NEUTRALS) == prefer_neutral]
    if not candidate_bottoms:
        candidate_bottoms = bottoms

    if texture == "textured":
        candidate_bottoms = [b for b in candidate_bottoms if b["texture"] == "smooth"] or candidate_bottoms

    if pattern != "solid":
        candidate_shoes = [s for s in shoes if s["color"] in ["white", "black"]]
    else:
        candidate_shoes = shoes

    b1 = candidate_bottoms[0]
    s1 = candidate_shoes[0]
    b2 = candidate_bottoms[-1]
    s2 = candidate_shoes[-1]
    acc = ITEM_POOL["accessories"]
    acc_a = [acc[0], acc[2]]
    acc_b = [acc[1], acc[3]]
    boards = [
        {"style": "feature_based_1", "bottom": b1, "shoes": s1, "palette": [base_hex, "#f5f5f5", "#111111"], "title": "Feature Pair A", "accessories": acc_a},
        {"style": "feature_based_2", "bottom": b2, "shoes": s2, "palette": [base_hex, "#e0e0e0", "#333333"], "title": "Feature Pair B", "accessories": acc_b},
    ]
    return boards


def recommend_from_features(item_type: str, features: Dict[str, object]) -> List[Board]:
    boards = _choose_bottom_and_shoes_from_features(features)
    if item_type in ["bottom", "shoes"]:
        tops = ITEM_POOL["tops"]
        t1 = tops[0]
        t2 = tops[-1]
        boards[0]["top"] = t1
        boards[1]["top"] = t2
    return boards
