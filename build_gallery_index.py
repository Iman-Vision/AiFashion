"""
Build a matching index from the Re-PolyVore dataset.

For each category folder under Re-PolyVore/Re-PolyVore/Re-PolyVore, samples N
images, extracts dominant color / pattern / texture with ai_fashion.analyzer,
copies the sampled images into gallery_assets/<category>/ (a small permanent
local folder — the point is that once this has run, the multi-GB Re-PolyVore
dump itself is no longer needed and can be deleted), and stores the
resulting feature vectors + local paths in models/gallery_index.json.
ai_fashion.recommender loads this index at runtime and matches the uploaded
item against it with cosine similarity — no training required.

Usage:
    python build_gallery_index.py
    python build_gallery_index.py --per-category 400
"""
import argparse
import json
import random
import shutil
from pathlib import Path

from ai_fashion.analyzer import analyze_image, feature_vector

BASE_DIR = Path(__file__).parent
DEFAULT_ROOT = BASE_DIR / "Re-PolyVore" / "Re-PolyVore" / "Re-PolyVore"
DEFAULT_OUT = BASE_DIR / "models" / "gallery_index.json"
DEFAULT_ASSETS = BASE_DIR / "gallery_assets"

# Map dataset folders -> app slots. Dresses are intentionally excluded —
# the app only deals in top / outwear / bottom / shoes. Folders not listed
# here are ignored.
SLOT_MAP = {
    "top": "tops",
    "outwear": "outwear",
    "pants": "bottoms",
    "skirt": "bottoms",
    "shoes": "shoes",
    "bag": "accessories",
    "bracelet": "accessories",
    "brooch": "accessories",
    "earrings": "accessories",
    "eyewear": "accessories",
    "gloves": "accessories",
    "hairwear": "accessories",
    "hats": "accessories",
    "necklace": "accessories",
    "neckwear": "accessories",
    "rings": "accessories",
    "watches": "accessories",
}

# Sub-label shown/used for accessory category badges.
ACCESSORY_LABEL = {
    "bag": "bag", "bracelet": "jewelry", "brooch": "jewelry", "earrings": "jewelry",
    "eyewear": "glasses", "gloves": "accessory", "hairwear": "hair", "hats": "hats",
    "necklace": "jewelry", "neckwear": "accessory", "rings": "jewelry", "watches": "watch",
}

NAME_OVERRIDE = {
    "top": "Top", "outwear": "Outwear Piece", "pants": "Pants", "skirt": "Skirt",
    "shoes": "Shoes", "bag": "Bag", "bracelet": "Bracelet",
    "brooch": "Brooch", "earrings": "Earrings", "eyewear": "Eyewear", "gloves": "Gloves",
    "hairwear": "Hair Accessory", "hats": "Hat", "necklace": "Necklace",
    "neckwear": "Neckwear", "rings": "Ring", "watches": "Watch",
}


def build_index(root: Path, out_path: Path, assets_dir: Path, per_category: int, seed: int = 7) -> None:
    rng = random.Random(seed)
    pool = {slot: [] for slot in set(SLOT_MAP.values())}

    for folder, slot in SLOT_MAP.items():
        cat_dir = root / folder
        if not cat_dir.exists():
            print(f"  skip {folder} (not found)")
            continue

        out_cat_dir = assets_dir / folder
        out_cat_dir.mkdir(parents=True, exist_ok=True)

        files = [f for f in cat_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        rng.shuffle(files)
        sample = files[:per_category]
        print(f"  {folder}: sampling {len(sample)}/{len(files)} -> slot '{slot}'")

        for i, f in enumerate(sample):
            try:
                feats = analyze_image(str(f))
            except Exception:
                continue
            colors = feats.get("dominant_colors") or ["#999999"]
            vec = feature_vector(colors[0], feats.get("pattern", "solid"), feats.get("texture", "smooth"))

            dest = out_cat_dir / f.name
            shutil.copy2(f, dest)
            rel = f"{folder}/{f.name}"

            item = {
                "name": f"{NAME_OVERRIDE.get(folder, folder.title())} {i + 1}",
                "image": f"/gallery/{rel}",
                "vector": vec.tolist(),
            }
            if slot == "accessories":
                item["category"] = ACCESSORY_LABEL.get(folder, folder)
            pool[slot].append(item)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pool, indent=2), encoding="utf-8")
    print(f"\nSaved index: {out_path}")
    print(f"Copied assets to: {assets_dir}")
    for slot, items in pool.items():
        print(f"  {slot}: {len(items)} items")


def main():
    p = argparse.ArgumentParser(description="Build gallery match index from Re-PolyVore")
    p.add_argument("--root", default=str(DEFAULT_ROOT), help="Path to Re-PolyVore category folders")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Output index JSON path")
    p.add_argument("--assets", default=str(DEFAULT_ASSETS), help="Local folder to copy sampled images into")
    p.add_argument("--per-category", type=int, default=400, help="Images to sample per category")
    args = p.parse_args()
    build_index(Path(args.root), Path(args.out), Path(args.assets), args.per_category)


if __name__ == "__main__":
    main()
