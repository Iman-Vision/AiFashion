"""
DeepFashion2 Data Preparation Script

Extracts and organizes clothing items from the DeepFashion2 dataset
into category-organized folders suitable for training.

Usage:
    python prepare_data.py --data-dir data/validation --output-dir dataset
    python prepare_data.py --data-dir data/train --output-dir dataset --augment

Requires: the DeepFashion2 dataset to be extracted first.
The zip files are password-protected. Get the password from:
https://docs.google.com/forms/d/e/1FAIpQLSeIoGaFfCQILrtIZPykkr8q_h9qQ5BoTYbjvf95aXbid0v2Bw/viewform
"""
import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

# DeepFashion2 has 13 categories
CATEGORY_NAMES = {
    1: "short_sleeve_top",
    2: "long_sleeve_top",
    3: "short_sleeve_outwear",
    4: "long_sleeve_outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short_sleeve_dress",
    11: "long_sleeve_dress",
    12: "vest_dress",
    13: "sling_dress",
}

# Map to broader groups for the app
SUPER_CATEGORIES = {
    "top": ["short_sleeve_top", "long_sleeve_top", "vest", "sling"],
    "outwear": ["short_sleeve_outwear", "long_sleeve_outwear"],
    "bottom": ["shorts", "trousers", "skirt"],
    "dress": ["short_sleeve_dress", "long_sleeve_dress", "vest_dress", "sling_dress"],
}

# Reverse mapping
NAME_TO_ID = {v: k for k, v in CATEGORY_NAMES.items()}
SUPER_TO_NAMES = {}
for super_cat, sub_cats in SUPER_CATEGORIES.items():
    for name in sub_cats:
        SUPER_TO_NAMES[name] = super_cat


def load_annotations(annos_dir: Path) -> Dict[str, dict]:
    annotations = {}
    for f in annos_dir.glob("*.json"):
        try:
            with open(f, "r") as fh:
                data = json.load(fh)
            annotations[f.stem] = data
        except Exception as e:
            print(f"  Skipping {f.name}: {e}")
    return annotations


def crop_item(
    image: Image.Image,
    bbox: List[float],
    padding: float = 0.05,
) -> Optional[Image.Image]:
    w, h = image.size
    x1, y1, x2, y2 = bbox

    pad_w = (x2 - x1) * padding
    pad_h = (y2 - y1) * padding
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)

    if x2 - x1 < 10 or y2 - y1 < 10:
        return None

    return image.crop((int(x1), int(y1), int(x2), int(y2)))


def prepare_dataset(
    data_dir: str,
    output_dir: str,
    target_size: int = 256,
    max_per_category: int = 0,
    use_super_categories: bool = True,
    min_scale: int = 0,
    augment: bool = False,
) -> None:
    data_path = Path(data_dir)
    out_path = Path(output_dir)

    annos_dir = data_path / "annos"
    images_dir = data_path / "image"

    if not annos_dir.exists():
        print(f"Annotations directory not found: {annos_dir}")
        print("Make sure the DeepFashion2 dataset is properly extracted.")
        return

    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return

    print(f"Loading annotations from {annos_dir}...")
    annotations = load_annotations(annos_dir)
    print(f"Found {len(annotations)} annotation files")

    category_counts: Dict[str, int] = {}
    skipped = 0
    processed = 0

    for img_name, anno_data in sorted(annotations.items()):
        img_path = images_dir / f"{img_name}.jpg"
        if not img_path.exists():
            skipped += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            continue

        items = anno_data if isinstance(anno_data, list) else anno_data.get("item", [])
        if isinstance(anno_data, dict) and not isinstance(items, list):
            items = [items] if items else []

        for item in items:
            if not isinstance(item, dict):
                continue

            cat_id = item.get("category_id", 0)
            cat_name = CATEGORY_NAMES.get(cat_id)
            if not cat_name:
                continue

            # Filter by minimum scale if specified
            scale = item.get("scale", 3)
            if scale < min_scale:
                continue

            # Check occlusion - skip heavily occluded items
            occlusion = item.get("occlusion", 1)
            if occlusion >= 3:
                continue

            bbox = item.get("bounding_box", [])
            if len(bbox) != 4:
                continue

            # DeepFashion2 uses [x1,y1,x2,y2] with y1 being bottom
            # PIL uses (left, top, right, bottom) with top=0
            x1, y1, x2, y2 = bbox
            # Convert: swap y coordinates
            pil_bbox = [x1, y1, x2, y2]

            cropped = crop_item(image, pil_bbox)
            if cropped is None:
                skipped += 1
                continue

            # Determine output category folder
            if use_super_categories:
                folder_name = SUPER_TO_NAMES.get(cat_name, "other")
            else:
                folder_name = cat_name

            cat_dir = out_path / folder_name
            cat_dir.mkdir(parents=True, exist_ok=True)

            # Resize to target size maintaining aspect ratio
            cropped = cropped.resize((target_size, target_size), Image.LANCZOS)

            # Save
            out_name = f"{img_name}_{cat_id}_{processed:06d}.jpg"
            cropped.save(cat_dir / out_name, "JPEG", quality=90)

            category_counts[folder_name] = category_counts.get(folder_name, 0) + 1
            processed += 1

            if max_per_category > 0 and category_counts.get(folder_name, 0) >= max_per_category:
                continue

            # Simple augmentation: horizontal flip
            if augment and random.random() > 0.5:
                flipped = cropped.transpose(Image.FLIP_LEFT_RIGHT)
                aug_name = f"{img_name}_{cat_id}_{processed:06d}_flip.jpg"
                flipped.save(cat_dir / aug_name, "JPEG", quality=90)
                category_counts[folder_name] = category_counts.get(folder_name, 0) + 1
                processed += 1

        if processed % 1000 == 0 and processed > 0:
            print(f"  Processed {processed} items...")

        if max_per_category > 0 and all(
            v >= max_per_category for v in category_counts.values()
        ):
            break

    print(f"\nDone! Processed: {processed}, Skipped: {skipped}")
    print(f"Output directory: {out_path}")
    print(f"\nCategory distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Prepare DeepFashion2 dataset for training")
    parser.add_argument("--data-dir", required=True, help="Path to extracted DeepFashion2 folder (e.g., data/validation)")
    parser.add_argument("--output-dir", default="dataset", help="Output directory for organized dataset")
    parser.add_argument("--target-size", type=int, default=256, help="Image size (default: 256)")
    parser.add_argument("--max-per-category", type=int, default=0, help="Max images per category (0=unlimited)")
    parser.add_argument("--no-super", action="store_true", help="Use individual categories instead of super-categories")
    parser.add_argument("--min-scale", type=int, default=0, help="Minimum scale (1=small, 2=modest, 3=large)")
    parser.add_argument("--augment", action="store_true", help="Apply simple augmentation (horizontal flip)")
    args = parser.parse_args()

    prepare_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_size=args.target_size,
        max_per_category=args.max_per_category,
        use_super_categories=not args.no_super,
        min_scale=args.min_scale,
        augment=args.augment,
    )


if __name__ == "__main__":
    main()
