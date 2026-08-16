"""
Build a classifier training set (top/outwear/bottom/dress/shoes) by sampling
and copying images from Re-PolyVore into dataset/<category>/. "bottom" is
merged from the pants + skirt folders since Re-PolyVore keeps them separate
but the app treats them as one category.

Usage:
    python build_classifier_dataset.py --per-category 4000
"""
import argparse
import random
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent
DEFAULT_ROOT = BASE_DIR / "Re-PolyVore" / "Re-PolyVore" / "Re-PolyVore"
DEFAULT_OUT = BASE_DIR / "dataset"

# category -> source folders to draw from (split evenly across sources)
# No "dress" — the app only detects/pairs top, outwear, bottom, shoes.
SOURCES = {
    "top": ["top"],
    "outwear": ["outwear"],
    "bottom": ["pants", "skirt"],
    "shoes": ["shoes"],
}


def build(root: Path, out_dir: Path, per_category: int, seed: int = 7) -> None:
    rng = random.Random(seed)

    if out_dir.exists():
        shutil.rmtree(out_dir)

    for category, sources in SOURCES.items():
        cat_dir = out_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        per_source = per_category // len(sources)

        copied = 0
        for src_name in sources:
            src_dir = root / src_name
            if not src_dir.exists():
                print(f"  skip {src_name} (not found)")
                continue
            files = [f for f in src_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
            rng.shuffle(files)
            sample = files[:per_source]
            for f in sample:
                shutil.copy2(f, cat_dir / f"{src_name}_{f.name}")
                copied += 1
        print(f"  {category}: {copied} images")

    print(f"\nDataset written to: {out_dir}")


def main():
    p = argparse.ArgumentParser(description="Build classifier training set from Re-PolyVore")
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--per-category", type=int, default=4000)
    args = p.parse_args()
    build(Path(args.root), Path(args.out), args.per_category)


if __name__ == "__main__":
    main()
