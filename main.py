import argparse
from pathlib import Path
from typing import Optional

from ai_fashion.detector import detect_item_type, detect_top_type
from ai_fashion.recommender import recommend_for_top, recommend_from_features
from ai_fashion.moodboard import render_moodboard_html
from ai_fashion.analyzer import analyze_image


def run(item_image: Optional[str], item_hint: Optional[str], item_type: str) -> None:
    itype = item_type
    if itype == "auto":
        itype = "top"
    boards = []
    up_label = None
    if item_image:
        itype = detect_item_type(item_image, item_hint)
        feats = analyze_image(item_image)
        boards = recommend_from_features(itype, feats)
        up_label = itype.title()
    else:
        top_type, _ = detect_top_type(None, item_hint)
        boards = recommend_for_top(top_type)
        itype = top_type
        up_label = top_type.title()
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    for i, b in enumerate(boards, start=1):
        out_path = out_dir / f"moodboard_{i}_{b['style']}.html"
        render_moodboard_html(b, str(out_path), item_image, itype, up_label)
        print(str(out_path.resolve()))


def main() -> None:
    p = argparse.ArgumentParser(prog="ai-fashion", description="Generate two AI-inspired mood boards from a clothing item.")
    p.add_argument("--image", "-i", help="Path or URL to an image", default=None)
    p.add_argument("--hint", "-t", help="Text hint for fallback classification", default=None)
    p.add_argument("--item", choices=["auto", "top", "outwear", "bottom", "dress", "shoes"], default="auto", help="What item is uploaded")
    args = p.parse_args()
    if not args.image and not args.hint:
        print("Provide --image or --hint")
        return
    run(args.image, args.hint, args.item)


if __name__ == "__main__":
    main()
