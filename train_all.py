"""
Master training script for the AI Fashion project.

Usage:
    # Step 1: Prepare DeepFashion2 data (after extracting the encrypted zips)
    python prepare_data.py --data-dir data/validation --output-dir dataset --augment

    # Step 2: Train the classifier
    python train_all.py --data-dir dataset --epochs-classifier 15

    # Quick train with fewer samples
    python train_all.py --data-dir dataset --max-samples 5000 --epochs-classifier 10
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="Train AI Fashion models on DeepFashion2")
    parser.add_argument("--data-dir", default="dataset", help="Directory with category subfolders (from prepare_data.py)")
    parser.add_argument("--epochs-classifier", type=int, default=15, help="Classifier training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit total training samples (0=unlimited)")
    parser.add_argument("--skip-classifier", action="store_true", help="Skip classifier training")
    parser.add_argument("--full-categories", action="store_true", help="Use all 13 DeepFashion2 categories")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' not found.")
        print(f"Run prepare_data.py first to organize the dataset:")
        print(f"  python prepare_data.py --data-dir data/validation --output-dir dataset --augment")
        return

    categories = [d.name for d in data_dir.iterdir() if d.is_dir()]
    if not categories:
        print(f"Error: No category folders found in '{data_dir}'")
        return

    print(f"Found {len(categories)} categories: {categories}")
    total_images = sum(len(list((data_dir / c).glob('*.jpg'))) for c in categories)
    print(f"Total images: {total_images}")

    if args.max_samples > 0:
        print(f"Limiting to {args.max_samples} samples per category")

    if not args.skip_classifier:
        print("\n" + "=" * 60)
        print("Training Clothing Classifier")
        print("=" * 60)
        from ai_fashion.classifier import train_classifier, CATEGORY_LABELS, FULL_CATEGORY_LABELS

        if args.full_categories:
            labels = FULL_CATEGORY_LABELS
        else:
            labels = CATEGORY_LABELS

        train_classifier(
            data_dir=str(data_dir),
            epochs=args.epochs_classifier,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            labels=labels,
        )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("Models saved to: models/")
    print("  - clothing_classifier.pt (item type classifier)")
    print("=" * 60)


if __name__ == "__main__":
    main()
