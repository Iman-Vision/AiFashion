"""
Clothing Classifier using transfer learning with MobileNetV3.
Trained on DeepFashion2 categories.
"""
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from pathlib import Path
from typing import Optional, Tuple, Dict, List

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
CLASSIFIER_PATH = MODEL_DIR / "clothing_classifier.pt"
IMAGE_SIZE = 224

# Super-categories used by the app. No "dress" — the app only detects and
# pairs top / outwear / bottom / shoes. Kept alphabetically sorted on
# purpose: ImageFolder assigns class indices by sorted folder name, so
# training and inference only agree on which index means what if this list
# is sorted the same way the dataset folders are.
CATEGORY_LABELS = sorted(["top", "outwear", "bottom", "shoes"])

# Full 13-category labels for finer classification
FULL_CATEGORY_LABELS = [
    "short_sleeve_top", "long_sleeve_top",
    "short_sleeve_outwear", "long_sleeve_outwear",
    "vest", "sling",
    "shorts", "trousers", "skirt",
    "short_sleeve_dress", "long_sleeve_dress",
    "vest_dress", "sling_dress",
]

# Map full categories to super-categories
FULL_TO_SUPER = {
    "short_sleeve_top": "top", "long_sleeve_top": "top",
    "vest": "top", "sling": "top",
    "short_sleeve_outwear": "outwear", "long_sleeve_outwear": "outwear",
    "shorts": "bottom", "trousers": "bottom", "skirt": "bottom",
    "short_sleeve_dress": "dress", "long_sleeve_dress": "dress",
    "vest_dress": "dress", "sling_dress": "dress",
}


class ClothingClassifier(nn.Module):
    def __init__(self, num_classes: int = len(CATEGORY_LABELS), pretrained: bool = True):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

        if pretrained:
            weights = MobileNet_V3_Small_Weights.DEFAULT
            self.backbone = mobilenet_v3_small(weights=weights)
        else:
            self.backbone = mobilenet_v3_small(weights=None)

        # classifier[0] is the Linear taking the backbone's pooled features
        # (576-d for mobilenet_v3_small) — we replace the whole classifier
        # Sequential, so we need its *input* size, not classifier[3]'s.
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def classifier_transform(image_size: int = IMAGE_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_classifier(
    model_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    num_classes: int = len(CATEGORY_LABELS),
) -> Tuple[Optional[ClothingClassifier], T.Compose]:
    transform = classifier_transform()
    path = Path(model_path) if model_path else CLASSIFIER_PATH

    if not path.exists():
        return None, transform

    device = device or get_device()
    model = ClothingClassifier(num_classes=num_classes).to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model, transform


def predict_category(
    image_path: str,
    model: ClothingClassifier,
    transform: T.Compose,
    device: Optional[torch.device] = None,
    labels: Optional[List[str]] = None,
) -> Tuple[str, float]:
    from PIL import Image
    import io
    import urllib.request

    device = device or get_device()
    if labels is None:
        labels = CATEGORY_LABELS

    if image_path.startswith("http://") or image_path.startswith("https://"):
        req = urllib.request.Request(image_path, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            data = response.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")

    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = probs.max(1)

    idx = predicted.item()
    return labels[idx], confidence.item()


def _evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    model.train()
    return 100.0 * correct / max(1, total)


def train_classifier(
    data_dir: str,
    model_out: Optional[str] = None,
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    image_size: int = IMAGE_SIZE,
    device: Optional[torch.device] = None,
    num_classes: int = len(CATEGORY_LABELS),
    labels: Optional[List[str]] = None,
    val_split: float = 0.15,
) -> float:
    """Trains the classifier, holding out `val_split` of the data as a
    never-trained-on test set, and returns its final held-out accuracy.
    Training-loop accuracy alone is meaningless for generalization — it's
    graded on data the model has already seen/memorized."""
    device = device or get_device()
    model_out_path = Path(model_out) if model_out else CLASSIFIER_PATH
    model_out_path.parent.mkdir(parents=True, exist_ok=True)

    # Training data is 100% flat product-catalog shots. A heavier-augmentation
    # variant (random-resized-crop, color jitter, random erasing) was tried
    # to reduce overfitting to that pattern, but it didn't fix real-photo
    # misclassification and cost confidence on the catalog-style photos this
    # is actually good at — reverted. Real fix for on-model photos needs
    # on-model training data, not more augmentation on flat shots (see
    # README's Known Limitations); the app now asks the user to confirm/
    # correct the detected type instead of trusting a shaky guess blindly.
    train_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = classifier_transform(image_size)  # same as real inference — no augmentation

    # Two ImageFolder views of the same directory (file order is identical
    # and deterministic) so the held-out split gets the plain eval transform
    # instead of the training-time augmentation.
    train_view = ImageFolder(data_dir, transform=train_transform)
    eval_view = ImageFolder(data_dir, transform=eval_transform)

    if labels is None:
        labels = train_view.classes
    num_classes = len(labels)

    indices = list(range(len(train_view)))
    random.Random(42).shuffle(indices)
    split_at = int(len(indices) * (1 - val_split))
    train_idx, val_idx = indices[:split_at], indices[split_at:]

    train_subset = Subset(train_view, train_idx)
    val_subset = Subset(eval_view, val_idx)
    dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Classes ({num_classes}): {labels}")
    print(f"Training samples: {len(train_subset)} | Held-out test samples: {len(val_subset)}")

    model = ClothingClassifier(num_classes=num_classes, pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        scheduler.step()
        avg_loss = total_loss / max(1, total)
        train_accuracy = 100.0 * correct / max(1, total)
        val_accuracy = _evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f} - train acc: {train_accuracy:.1f}% - TEST acc: {val_accuracy:.1f}%")

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            # torch.save truncates the destination immediately, so saving
            # straight to model_out_path would wipe the last-good checkpoint
            # the instant a new one starts writing — if the process dies
            # mid-write (killed, crash, OOM) the file is left empty/corrupt.
            # Write to a temp file and atomically replace instead.
            tmp_path = model_out_path.with_suffix(model_out_path.suffix + ".tmp")
            torch.save(model.state_dict(), tmp_path)
            tmp_path.replace(model_out_path)
            print(f"  Saved best model (held-out test acc: {val_accuracy:.1f}%)")

    print(f"\nBest held-out TEST accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {model_out_path}")
    return best_val_acc


def main():
    import argparse

    parser = argparse.ArgumentParser(prog="classifier", description="Train clothing classifier on DeepFashion2")
    parser.add_argument("--data-dir", required=True, help="Directory with category subfolders")
    parser.add_argument("--model-out", default=None, help="Output path for trained model")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--full-categories", action="store_true", help="Use all 13 DeepFashion2 categories")
    args = parser.parse_args()

    if args.full_categories:
        labels = FULL_CATEGORY_LABELS
    else:
        labels = CATEGORY_LABELS

    train_classifier(
        data_dir=args.data_dir,
        model_out=args.model_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        labels=labels,
    )


if __name__ == "__main__":
    main()
