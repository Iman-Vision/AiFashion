"""
Clothing Classifier using transfer learning with MobileNetV3.
Trained on DeepFashion2 categories.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
) -> None:
    device = device or get_device()
    model_out_path = Path(model_out) if model_out else CLASSIFIER_PATH
    model_out_path.parent.mkdir(parents=True, exist_ok=True)

    transform = classifier_transform(image_size)

    # Training data is 100% flat product-catalog shots (garment centered,
    # filling the frame, white background) — real user uploads are often a
    # person wearing the item, off-center, at an angle, with a cluttered
    # background. Without simulating that gap during training, the model
    # overfits to "centered flat product" and mispredicts confidently on
    # anything else (verified: a real photo mispredicted "shoes" at 99.97%).
    # RandomResizedCrop/perspective/erasing approximate that gap.
    train_transform = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.7, 1.3)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.RandomPerspective(distortion_scale=0.3, p=0.4),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        T.ToTensor(),
        T.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolder(data_dir, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    if labels is None:
        labels = dataset.classes
    num_classes = len(labels)
    print(f"Classes ({num_classes}): {labels}")
    print(f"Training samples: {len(dataset)}")

    model = ClothingClassifier(num_classes=num_classes, pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
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
        accuracy = 100.0 * correct / max(1, total)
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f} - accuracy: {accuracy:.1f}%")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_out_path)
            print(f"  Saved best model (loss: {avg_loss:.4f})")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Model saved to: {model_out_path}")


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
