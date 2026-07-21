"""
Clothing item type detector.
Uses the trained DeepFashion2 classifier when available,
falls back to MobileNet ImageNet inference and heuristics.
"""
import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple

from ai_fashion.classifier import (
    ClothingClassifier,
    classifier_transform,
    load_classifier,
    predict_category,
    CATEGORY_LABELS,
    FULL_TO_SUPER,
    FULL_CATEGORY_LABELS,
    IMAGE_SIZE as CLASSIFIER_IMAGE_SIZE,
)

_model = None
_preprocess = None
_classifier = None
_classifier_transform = None


def _get_mobilenet():
    global _model, _preprocess
    if _model is None:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        _model = mobilenet_v3_small(weights=weights)
        _model.eval()
        _preprocess = weights.transforms()
    return _model, _preprocess


def _get_classifier():
    global _classifier, _classifier_transform
    if _classifier is None:
        _classifier, _classifier_transform = load_classifier()
    return _classifier, _classifier_transform


# Fallback: ImageNet classes mapped to fashion categories
FASHION_MAPPING = {
    608: "jacket", 617: "jacket",
    459: "shirt", 600: "shirt", 701: "shirt",
    841: "tshirt", 610: "shirt",
    603: "bottom", 845: "bottom",
    770: "shoes", 626: "shoes", 514: "shoes", 452: "shoes", 802: "shoes", 931: "shoes",
}


def detect_item_type(image_path: Optional[str] = None, hint: Optional[str] = None) -> str:
    """
    Detect the type of clothing item in the image.
    Returns: 'top', 'outwear', 'bottom', 'dress', or 'shoes'
    """
    if hint:
        h = hint.lower()
        if any(k in h for k in ["shirt", "t-shirt", "tee", "sweater", "hoodie", "blouse", "polo", "tank", "vest top"]):
            return "top"
        if any(k in h for k in ["jacket", "coat", "blazer", "cardigan", "outwear"]):
            return "outwear"
        if any(k in h for k in ["pants", "jeans", "trousers", "shorts", "skirt", "bottom", "chinos"]):
            return "bottom"
        if any(k in h for k in ["dress", "gown", "frock"]):
            return "dress"
        if any(k in h for k in ["shoe", "sneaker", "boot", "sandal", "loafer", "heel"]):
            return "shoes"

    # Try trained DeepFashion2 classifier
    if image_path and Path(image_path).exists():
        classifier, c_transform = _get_classifier()
        if classifier is not None:
            try:
                predicted, confidence = predict_category(
                    image_path, classifier, c_transform, labels=CATEGORY_LABELS
                )
                return predicted
            except Exception as e:
                print(f"Classifier failed: {e}. Falling back to MobileNet.")

        # Try MobileNet with fashion mapping
        try:
            model, preprocess = _get_mobilenet()
            img = Image.open(image_path).convert("RGB")
            batch = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                prediction = model(batch).squeeze(0)
                probs = torch.nn.functional.softmax(prediction, dim=0)
                class_id = probs.argmax().item()
                if class_id in FASHION_MAPPING:
                    return FASHION_MAPPING[class_id]
        except Exception as e:
            print(f"MobileNet inference failed: {e}")

    # Filename heuristics
    if image_path:
        name = Path(image_path).name.lower()
        if any(k in name for k in ["tshirt", "t-shirt", "tee"]):
            return "top"
        if "shirt" in name:
            return "top"
        if any(k in name for k in ["sweater", "jumper", "hoodie"]):
            return "top"
        if any(k in name for k in ["jacket", "coat"]):
            return "outwear"
        if any(k in name for k in ["pants", "jeans", "trousers", "shorts"]):
            return "bottom"
        if "dress" in name:
            return "dress"
        if "shoe" in name or "sneaker" in name or "boot" in name:
            return "shoes"

    return "top"


def detect_top_type(image_path: Optional[str] = None, hint: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Legacy interface: detect the top clothing type.
    Returns: (type_label, error_or_none)
    """
    if hint:
        h = hint.lower()
        if "shirt" in h:
            return "shirt", None
        if "t-shirt" in h or "tee" in h:
            return "tshirt", None
        if "sweater" in h or "jumper" in h:
            return "sweater", None
        if "hoodie" in h:
            return "hoodie", None
        if "blouse" in h:
            return "blouse", None
        if "polo" in h:
            return "polo", None
        if "jacket" in h:
            return "jacket", None
        if "coat" in h:
            return "coat", None

    item_type = detect_item_type(image_path)
    return item_type, None
