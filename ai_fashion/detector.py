import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple

# Load a pre-trained Neural Network (MobileNetV3 is lightweight and fast)
_model = None
_preprocess = None

def _get_model():
    global _model, _preprocess
    if _model is None:
        # Using MobileNetV3 Small for fast CPU inference
        weights = MobileNet_V3_Small_Weights.DEFAULT
        _model = mobilenet_v3_small(weights=weights)
        _model.eval()
        _preprocess = weights.transforms()
    return _model, _preprocess

# Mapping ImageNet classes to our fashion categories
# These indices correspond to various clothing items in ImageNet
FASHION_MAPPING = {
    # Tops / Upper body
    608: "jacket", # jean jacket
    617: "jacket", # lab coat
    459: "shirt",  # brassiere (fallback to shirt)
    600: "shirt",  # hook
    701: "shirt",  # pajama
    841: "tshirt", # sweatshirt
    610: "shirt",  # jersey
    # Bottoms
    603: "bottom", # jean, blue jean
    845: "bottom", # suit, suit of clothes
    # Shoes
    770: "shoes",  # running shoe
    626: "shoes",  # loafer
    514: "shoes",  # cowboy boot
    452: "shoes",  # boot
    802: "shoes",  # shoe shop (often detects shoes)
    931: "shoes",  # clog
}

def detect_top_type(image_path: Optional[str] = None, hint: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Uses a Neural Network to detect the type of clothing.
    """
    # 1. Try Hint first (explicit user intent)
    if hint:
        h = hint.lower()
        if "shirt" in h: return "shirt", None
        if "t-shirt" in h or "tee" in h: return "tshirt", None
        if "sweater" in h or "jumper" in h: return "sweater", None
        if "hoodie" in h: return "hoodie", None
        if "blouse" in h: return "blouse", None
        if "polo" in h: return "polo", None
        if "jacket" in h: return "jacket", None
        if "coat" in h: return "coat", None

    # 2. Try Neural Network Inference
    if image_path and Path(image_path).exists():
        try:
            model, preprocess = _get_model()
            img = Image.open(image_path).convert("RGB")
            batch = preprocess(img).unsqueeze(0)
            
            with torch.no_grad():
                prediction = model(batch).squeeze(0)
                probs = torch.nn.functional.softmax(prediction, dim=0)
                class_id = probs.argmax().item()
                
                # Check if the detected class is in our fashion mapping
                if class_id in FASHION_MAPPING:
                    return FASHION_MAPPING[class_id], None
        except Exception as e:
            print(f"NN Inference failed: {e}. Falling back to heuristics.")

    # 3. Fallback to filename heuristics
    if image_path:
        name = Path(image_path).name.lower()
        if any(k in name for k in ["tshirt", "t-shirt", "tee"]): return "tshirt", None
        if "shirt" in name: return "shirt", None
        if "sweater" in name or "jumper" in name: return "sweater", None
        if "hoodie" in name: return "hoodie", None
        if "jacket" in name: return "jacket", None
        
    return "shirt", None
