from pathlib import Path
from typing import Optional, Tuple


def detect_top_type(image_path: Optional[str] = None, hint: Optional[str] = None) -> Tuple[str, Optional[str]]:
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
    if image_path:
        name = Path(image_path).name.lower()
        if any(k in name for k in ["tshirt", "t-shirt", "tee"]):
            return "tshirt", None
        if "shirt" in name:
            return "shirt", None
        if "sweater" in name or "jumper" in name:
            return "sweater", None
        if "hoodie" in name:
            return "hoodie", None
        if "blouse" in name:
            return "blouse", None
        if "polo" in name:
            return "polo", None
        if "jacket" in name:
            return "jacket", None
        if "coat" in name:
            return "coat", None
    return "shirt", None
