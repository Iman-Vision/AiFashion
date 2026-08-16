
import base64
import json
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from ai_fashion.analyzer import analyze_image, detect_item_type, feature_vector
from ai_fashion.detector import detect_item_type as detect_item_type_nn
from ai_fashion.recommender import recommend_from_features, recommend_from_uploads, accessory_options, UPLOADED_SLOT


BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
GALLERY_DIR = BASE_DIR / "gallery_assets"

DISPLAY_TYPE = {
    "top": "Top",
    "outwear": "Outwear",
    "bottom": "Bottom",
    "shoes": "Shoes",
}

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=False)


@app.route("/gallery/<path:relpath>")
def gallery(relpath):
    return send_from_directory(GALLERY_DIR, relpath, as_attachment=False)


def _save_upload(file_storage) -> str:
    name = f"{uuid.uuid4().hex}_{file_storage.filename}"
    path = UPLOAD_DIR / name
    file_storage.save(str(path))
    return str(path)


def _save_camera_capture(data_uri: str) -> str:
    header, b64 = data_uri.split(",", 1)
    raw = base64.b64decode(b64)
    name = f"{uuid.uuid4().hex}.png"
    path = UPLOAD_DIR / name
    with open(path, "wb") as f:
        f.write(raw)
    return str(path)


def _detect(saved_path: str) -> str:
    # Try the neural network classifier first, fall back to heuristic
    try:
        return detect_item_type_nn(saved_path)
    except Exception:
        return detect_item_type(saved_path)


def _feature_vector_for(feats) -> np.ndarray:
    dom = feats.get("dominant_colors") or []
    return feature_vector(
        dom[0] if dom else "#cccccc",
        str(feats.get("pattern", "solid")),
        str(feats.get("texture", "smooth")),
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    img_files = [f for f in request.files.getlist("image") if f and f.filename]
    cam_data = request.form.get("camera_image")

    saved_paths = [_save_upload(f) for f in img_files]
    if cam_data and cam_data.startswith("data:image"):
        saved_paths.append(_save_camera_capture(cam_data))

    if not saved_paths:
        return redirect(url_for("index", error="missing_image"))

    if len(saved_paths) == 1:
        saved_path = saved_paths[0]
        detected = _detect(saved_path)
        feats = analyze_image(saved_path)
        boards = recommend_from_features(detected, feats, count=3)
        image_url = url_for("uploads", filename=Path(saved_path).name)
        display_type = DISPLAY_TYPE.get(detected, detected.title())
        target_vector = _feature_vector_for(feats)

        uploaded_slot = UPLOADED_SLOT.get(detected)
        if uploaded_slot:
            for b in boards:
                b[uploaded_slot] = {"name": display_type, "image": image_url}

        return render_template(
            "board.html",
            boards=boards,
            uploaded_label=display_type,
            features=feats,
            target_vector=json.dumps(target_vector.tolist()),
        )

    # Multiple photos: detect each, place it in its own slot, and build
    # boards that only fill the gaps from the matched dataset.
    uploaded = {}
    last_feats = None
    vectors = []
    for saved_path in saved_paths:
        detected = _detect(saved_path)
        feats = analyze_image(saved_path)
        last_feats = feats
        slot = UPLOADED_SLOT.get(detected)
        if not slot:
            continue
        vec = _feature_vector_for(feats)
        vectors.append(vec)
        uploaded[slot] = {
            "name": DISPLAY_TYPE.get(detected, detected.title()),
            "image": url_for("uploads", filename=Path(saved_path).name),
            "vector": vec.tolist(),
        }

    boards = recommend_from_uploads(uploaded, count=3)
    target_vector = np.mean(vectors, axis=0) if vectors else feature_vector("#cccccc")

    return render_template(
        "board.html",
        boards=boards,
        uploaded_label="Your Photos",
        features=last_feats,
        target_vector=json.dumps(target_vector.tolist()),
    )


@app.route("/accessorize", methods=["POST"])
def accessorize():
    """Opt-in accessory matches for a look — called only when the user asks
    to see what jewelry/bags/watches would go with the board, not baked
    into the initial result."""
    raw = request.form.get("target_vector") or "[]"
    try:
        vector = np.array(json.loads(raw), dtype=np.float64)
    except (ValueError, TypeError):
        return jsonify([])
    if vector.size == 0:
        return jsonify([])
    items = accessory_options(vector, count=6)
    return jsonify([{"name": it["name"], "image": it["image"], "category": it.get("category", "")} for it in items])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
