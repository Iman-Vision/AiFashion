
import base64
import json
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from ai_fashion.analyzer import analyze_image, detect_item_type, feature_vector, estimate_shoe_formality, _open_image
from ai_fashion.detector import detect_item_type as detect_item_type_nn
from ai_fashion.recommender import (
    recommend_from_features,
    recommend_from_uploads,
    suggest_additions,
    UPLOADED_SLOT,
    ESSENTIAL_SLOTS,
)


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


def _feature_vector_for(feats, saved_path: str = None, item_type: str = None) -> np.ndarray:
    dom = feats.get("dominant_colors") or []
    formality = "neutral"
    if item_type == "shoes" and saved_path:
        try:
            formality = estimate_shoe_formality(_open_image(saved_path))
        except Exception:
            pass
    return feature_vector(
        dom[0] if dom else "#cccccc",
        str(feats.get("pattern", "solid")),
        str(feats.get("texture", "smooth")),
        formality,
    )


def _board_context_vectors(board) -> list:
    return [np.array(board[s]["vector"]).tolist() for s in ESSENTIAL_SLOTS if board.get(s) and "vector" in board[s]]


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
        target_vector = _feature_vector_for(feats, saved_path, detected)
        boards = recommend_from_features(detected, feats, count=3)
        image_url = url_for("uploads", filename=Path(saved_path).name)
        display_type = DISPLAY_TYPE.get(detected, detected.title())

        uploaded_slot = UPLOADED_SLOT.get(detected)
        if uploaded_slot:
            for b in boards:
                b[uploaded_slot] = {"name": display_type, "image": image_url, "vector": target_vector.tolist()}

        return render_template(
            "board.html",
            boards=boards,
            uploaded_label=display_type,
            features=feats,
            target_vector=json.dumps(target_vector.tolist()),
            context_vectors=json.dumps(_board_context_vectors(boards[0]) if boards else []),
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
        vec = _feature_vector_for(feats, saved_path, detected)
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
        context_vectors=json.dumps(_board_context_vectors(boards[0]) if boards else []),
    )


@app.route("/suggest", methods=["POST"])
def suggest():
    """Sidebar 'what else goes with this' — ranks an optional add-on
    (an outwear layer, or an accessory category) against the pieces
    actually on the board, not just the originally uploaded item. Each
    result comes back with a 'fits' flag: true means it's coherent enough
    to go straight onto the moodboard, false means it's shown only as a
    loose suggestion, never spliced into the board itself."""
    category = (request.form.get("category") or "").strip().lower()
    raw = request.form.get("context_vectors") or "[]"
    try:
        context = [np.array(v, dtype=np.float64) for v in json.loads(raw)]
    except (ValueError, TypeError):
        context = []
    results = suggest_additions(category, context, count=8)
    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
