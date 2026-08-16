
import base64
import uuid
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from ai_fashion.analyzer import analyze_image, detect_item_type, feature_vector
from ai_fashion.detector import detect_item_type as detect_item_type_nn
from ai_fashion.recommender import recommend_from_features, recommend_from_uploads, UPLOADED_SLOT


BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
GALLERY_DIR = BASE_DIR / "Re-PolyVore" / "Re-PolyVore" / "Re-PolyVore"

DISPLAY_TYPE = {
    "top": "Top",
    "outwear": "Outwear",
    "bottom": "Bottom",
    "dress": "Dress",
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
        boards = recommend_from_features(detected, feats)
        image_url = url_for("uploads", filename=Path(saved_path).name)
        display_type = DISPLAY_TYPE.get(detected, detected.title())

        uploaded_slot = UPLOADED_SLOT.get(detected)
        if uploaded_slot:
            for b in boards:
                b[uploaded_slot] = {"name": display_type, "image": image_url}

        return render_template(
            "board.html",
            boards=boards,
            uploaded_label=display_type,
            features=feats,
        )

    # Multiple photos: detect each, place it in its own slot, and build one
    # board that only fills gaps from the matched dataset.
    uploaded = {}
    last_feats = None
    for saved_path in saved_paths:
        detected = _detect(saved_path)
        feats = analyze_image(saved_path)
        last_feats = feats
        slot = UPLOADED_SLOT.get(detected)
        if not slot:
            continue
        dom = feats.get("dominant_colors") or []
        vec = feature_vector(
            dom[0] if dom else "#cccccc",
            str(feats.get("pattern", "solid")),
            str(feats.get("texture", "smooth")),
        )
        uploaded[slot] = {
            "name": DISPLAY_TYPE.get(detected, detected.title()),
            "image": url_for("uploads", filename=Path(saved_path).name),
            "vector": vec,
        }

    board = recommend_from_uploads(uploaded)
    return render_template(
        "board.html",
        boards=[board],
        uploaded_label="Your Photos",
        features=last_feats,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
