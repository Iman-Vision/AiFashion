import base64
import os
import uuid
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from ai_fashion.analyzer import analyze_image, detect_item_type
from ai_fashion.recommender import recommend_from_features


BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=False)


@app.route("/analyze", methods=["POST"])
def analyze():
    img_file = request.files.get("image")
    cam_data = request.form.get("camera_image")
    saved_path = None
    if cam_data and cam_data.startswith("data:image"):
        header, b64 = cam_data.split(",", 1)
        raw = base64.b64decode(b64)
        name = f"{uuid.uuid4().hex}.png"
        path = UPLOAD_DIR / name
        with open(path, "wb") as f:
            f.write(raw)
        saved_path = str(path)
    elif img_file and img_file.filename:
        name = f"{uuid.uuid4().hex}_{img_file.filename}"
        path = UPLOAD_DIR / name
        img_file.save(str(path))
        saved_path = str(path)
    if not saved_path:
        return redirect(url_for("index", error="missing_image"))
    detected = detect_item_type(saved_path)
    feats = analyze_image(saved_path)
    boards = recommend_from_features(detected, feats)
    image_url = url_for("uploads", filename=Path(saved_path).name)
    return render_template("board.html", boards=boards, uploaded_image=image_url, uploaded_type=detected, features=feats)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
