import base64
import binascii
import logging
import os
import secrets
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge


BASE_DIR = Path(__file__).resolve().parent
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
EMOTION_LABELS = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)
MAX_IMAGE_DIMENSION = 6000
MAX_IMAGE_PIXELS = 25_000_000
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("FLASK_SECRET_KEY") or secrets.token_urlsafe(32),
    MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "").lower()
    in {"1", "true", "yes"},
    PERMANENT_SESSION_LIFETIME=60 * 60 * 8,
    APP_USERNAME=os.environ.get("APP_USERNAME", ""),
    APP_PASSWORD=os.environ.get("APP_PASSWORD", ""),
)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

if not os.environ.get("FLASK_SECRET_KEY"):
    logger.warning("FLASK_SECRET_KEY is not set; sessions will reset when the app restarts.")


@lru_cache(maxsize=1)
def _get_deepface():
    from deepface import DeepFace

    return DeepFace


def _json_error(message, status):
    return jsonify({"success": False, "error": message}), status


def _decode_image(image_bytes):
    if not image_bytes:
        raise ValueError("The uploaded image is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise RequestEntityTooLarge()

    try:
        with Image.open(BytesIO(image_bytes)) as image_file:
            if image_file.format not in {"JPEG", "PNG"}:
                raise ValueError("Only JPEG and PNG images are supported.")
            width, height = image_file.size
            if (
                width <= 0
                or height <= 0
                or width > MAX_IMAGE_DIMENSION
                or height > MAX_IMAGE_DIMENSION
                or width * height > MAX_IMAGE_PIXELS
            ):
                raise ValueError("The image dimensions are too large.")
            image_file.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("The uploaded file is not a valid image.") from error

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file is not a valid image.")
    return image


def _read_uploaded_image(file_storage, check_extension=False):
    if file_storage is None or not file_storage.filename:
        raise ValueError("Choose an image to upload.")
    if check_extension:
        extension = Path(file_storage.filename).suffix.lower().lstrip(".")
        if extension not in ALLOWED_EXTENSIONS:
            raise ValueError("Only JPG, JPEG, and PNG files are supported.")
    return _decode_image(file_storage.read())


def _read_image_from_request():
    if request.files:
        uploaded_file = request.files.get("image") or request.files.get("file")
        if uploaded_file is None:
            uploaded_file = next(iter(request.files.values()), None)
        return _read_uploaded_image(uploaded_file)

    image_data = None
    if request.is_json:
        payload = request.get_json(silent=True)
        if isinstance(payload, dict):
            image_data = (
                payload.get("image")
                or payload.get("imageData")
                or payload.get("image_data")
                or payload.get("frame")
            )
    if image_data is None:
        image_data = (
            request.form.get("image")
            or request.form.get("imageData")
            or request.form.get("image_data")
            or request.form.get("frame")
        )
    if not isinstance(image_data, str) or not image_data.strip():
        raise ValueError("No image data received.")

    if image_data.startswith("data:"):
        header, separator, image_data = image_data.partition(",")
        if not separator or header not in {
            "data:image/jpeg;base64",
            "data:image/png;base64",
        }:
            raise ValueError("Only JPEG and PNG images are supported.")
    try:
        image_bytes = base64.b64decode(image_data, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Invalid base64 image data.") from error
    return _decode_image(image_bytes)


def _face_result(face):
    scores = face.get("emotion") or {}
    normalized_scores = {
        label: float(
            next(
                (
                    value
                    for key, value in scores.items()
                    if str(key).lower() == label
                ),
                0.0,
            )
        )
        for label in EMOTION_LABELS
    }
    dominant = str(face.get("dominant_emotion") or "Unknown").lower()
    if dominant not in EMOTION_LABELS:
        dominant = max(normalized_scores, key=normalized_scores.get)

    return {
        "emotion": dominant,
        "confidence": normalized_scores[dominant],
        "emotions": normalized_scores,
        "region": face.get("region") or {},
    }


def _analyze_image(image):
    result = _get_deepface().analyze(
        img_path=image,
        actions=["emotion"],
        detector_backend="opencv",
        enforce_detection=True,
        silent=True,
    )
    raw_faces = result if isinstance(result, list) else [result]
    faces = [_face_result(face) for face in raw_faces if isinstance(face, dict)]
    if not faces:
        raise ValueError("No face detected. Try an image with a clear, visible face.")

    primary = faces[0]
    return {
        "success": True,
        "emotion": primary["emotion"],
        "confidence": primary["confidence"],
        "emotions": primary["emotions"],
        "face_count": len(faces),
        "faces": faces,
    }


def _login_required():
    return bool(session.get("logged_in"))


@app.errorhandler(RequestEntityTooLarge)
def handle_large_request(_error):
    return _json_error("Image exceeds the 8 MB upload limit.", 413)


@app.route("/", methods=["GET"])
def index():
    if _login_required():
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username", "")
    password = request.form.get("password", "")
    configured_username = app.config["APP_USERNAME"]
    configured_password = app.config["APP_PASSWORD"]

    if not configured_username or not configured_password:
        return render_template(
            "login.html",
            error="Login is not configured. Set APP_USERNAME and APP_PASSWORD in .env.",
        ), 503

    if secrets.compare_digest(username, configured_username) and secrets.compare_digest(
        password, configured_password
    ):
        session.clear()
        session.permanent = True
        session["logged_in"] = True
        return redirect(url_for("dashboard"))

    return render_template("login.html", error="Invalid username or password."), 401


@app.route("/dashboard", methods=["GET"])
def dashboard():
    if not _login_required():
        return redirect(url_for("index"))
    return render_template("dashboard.html")


@app.route("/emotion", methods=["GET"])
def emotion():
    if not _login_required():
        return redirect(url_for("index"))
    return render_template("emotion.html")


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("index"))


def _detect_response():
    if not _login_required():
        return _json_error("Authentication required.", 401)
    try:
        image = _read_image_from_request()
    except ValueError as error:
        return _json_error(str(error), 400)
    except RequestEntityTooLarge:
        return _json_error("Image exceeds the 8 MB upload limit.", 413)

    try:
        return jsonify(_analyze_image(image))
    except ValueError as error:
        logger.info("Emotion analysis could not detect a face: %s", error)
        return _json_error(
            "No face detected. Try an image with a clear, visible face.", 422
        )
    except Exception:
        logger.exception("Emotion analysis failed.")
        return _json_error(
            "Emotion analysis is temporarily unavailable. Please try again.", 503
        )


@app.route("/upload", methods=["POST"])
def upload():
    if not _login_required():
        return _json_error("Authentication required.", 401)
    uploaded_file = request.files.get("image")
    try:
        image = _read_uploaded_image(uploaded_file, check_extension=True)
    except ValueError as error:
        return _json_error(str(error), 400)
    except RequestEntityTooLarge:
        return _json_error("Image exceeds the 8 MB upload limit.", 413)

    try:
        return jsonify(_analyze_image(image))
    except ValueError as error:
        logger.info("Uploaded image did not contain a detectable face: %s", error)
        return _json_error(
            "No face detected. Try an image with a clear, visible face.", 422
        )
    except Exception:
        logger.exception("Uploaded image analysis failed.")
        return _json_error(
            "Emotion analysis is temporarily unavailable. Please try again.", 503
        )


@app.route("/detect", methods=["POST"])
@app.route("/camera_detect", methods=["POST"])
def detect():
    return _detect_response()


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"},
    )
