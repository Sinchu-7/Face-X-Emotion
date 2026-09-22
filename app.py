from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from deepface import DeepFace
import cv2
import os
import base64
import numpy as np

app = Flask(__name__)

app.secret_key = "facial_emotion_secret_key"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# CONVERT NUMPY VALUES
# ============================================================

def convert_numpy(obj):

    if isinstance(obj, dict):
        return {
            str(key): convert_numpy(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [convert_numpy(value) for value in obj]

    if isinstance(obj, tuple):
        return [convert_numpy(value) for value in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    return obj


# ============================================================
# LOGIN PAGE
# ============================================================

@app.route("/")
def index():
    return render_template("login.html")


# ============================================================
# LOGIN
# ============================================================

@app.route("/login", methods=["POST"])
def login():

    username = request.form.get("username", "")
    password = request.form.get("password", "")

    if username == "admin" and password == "1234":

        session["logged_in"] = True

        return redirect(url_for("emotion"))

    return render_template(
        "login.html",
        error="Invalid username or password"
    )


# ============================================================
# EMOTION PAGE
# ============================================================

@app.route("/emotion")
def emotion():

    if not session.get("logged_in"):
        return redirect(url_for("index"))

    return render_template("emotion.html")


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("index"))


# ============================================================
# EMOTION DETECTION
#
# Accept BOTH:
#
# /detect
# /camera_detect
#
# ============================================================

@app.route("/detect", methods=["POST"])
@app.route("/camera_detect", methods=["POST"])
def detect():

    try:

        print("\n======================================")
        print("EMOTION DETECTION REQUEST")
        print("======================================")

        print("Content-Type:", request.content_type)

        image_data = None

        # ====================================================
        # METHOD 1: JSON
        # ====================================================

        if request.is_json:

            data = request.get_json(silent=True)

            print("JSON received:", bool(data))

            if data:

                image_data = (
                    data.get("image")
                    or data.get("imageData")
                    or data.get("image_data")
                    or data.get("frame")
                )


        # ====================================================
        # METHOD 2: FORM DATA
        # ====================================================

        if image_data is None:

            image_data = (
                request.form.get("image")
                or request.form.get("imageData")
                or request.form.get("image_data")
                or request.form.get("frame")
            )

            if image_data:

                print("Image received through form data")


        # ====================================================
        # METHOD 3: UPLOADED FILE
        # ====================================================

        if image_data is None and request.files:

            print("Uploaded files:", list(request.files.keys()))

            uploaded_file = None

            if "image" in request.files:
                uploaded_file = request.files["image"]

            elif "file" in request.files:
                uploaded_file = request.files["file"]

            else:
                uploaded_file = next(
                    iter(request.files.values()),
                    None
                )

            if uploaded_file:

                image_bytes = uploaded_file.read()

                np_array = np.frombuffer(
                    image_bytes,
                    np.uint8
                )

                image = cv2.imdecode(
                    np_array,
                    cv2.IMREAD_COLOR
                )

                if image is not None:

                    print("Image received as uploaded file")

                else:

                    return jsonify({
                        "success": False,
                        "error": "Uploaded file is not a valid image"
                    }), 400


        # ====================================================
        # NO IMAGE FOUND
        # ====================================================

        if image_data is None and "image" not in locals():

            print("ERROR: No image data received")

            return jsonify({
                "success": False,
                "error": "No image data received"
            }), 400


        # ====================================================
        # DECODE BASE64 IMAGE
        # ====================================================

        if image_data is not None:

            print("Image data received")

            # Make sure it is a string
            if isinstance(image_data, bytes):
                image_data = image_data.decode("utf-8")

            image_data = str(image_data)

            # Remove data:image/jpeg;base64,...
            if "," in image_data:

                image_data = image_data.split(",", 1)[1]

            try:

                image_bytes = base64.b64decode(
                    image_data
                )

            except Exception as e:

                print("Base64 decode error:", e)

                return jsonify({
                    "success": False,
                    "error": "Invalid base64 image"
                }), 400


            np_array = np.frombuffer(
                image_bytes,
                np.uint8
            )

            image = cv2.imdecode(
                np_array,
                cv2.IMREAD_COLOR
            )


        # ====================================================
        # CHECK IMAGE
        # ====================================================

        if image is None:

            print("ERROR: Image could not be decoded")

            return jsonify({
                "success": False,
                "error": "Could not decode camera image"
            }), 400


        print("Image successfully decoded")
        print("Image shape:", image.shape)


        # ====================================================
        # SAVE IMAGE
        # ====================================================

        image_path = os.path.join(
            UPLOAD_FOLDER,
            "captured_image.jpg"
        )

        cv2.imwrite(
            image_path,
            image
        )

        print("Image saved:", image_path)


        # ====================================================
        # RUN DEEPFACE
        # ====================================================

        print("Running DeepFace...")

        result = DeepFace.analyze(
            img_path=image_path,
            actions=["emotion"],
            enforce_detection=False
        )

        print("DeepFace finished")


        # ====================================================
        # HANDLE LIST RESULT
        # ====================================================

        if isinstance(result, list):

            if len(result) == 0:

                return jsonify({
                    "success": False,
                    "error": "No face detected"
                }), 400

            result = result[0]


        # ====================================================
        # GET EMOTIONS
        # ====================================================

        emotions = result.get(
            "emotion",
            {}
        )


        # ====================================================
        # GET DOMINANT EMOTION
        # ====================================================

        dominant_emotion = result.get(
            "dominant_emotion",
            "Unknown"
        )


        # ====================================================
        # GET CONFIDENCE
        # ====================================================

        confidence = emotions.get(
            dominant_emotion,
            0
        )

        confidence = float(confidence)


        # ====================================================
        # CONVERT NUMPY TYPES
        # ====================================================

        emotions = convert_numpy(emotions)
        result = convert_numpy(result)


        # ====================================================
        # PRINT RESULT
        # ====================================================

        print("")
        print("RESULT")
        print("--------------------------------------")
        print("Emotion:", dominant_emotion)
        print("Confidence:", confidence)
        print("--------------------------------------")


        # ====================================================
        # RETURN RESULT
        # ====================================================

        return jsonify({
            "success": True,
            "emotion": str(dominant_emotion),
            "confidence": confidence,
            "emotions": emotions,
            "details": result
        })


    except Exception as e:

        print("")
        print("======================================")
        print("DETECTION ERROR")
        print("======================================")
        print(str(e))
        print("======================================")

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print("")
    print("======================================")
    print(" FACIAL EMOTION RECOGNITION SYSTEM")
    print("======================================")
    print("")
    print("Website:")
    print("http://127.0.0.1:5000")
    print("")
    print("Demo Login:")
    print("Username: admin")
    print("Password: 1234")
    print("")
    print("======================================")
    print("")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )