import cv2
from deepface import DeepFace

# ==========================================
# START LAPTOP CAMERA
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open laptop camera.")
    exit()

print("===================================")
print("FACIAL EMOTION RECOGNITION")
print("===================================")
print("Camera started!")
print("Press Q or ESC to quit.")
print()
print("First run may take some time while")
print("DeepFace downloads its pretrained model.")


# ==========================================
# CAMERA LOOP
# ==========================================

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    # Flip camera like a mirror
    frame = cv2.flip(frame, 1)

    try:

        # ==================================
        # EMOTION ANALYSIS
        # ==================================

        result = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=False,
            silent=True
        )

        # DeepFace may return a list
        if isinstance(result, list):
            result = result[0]

        # ==================================
        # GET FACE REGION
        # ==================================

        region = result.get("region", {})

        x = region.get("x", 0)
        y = region.get("y", 0)
        w = region.get("w", 0)
        h = region.get("h", 0)

        # ==================================
        # GET EMOTION
        # ==================================

        emotion = result.get(
            "dominant_emotion",
            "Unknown"
        )

        emotion_scores = result.get(
            "emotion",
            {}
        )

        confidence = emotion_scores.get(
            emotion,
            0
        )

        # ==================================
        # DRAW FACE BOX
        # ==================================

        if w > 0 and h > 0:

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # ==================================
            # DISPLAY EMOTION
            # ==================================

            text = f"{emotion.upper()} ({confidence:.1f}%)"

            cv2.putText(
                frame,
                text,
                (x, max(y - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    except Exception as e:

        # Keep camera running even if
        # DeepFace temporarily fails
        pass


    # ==========================================
    # SHOW CAMERA
    # ==========================================

    cv2.imshow(
        "Facial Emotion Recognition",
        frame
    )


    # ==========================================
    # KEYBOARD
    # ==========================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("Q") or key == 27:
        break


# ==========================================
# CLOSE CAMERA
# ==========================================

camera.release()

cv2.destroyAllWindows()

print("Camera stopped.")
print("Program ended.")