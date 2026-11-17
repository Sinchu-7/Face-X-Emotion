# Face-X-Emotion

Face-X-Emotion is an educational Flask application for estimating facial-expression categories from an uploaded image or a browser camera frame. It uses DeepFace for inference. These estimates are not reliable measurements of a person's internal emotional state and should not be used for consequential decisions.

## Requirements

- Windows, macOS, or Linux
- 64-bit Python 3.11 (recommended; the pinned TensorFlow release is tested with Python 3.11)
- A webcam for camera mode
- An internet connection on the first inference run so DeepFace can download its pretrained detector and emotion model, if not already cached

The application pins TensorFlow 2.16.2 (which supplies Keras 3), NumPy 1.26.4, OpenCV 4.10, DeepFace 0.0.93, Flask 3.1, and the supporting packages in `requirements.txt`. The inference path passes decoded OpenCV BGR images directly to DeepFace; it does not load the repository's locally trained CNN artifacts.

## Windows PowerShell setup

From the repository folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

In `.env`, replace the placeholder values for `FLASK_SECRET_KEY`, `APP_USERNAME`, and `APP_PASSWORD` with your own local secret and login credentials. Generate a secret with:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Save `.env` and run the application:

```powershell
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). Sign in with the username and password you set in `.env`; there are no built-in demo credentials.

To run the tests:

```powershell
python -m pytest
```

For browser webcam access, use `http://127.0.0.1:5000` or HTTPS and grant camera permission. Camera inference is an on-demand snapshot, not continuous video analysis. Stop the camera in the page when finished.

## Features

- Environment-configured local login and session-protected dashboard
- Browser webcam snapshot analysis with start/stop controls
- JPEG/PNG image upload with preview and server-side validation
- Seven emotion scores, dominant class, confidence, and per-face results
- JSON API responses for upload and camera requests

The app does not persist accounts or upload images and does not use a database. The login is a single local credential pair read from `.env`; it is not intended to provide multi-user production authentication. Flask's signed session cookie is HTTP-only and same-site. Set `SESSION_COOKIE_SECURE=true` when serving over HTTPS.

## Model files and training

- `model/emotion_model.keras` is the CNN artifact saved by `src/train.py`.
- `model/fer.json` contains a serialized Keras model architecture and `model/fer.h5` is a separate legacy HDF5 model/weights artifact. The repository has no loader that verifies or pairs these two files.
- The web and standalone camera inference use DeepFace's own pretrained emotion model. DeepFace caches its loaded models in-process and downloads model weights to its user cache on first use. The project artifacts above are preserved and are not silently substituted into that inference path.

`src/train.py` expects the existing `dataset/train` and `dataset/test` folders. Run training from the repository root with:

```powershell
python src\train.py
```

Training may take a long time and will overwrite `model/emotion_model.keras`.

## Project layout

```text
app.py                  Flask application and JSON endpoints
src\train.py            Local CNN training pipeline
src\emotion_camera.py   Standalone OpenCV/DeepFace camera demo
templates\              Flask/Jinja pages
model\                  Preserved model artifacts
dataset\train, test\    Seven-class image dataset
tests\                  Flask application tests
requirements.txt        Pinned Python dependencies
.env.example            Safe local configuration template
```
