import base64
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

import app as application


@pytest.fixture
def client(monkeypatch):
    application.app.config.update(
        TESTING=True,
        SECRET_KEY="test-only-secret",
        APP_USERNAME="test-user",
        APP_PASSWORD="test-password",
    )
    with application.app.test_client() as test_client:
        yield test_client


def sign_in(client):
    return client.post(
        "/login",
        data={"username": "test-user", "password": "test-password"},
        follow_redirects=False,
    )


def png_bytes():
    image = Image.new("RGB", (48, 48), color=(120, 80, 60))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def analysis_result(face_count=1):
    scores = {
        "angry": 1.0,
        "disgust": 2.0,
        "fear": 3.0,
        "happy": 80.0,
        "sad": 4.0,
        "surprise": 5.0,
        "neutral": 5.0,
    }
    return {
        "success": True,
        "emotion": "happy",
        "confidence": 80.0,
        "emotions": scores,
        "face_count": face_count,
        "faces": [
            {
                "emotion": "happy",
                "confidence": 80.0,
                "emotions": scores,
                "region": {"x": index, "y": 2, "w": 30, "h": 30},
            }
            for index in range(face_count)
        ],
    }


def test_app_startup_and_public_login_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Welcome back" in response.data


def test_dashboard_and_emotion_pages_require_login(client):
    assert client.get("/dashboard").status_code == 302
    assert client.get("/emotion").status_code == 302


def test_login_rejects_invalid_credentials(client):
    response = client.post(
        "/login",
        data={"username": "test-user", "password": "wrong"},
    )
    assert response.status_code == 401
    assert b"Invalid username or password" in response.data


def test_login_dashboard_and_logout(client):
    response = sign_in(client)
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    assert client.get("/dashboard").status_code == 200
    assert client.get("/emotion").status_code == 200

    response = client.post("/logout")
    assert response.status_code == 302
    assert client.get("/dashboard").status_code == 302


def test_dashboard_links_to_camera_and_upload(client):
    sign_in(client)
    response = client.get("/dashboard")
    assert b"#camera-panel" in response.data
    assert b"#upload-panel" in response.data


def test_authenticated_upload_returns_emotion_result(client, monkeypatch):
    sign_in(client)
    monkeypatch.setattr(application, "_analyze_image", Mock(return_value=analysis_result()))
    response = client.post(
        "/upload",
        data={"image": (BytesIO(png_bytes()), "portrait.png")},
        content_type="multipart/form-data",
    )
    body = response.get_json()
    assert response.status_code == 200
    assert body["success"] is True
    assert body["emotion"] == "happy"
    assert body["confidence"] == 80.0
    assert set(body["emotions"]) == set(application.EMOTION_LABELS)
    assert body["face_count"] == 1


def test_upload_requires_authentication(client):
    response = client.post("/upload")
    assert response.status_code == 401
    assert response.get_json() == {
        "success": False,
        "error": "Authentication required.",
    }


def test_upload_rejects_invalid_extension(client, monkeypatch):
    sign_in(client)
    analyze = Mock()
    monkeypatch.setattr(application, "_analyze_image", analyze)
    response = client.post(
        "/upload",
        data={"image": (BytesIO(png_bytes()), "portrait.gif")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "Only JPG, JPEG, and PNG" in response.get_json()["error"]
    analyze.assert_not_called()


def test_upload_rejects_corrupted_image(client, monkeypatch):
    sign_in(client)
    response = client.post(
        "/upload",
        data={"image": (BytesIO(b"not an image"), "portrait.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert response.get_json()["success"] is False


def test_upload_requires_a_file(client):
    sign_in(client)
    response = client.post("/upload", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "Choose an image" in response.get_json()["error"]


def test_camera_endpoint_decodes_base64_and_returns_all_scores(client, monkeypatch):
    sign_in(client)
    analyze = Mock(return_value=analysis_result())
    monkeypatch.setattr(application, "_analyze_image", analyze)
    encoded = base64.b64encode(png_bytes()).decode("ascii")
    response = client.post("/detect", json={"image": encoded})
    body = response.get_json()
    assert response.status_code == 200
    assert body["emotion"] == "happy"
    assert set(body["emotions"]) == set(application.EMOTION_LABELS)
    assert analyze.call_args.args[0].shape == (48, 48, 3)


def test_camera_endpoint_rejects_missing_or_invalid_image(client):
    sign_in(client)
    missing = client.post("/camera_detect", json={})
    invalid = client.post("/detect", json={"image": "not-base64!"})
    assert missing.status_code == 400
    assert missing.get_json()["success"] is False
    assert invalid.status_code == 400
    assert "base64" in invalid.get_json()["error"]


def test_analysis_reports_each_face_and_all_emotion_scores(monkeypatch):
    scores = {label: float(index) for index, label in enumerate(application.EMOTION_LABELS)}
    deepface = SimpleNamespace(
        analyze=Mock(
            return_value=[
                {"dominant_emotion": "happy", "emotion": scores, "region": {"x": 1}},
                {"dominant_emotion": "sad", "emotion": scores, "region": {"x": 50}},
            ]
        )
    )
    monkeypatch.setattr(application, "_get_deepface", lambda: deepface)

    result = application._analyze_image(application._decode_image(png_bytes()))

    assert result["face_count"] == 2
    assert result["emotion"] == "happy"
    assert len(result["faces"]) == 2
    assert all(set(face["emotions"]) == set(application.EMOTION_LABELS) for face in result["faces"])
    deepface.analyze.assert_called_once()
    assert deepface.analyze.call_args.kwargs["enforce_detection"] is True


def test_no_face_and_internal_analysis_errors_have_safe_api_responses(client, monkeypatch):
    sign_in(client)
    monkeypatch.setattr(
        application,
        "_analyze_image",
        Mock(side_effect=ValueError("No face detected")),
    )
    no_face = client.post(
        "/upload",
        data={"image": (BytesIO(png_bytes()), "portrait.png")},
        content_type="multipart/form-data",
    )
    assert no_face.status_code == 422
    assert "No face detected" in no_face.get_json()["error"]

    monkeypatch.setattr(
        application,
        "_analyze_image",
        Mock(side_effect=RuntimeError("private model details")),
    )
    failed = client.post(
        "/upload",
        data={"image": (BytesIO(png_bytes()), "portrait.png")},
        content_type="multipart/form-data",
    )
    assert failed.status_code == 503
    assert "private model details" not in failed.get_json()["error"]
