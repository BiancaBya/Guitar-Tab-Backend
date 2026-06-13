import os

import numpy as np
import pandas as pd

from app.domain import models
from app.controller import main as main_module


class DummyModel:
    def to(self, device):
        return self

    def eval(self):
        return self


def _mock_audio_dataframe():
    return pd.DataFrame([
        {
            "pitch": 64,
            "s_dur": 0.125,
            "diff_prev": 0.0,
            "diff_next": 0.02,
            "s_on": 0.0,
            "s_off": 0.5,
            "s_dur_raw": 0.5,
        },
        {
            "pitch": 67,
            "s_dur": 0.125,
            "diff_prev": 0.02,
            "diff_next": 0.0,
            "s_on": 0.5,
            "s_off": 1.0,
            "s_dur_raw": 0.5,
        },
    ])


def _mock_logits(*args, **kwargs):
    return np.array([
        [8, 1, 1, 1, 1, 1],
        [1, 8, 1, 1, 1, 1],
    ], dtype=np.float32)


def _post_fake_audio(client, auth_headers, tmp_path, filename="sample.wav"):
    audio_path = tmp_path / filename
    audio_path.write_bytes(b"fake audio content")

    with audio_path.open("rb") as f:
        return client.post(
            "/predict-tab/",
            files={"file": (filename, f, "audio/wav")},
            headers=auth_headers,
        )


def test_predict_tab_success_saves_original_and_beginner_versions(
    client,
    test_db,
    registered_user,
    auth_headers,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(main_module, "audio_to_model_input", lambda path: _mock_audio_dataframe())
    monkeypatch.setattr(main_module, "predict_logits_windowed", _mock_logits)
    monkeypatch.setattr(main_module, "model", DummyModel())

    response = _post_fake_audio(client, auth_headers, tmp_path, "sample.wav")

    body = response.json()
    assert response.status_code == 200
    assert body["filename"] == "sample.wav"
    assert body["message"] == "Tablatures generated and saved successfully!"
    assert len(body["tablature"]) == 2
    assert len(body["tablature_beginner"]) == 2
    assert body["tablature"][0]["string"] == 1
    assert body["tablature"][0]["fret"] == 0

    user = test_db.query(models.User).filter(models.User.username == registered_user["username"]).first()
    saved_tabs = test_db.query(models.Tablature).filter(models.Tablature.user_id == user.id).all()
    assert len(saved_tabs) == 1
    assert saved_tabs[0].filename == "sample.wav"


def test_predict_tab_response_contains_id_and_expected_fields(client, auth_headers, monkeypatch, tmp_path):
    monkeypatch.setattr(main_module, "audio_to_model_input", lambda path: _mock_audio_dataframe())
    monkeypatch.setattr(main_module, "predict_logits_windowed", _mock_logits)

    response = _post_fake_audio(client, auth_headers, tmp_path, "fields.wav")

    body = response.json()
    assert response.status_code == 200
    assert isinstance(body["id"], int)
    assert set(["time", "duration", "pitch", "string", "fret"]).issubset(body["tablature"][0].keys())
    assert set(["time", "duration", "pitch", "string", "fret"]).issubset(body["tablature_beginner"][0].keys())


def test_predict_tab_saves_result_visible_in_history(client, auth_headers, monkeypatch, tmp_path):
    monkeypatch.setattr(main_module, "audio_to_model_input", lambda path: _mock_audio_dataframe())
    monkeypatch.setattr(main_module, "predict_logits_windowed", _mock_logits)

    predict_response = _post_fake_audio(client, auth_headers, tmp_path, "history_sample.wav")
    history_response = client.get("/history", headers=auth_headers)

    assert predict_response.status_code == 200
    assert history_response.status_code == 200
    assert len(history_response.json()) == 1
    assert history_response.json()[0]["filename"] == "history_sample.wav"


def test_predict_tab_temporary_uploaded_file_is_removed(client, auth_headers, monkeypatch, tmp_path):
    monkeypatch.setattr(main_module, "audio_to_model_input", lambda path: _mock_audio_dataframe())
    monkeypatch.setattr(main_module, "predict_logits_windowed", _mock_logits)

    response = _post_fake_audio(client, auth_headers, tmp_path, "temporary.wav")

    assert response.status_code == 200
    assert not os.path.exists(os.path.join(main_module.UPLOAD_DIR, "temporary.wav"))


def test_predict_tab_no_notes_detected_returns_400(client, auth_headers, monkeypatch, tmp_path):
    monkeypatch.setattr(main_module, "audio_to_model_input", lambda path: pd.DataFrame())

    response = _post_fake_audio(client, auth_headers, tmp_path, "empty.wav")

    assert response.status_code == 400
    assert response.json()["detail"] == "No notes detected in audio."


def test_predict_tab_basic_pitch_failure_returns_400(client, auth_headers, monkeypatch, tmp_path):
    monkeypatch.setattr(main_module, "audio_to_model_input", lambda path: None)

    response = _post_fake_audio(client, auth_headers, tmp_path, "invalid.wav")

    assert response.status_code == 400
    assert response.json()["detail"] == "No notes detected in audio."


def test_predict_tab_internal_error_returns_500(client, auth_headers, monkeypatch, tmp_path):
    def raise_error(path):
        raise RuntimeError("processing failed")

    monkeypatch.setattr(main_module, "audio_to_model_input", raise_error)

    response = _post_fake_audio(client, auth_headers, tmp_path, "broken.wav")

    assert response.status_code == 500
    assert "processing failed" in response.json()["detail"]


def test_predict_tab_handles_invalid_decoded_string_by_falling_back_to_feasible_string(
    client,
    auth_headers,
    monkeypatch,
    tmp_path,
):
    df = pd.DataFrame([
        {
            "pitch": 40,
            "s_dur": 0.125,
            "diff_prev": 0.0,
            "diff_next": 0.0,
            "s_on": 0.0,
            "s_off": 0.5,
            "s_dur_raw": 0.5,
        }
    ])
    monkeypatch.setattr(main_module, "audio_to_model_input", lambda path: df)
    monkeypatch.setattr(
        main_module,
        "predict_logits_windowed",
        lambda model, pitch_arr, rest_arr, device: np.array([[1, 1, 1, 1, 1, 9]], dtype=np.float32),
    )
    monkeypatch.setattr(main_module, "decode_with_viterbi", lambda logits, pitches: [0])

    response = _post_fake_audio(client, auth_headers, tmp_path, "fallback.wav")

    body = response.json()
    assert response.status_code == 200
    assert body["tablature"][0]["string"] == 6
    assert body["tablature"][0]["fret"] == 0


def test_predict_tab_allows_beginner_entry_without_position(client, auth_headers, monkeypatch, tmp_path):
    df = pd.DataFrame([
        {
            "pitch": 64,
            "s_dur": 0.125,
            "diff_prev": 0.0,
            "diff_next": 0.0,
            "s_on": 0.0,
            "s_off": 0.5,
            "s_dur_raw": 0.5,
        }
    ])
    monkeypatch.setattr(main_module, "audio_to_model_input", lambda path: df)
    monkeypatch.setattr(
        main_module,
        "predict_logits_windowed",
        lambda model, pitch_arr, rest_arr, device: np.array([[9, 1, 1, 1, 1, 1]], dtype=np.float32),
    )
    monkeypatch.setattr(main_module, "transpose_song_to_beginner", lambda pitches: ([None], [None]))

    response = _post_fake_audio(client, auth_headers, tmp_path, "no_beginner.wav")

    body = response.json()
    assert response.status_code == 200
    assert body["tablature_beginner"][0]["string"] is None
    assert body["tablature_beginner"][0]["fret"] is None
