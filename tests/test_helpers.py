import json

import numpy as np
import pytest
import torch
from fastapi import HTTPException

from app import models
from app.main import (
    build_export_filename,
    decode_with_viterbi,
    feasible_string_indices,
    get_tablature_or_404,
    parse_saved_tablature,
    predict_logits_windowed,
)


class DummyWindowModel:
    def __call__(self, inputs):
        pitch_tensor, _ = inputs
        batch_size, seq_len = pitch_tensor.shape
        logits = torch.zeros((batch_size, 6, seq_len), dtype=torch.float32, device=pitch_tensor.device)
        logits[:, 0, :] = 5.0
        return logits


def test_feasible_string_indices_returns_valid_strings_for_pitch():
    result = feasible_string_indices(64)

    assert 0 in result
    assert all(0 <= string_index <= 5 for string_index in result)


def test_feasible_string_indices_returns_empty_for_unplayable_pitch():
    result = feasible_string_indices(20)

    assert result == []


def test_feasible_string_indices_low_e_pitch_is_playable_on_sixth_string():
    result = feasible_string_indices(40)

    assert result == [5]


def test_decode_with_viterbi_returns_path_with_same_length_as_pitches():
    pitches = np.array([64, 66, 67], dtype=np.int64)
    logits = np.array([
        [5, 1, 1, 1, 1, 1],
        [5, 1, 1, 1, 1, 1],
        [5, 1, 1, 1, 1, 1],
    ], dtype=np.float32)

    path = decode_with_viterbi(logits, pitches)

    assert len(path) == len(pitches)
    assert all(0 <= string_index <= 5 for string_index in path)


def test_decode_with_viterbi_empty_input_returns_empty_list():
    path = decode_with_viterbi(np.zeros((0, 6), dtype=np.float32), [])

    assert path == []


def test_decode_with_viterbi_single_pitch_returns_one_string():
    pitches = np.array([40], dtype=np.int64)
    logits = np.array([[1, 1, 1, 1, 1, 9]], dtype=np.float32)

    path = decode_with_viterbi(logits, pitches)

    assert path == [5]


def test_decode_with_viterbi_respects_feasible_strings_even_if_logits_prefer_invalid_string():
    pitches = np.array([40], dtype=np.int64)
    logits = np.array([[99, 99, 99, 99, 99, 1]], dtype=np.float32)

    path = decode_with_viterbi(logits, pitches)

    assert path == [5]


def test_predict_logits_windowed_empty_input_returns_empty_array():
    result = predict_logits_windowed(
        DummyWindowModel(),
        np.array([], dtype=np.int64),
        np.zeros((0, 3), dtype=np.float32),
        torch.device("cpu"),
    )

    assert result.shape == (0, 6)


def test_predict_logits_windowed_short_sequence_returns_original_length():
    pitches = np.array([64, 65, 67], dtype=np.int64)
    rest = np.zeros((3, 3), dtype=np.float32)

    result = predict_logits_windowed(DummyWindowModel(), pitches, rest, torch.device("cpu"))

    assert result.shape == (3, 6)
    assert np.all(result[:, 0] == 5.0)


def test_predict_logits_windowed_long_sequence_returns_original_length():
    pitches = np.array([64] * 80, dtype=np.int64)
    rest = np.zeros((80, 3), dtype=np.float32)

    result = predict_logits_windowed(DummyWindowModel(), pitches, rest, torch.device("cpu"))

    assert result.shape == (80, 6)
    assert np.all(result[:, 0] == 5.0)


def test_build_export_filename_sanitizes_filename():
    filename = build_export_filename("my song final!.wav", "original")

    assert filename == "my_song_final_original_tab.pdf"


def test_build_export_filename_uses_default_name_for_empty_filename():
    filename = build_export_filename("", "beginner")

    assert filename == "tablature_beginner_tab.pdf"


def test_build_export_filename_removes_folder_like_characters():
    filename = build_export_filename("folder/name song.wav", "original")

    assert filename == "folder_name_song_original_tab.pdf"


def test_parse_saved_tablature_valid_json():
    tab = models.Tablature(
        filename="song.wav",
        json_content=json.dumps({"tablature": [], "tablature_beginner": []}),
        user_id=1,
    )

    parsed = parse_saved_tablature(tab)

    assert parsed == {"tablature": [], "tablature_beginner": []}


def test_parse_saved_tablature_empty_content_returns_empty_dict():
    tab = models.Tablature(filename="song.wav", json_content="", user_id=1)

    parsed = parse_saved_tablature(tab)

    assert parsed == {}


def test_parse_saved_tablature_invalid_json_raises_http_500():
    tab = models.Tablature(
        filename="song.wav",
        json_content="not-json",
        user_id=1,
    )

    with pytest.raises(HTTPException) as exc:
        parse_saved_tablature(tab)

    assert exc.value.status_code == 500


def test_get_tablature_or_404_returns_only_current_user_tab(test_db):
    user = models.User(username="user1", password_hash="hash")
    tab = models.Tablature(filename="song.wav", json_content="{}", owner=user)
    test_db.add(user)
    test_db.add(tab)
    test_db.commit()
    test_db.refresh(user)
    test_db.refresh(tab)

    result = get_tablature_or_404(tab.id, test_db, user)

    assert result.id == tab.id


def test_get_tablature_or_404_rejects_foreign_tab(test_db):
    owner = models.User(username="owner", password_hash="hash")
    other = models.User(username="other", password_hash="hash")
    tab = models.Tablature(filename="song.wav", json_content="{}", owner=owner)
    test_db.add_all([owner, other, tab])
    test_db.commit()
    test_db.refresh(other)
    test_db.refresh(tab)

    with pytest.raises(HTTPException) as exc:
        get_tablature_or_404(tab.id, test_db, other)

    assert exc.value.status_code == 404


def test_get_tablature_or_404_rejects_missing_tab(test_db):
    user = models.User(username="user1", password_hash="hash")
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)

    with pytest.raises(HTTPException) as exc:
        get_tablature_or_404(9999, test_db, user)

    assert exc.value.status_code == 404
