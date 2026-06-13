import pandas as pd

from app.service.utils import (
    deduplicate_nearby_notes,
    get_beginner_position,
    preprocess_note_events,
    transpose_song_to_beginner,
)

EXPECTED_COLUMNS = [
    "pitch", "s_dur", "diff_prev", "diff_next", "s_on", "s_off", "s_dur_raw"
]


def test_deduplicate_nearby_notes_empty_dataframe_returns_empty_dataframe():
    df = pd.DataFrame(columns=["pitch", "s_on", "s_off"])

    result = deduplicate_nearby_notes(df)

    assert result.empty
    assert list(result.columns) == ["pitch", "s_on", "s_off"]


def test_deduplicate_nearby_notes_removes_same_pitch_close_onset():
    df = pd.DataFrame([
        {"pitch": 60, "s_on": 0.00, "s_off": 0.40},
        {"pitch": 60, "s_on": 0.02, "s_off": 0.42},
        {"pitch": 60, "s_on": 0.20, "s_off": 0.55},
    ])

    result = deduplicate_nearby_notes(df, time_threshold=0.03)

    assert len(result) == 2
    assert list(result["s_on"]) == [0.00, 0.20]


def test_deduplicate_nearby_notes_keeps_different_pitches_with_close_onset():
    df = pd.DataFrame([
        {"pitch": 60, "s_on": 0.00, "s_off": 0.40},
        {"pitch": 64, "s_on": 0.01, "s_off": 0.42},
    ])

    result = deduplicate_nearby_notes(df, time_threshold=0.03)

    assert len(result) == 2
    assert list(result["pitch"]) == [60, 64]


def test_deduplicate_nearby_notes_keeps_same_pitch_when_onsets_are_far_enough():
    df = pd.DataFrame([
        {"pitch": 60, "s_on": 0.00, "s_off": 0.30},
        {"pitch": 60, "s_on": 0.10, "s_off": 0.40},
    ])

    result = deduplicate_nearby_notes(df, time_threshold=0.03)

    assert len(result) == 2


def test_preprocess_note_events_empty_input_returns_expected_columns():
    df = preprocess_note_events([])

    assert df.empty
    assert list(df.columns) == EXPECTED_COLUMNS


def test_preprocess_note_events_keeps_valid_guitar_notes():
    note_events = [
        [0.00, 0.50, 64, 0.8, None],
        [0.60, 1.00, 67, 0.8, None],
    ]

    df = preprocess_note_events(note_events)

    assert len(df) == 2
    assert list(df["pitch"]) == [64, 67]
    assert list(df.columns) == EXPECTED_COLUMNS


def test_preprocess_note_events_filters_notes_under_guitar_range():
    note_events = [
        [0.00, 0.50, 30, 0.8, None],
        [0.60, 1.00, 64, 0.8, None],
    ]

    df = preprocess_note_events(note_events)

    assert len(df) == 1
    assert int(df.iloc[0]["pitch"]) == 64


def test_preprocess_note_events_filters_notes_above_guitar_range():
    note_events = [
        [0.00, 0.50, 64, 0.8, None],
        [0.60, 1.00, 100, 0.8, None],
    ]

    df = preprocess_note_events(note_events)

    assert len(df) == 1
    assert int(df.iloc[0]["pitch"]) == 64


def test_preprocess_note_events_filters_very_short_notes():
    note_events = [
        [0.00, 0.03, 64, 0.8, None],
        [0.10, 0.50, 67, 0.8, None],
    ]

    df = preprocess_note_events(note_events)

    assert len(df) == 1
    assert int(df.iloc[0]["pitch"]) == 67


def test_preprocess_note_events_removes_exact_duplicates():
    note_events = [
        [0.00, 0.50, 64, 0.8, None],
        [0.00, 0.50, 64, 0.8, None],
    ]

    df = preprocess_note_events(note_events)

    assert len(df) == 1
    assert int(df.iloc[0]["pitch"]) == 64


def test_preprocess_note_events_normalizes_duration_by_four():
    note_events = [[0.00, 2.00, 64, 0.8, None]]

    df = preprocess_note_events(note_events)

    assert float(df.iloc[0]["s_dur_raw"]) == 2.0
    assert float(df.iloc[0]["s_dur"]) == 0.5


def test_preprocess_note_events_clips_long_duration_to_one():
    note_events = [[0.00, 8.00, 64, 0.8, None]]

    df = preprocess_note_events(note_events)

    assert float(df.iloc[0]["s_dur_raw"]) == 8.0
    assert float(df.iloc[0]["s_dur"]) == 1.0


def test_preprocess_note_events_calculates_pitch_differences():
    note_events = [
        [0.00, 0.50, 64, 0.8, None],
        [0.60, 1.00, 67, 0.8, None],
    ]

    df = preprocess_note_events(note_events)
    expected_diff = (67 / 127.0) - (64 / 127.0)

    assert abs(float(df.iloc[0]["diff_next"]) - expected_diff) < 1e-6
    assert abs(float(df.iloc[1]["diff_prev"]) - expected_diff) < 1e-6


def test_get_beginner_position_exact_pitch_in_first_four_frets():
    string_index, fret = get_beginner_position(64)

    assert string_index == 0
    assert fret == 0


def test_get_beginner_position_transposes_by_pitch_class_when_needed():
    string_index, fret = get_beginner_position(76)

    assert string_index is not None
    assert fret is not None
    assert 0 <= fret <= 4


def test_transpose_song_to_beginner_returns_one_position_per_pitch():
    pitches = [64, 67, 72, 76]

    strings, frets = transpose_song_to_beginner(pitches)

    assert len(strings) == len(pitches)
    assert len(frets) == len(pitches)
    assert all(fret is None or 0 <= fret <= 4 for fret in frets)
