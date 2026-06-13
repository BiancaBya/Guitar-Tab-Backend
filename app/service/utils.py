import pandas as pd
import numpy as np
from basic_pitch.inference import predict

OPEN_STRING_MIDI = [64, 59, 55, 50, 45, 40]

MIN_GUITAR_MIDI = 40
MAX_GUITAR_MIDI = 88


def deduplicate_nearby_notes(df: pd.DataFrame, time_threshold: float = 0.03) -> pd.DataFrame:
    if df.empty:
        return df

    rows = []
    last_seen_onset_for_pitch = {}

    for _, row in df.iterrows():
        pitch = int(row["pitch"])
        onset = float(row["s_on"])

        if pitch not in last_seen_onset_for_pitch:
            rows.append(row)
            last_seen_onset_for_pitch[pitch] = onset
            continue

        if onset - last_seen_onset_for_pitch[pitch] > time_threshold:
            rows.append(row)
            last_seen_onset_for_pitch[pitch] = onset

    if not rows:
        return pd.DataFrame(columns=df.columns)

    return pd.DataFrame(rows).reset_index(drop=True)


def preprocess_note_events(note_events):
    data = []
    for note in note_events:
        onset = float(note[0])
        offset = float(note[1])
        midi_pitch = int(note[2])

        duration = max(0.0, offset - onset)

        data.append({
            "s_on": onset,
            "s_off": offset,
            "s_dur_raw": duration,
            "pitch": midi_pitch
        })

    if not data:
        return pd.DataFrame(columns=[
            "pitch", "s_dur", "diff_prev", "diff_next", "s_on", "s_off", "s_dur_raw"
        ])

    df = pd.DataFrame(data)
    df = df.sort_values(by="s_on").reset_index(drop=True)

    df = df[(df["pitch"] >= MIN_GUITAR_MIDI) & (df["pitch"] <= MAX_GUITAR_MIDI)].copy()

    if df.empty:
        return pd.DataFrame(columns=[
            "pitch", "s_dur", "diff_prev", "diff_next", "s_on", "s_off", "s_dur_raw"
        ])

    df = df[df["s_dur_raw"] >= 0.05].copy()

    if df.empty:
        return pd.DataFrame(columns=[
            "pitch", "s_dur", "diff_prev", "diff_next", "s_on", "s_off", "s_dur_raw"
        ])

    df = df.drop_duplicates(subset=["s_on", "s_off", "pitch"]).copy()
    df = deduplicate_nearby_notes(df, time_threshold=0.03)

    if df.empty:
        return pd.DataFrame(columns=[
            "pitch", "s_dur", "diff_prev", "diff_next", "s_on", "s_off", "s_dur_raw"
        ])

    df["s_dur"] = np.clip(df["s_dur_raw"].values.astype(np.float32), 0.0, 4.0) / 4.0

    norm_pitch = df["pitch"].values.astype(np.float32) / 127.0

    diff_prev = np.zeros(len(df), dtype=np.float32)
    diff_next = np.zeros(len(df), dtype=np.float32)

    if len(df) > 1:
        diff_prev[1:] = norm_pitch[1:] - norm_pitch[:-1]
        diff_next[:-1] = norm_pitch[1:] - norm_pitch[:-1]

    df["diff_prev"] = diff_prev
    df["diff_next"] = diff_next

    return df[[
        "pitch", "s_dur", "diff_prev", "diff_next",
        "s_on", "s_off", "s_dur_raw"
    ]].reset_index(drop=True)


def audio_to_model_input(audio_path):
    try:
        model_output, midi_data, note_events = predict(
            audio_path,
            onset_threshold=0.6,
            frame_threshold=0.35,
            minimum_note_length=80,
            minimum_frequency=82.41,
            maximum_frequency=1318.51,
            melodia_trick=True
        )
    except Exception as e:
        print(f"Error in Basic Pitch: {e}")
        return None

    return preprocess_note_events(note_events)


def build_beginner_positions(max_fret=4):
    exact_pitch_positions = {}
    pitch_class_positions = {pc: [] for pc in range(12)}

    for s_idx, open_midi in enumerate(OPEN_STRING_MIDI):
        for fret in range(max_fret + 1):
            midi_pitch = open_midi + fret
            pos = (s_idx, fret, midi_pitch)

            if midi_pitch not in exact_pitch_positions:
                exact_pitch_positions[midi_pitch] = []
            exact_pitch_positions[midi_pitch].append(pos)

            pitch_class_positions[midi_pitch % 12].append(pos)

    for midi_pitch in exact_pitch_positions:
        exact_pitch_positions[midi_pitch].sort(key=lambda x: (x[1], x[0]))

    for pc in pitch_class_positions:
        pitch_class_positions[pc].sort(key=lambda x: (x[1], x[0], x[2]))

    return exact_pitch_positions, pitch_class_positions


BEGINNER_EXACT_POSITIONS, BEGINNER_PC_POSITIONS = build_beginner_positions(4)


def get_beginner_position(target_midi_pitch):
    target_midi_pitch = int(target_midi_pitch)

    exact_candidates = BEGINNER_EXACT_POSITIONS.get(target_midi_pitch, [])
    if exact_candidates:
        best = exact_candidates[0]
        return best[0], best[1]

    pitch_class = target_midi_pitch % 12
    octave_candidates = BEGINNER_PC_POSITIONS.get(pitch_class, [])

    if not octave_candidates:
        return None, None

    best = min(
        octave_candidates,
        key=lambda x: (abs(x[2] - target_midi_pitch), x[1], x[0])
    )
    return best[0], best[1]


def transpose_song_to_beginner(midi_pitches):
    beg_strings = []
    beg_frets = []

    for midi_pitch in midi_pitches:
        s, f = get_beginner_position(int(midi_pitch))
        beg_strings.append(s)
        beg_frets.append(f)

    return beg_strings, beg_frets


