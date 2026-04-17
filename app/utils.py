import pandas as pd
import numpy as np
from basic_pitch.inference import predict

OPEN_STRING_MIDI = [64, 59, 55, 50, 45, 40]

MIN_GUITAR_MIDI = 40   
MAX_GUITAR_MIDI = 88


def deduplicate_nearby_notes(df: pd.DataFrame, time_threshold: float = 0.03) -> pd.DataFrame:
    """
    Elimină duplicate apropiate pentru același pitch.
    """
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
    """
    Creează inputul pentru model și păstrează și durata brută în secunde
    pentru outputul API.
    """
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


def get_beginner_position(target_midi_pitch):
    beginner_zone = {}

    # pitches in the first 4 frets
    # pitch = key, [string, fret] = value
    for s_idx, open_midi in enumerate(OPEN_STRING_MIDI):
        for fret in range(5):
            p = open_midi + fret
            if p not in beginner_zone:
                beginner_zone[p] = (s_idx, fret)

    if target_midi_pitch in beginner_zone:
        return beginner_zone[target_midi_pitch]

    # octave displacement
    target_pc = target_midi_pitch % 12
    candidates = []

    for p_available, pos in beginner_zone.items():
        if p_available % 12 == target_pc:
            dist = abs(p_available - target_midi_pitch)
            candidates.append((dist, pos))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    return None, None


def transpose_song_to_beginner(midi_pitches):
    beg_strings = []
    beg_frets = []

    for m in midi_pitches:
        s, f = get_beginner_position(int(m))
        beg_strings.append(s)
        beg_frets.append(f)

    return beg_strings, beg_frets


