import pandas as pd
import numpy as np
from basic_pitch.inference import predict

OPEN_STRING_MIDI = [64, 59, 55, 50, 45, 40] 

def audio_to_model_input(audio_path):
    try:
        model_output, midi_data, note_events = predict(audio_path)
    except Exception as e:
        print(f"Error in Basic Pitch: {e}")
        return None

    data = []
    for note in note_events:
        data.append({
            's_on': note[0],
            's_off': note[1],
            's_dur': note[1] - note[0],
            'pitch': int(note[2])
        })
    
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = df.sort_values(by='s_on').reset_index(drop=True)

    df['prev_pitch'] = df['pitch'].shift(1).fillna(0).astype(int)
    df['next_pitch'] = df['pitch'].shift(-1).fillna(0).astype(int)
    df['prev_dur'] = df['s_dur'].shift(1).fillna(0)
    df['next_dur'] = df['s_dur'].shift(-1).fillna(0)
    df['diff_prev'] = df['pitch'] - df['prev_pitch']
    df['diff_next'] = df['next_pitch'] - df['pitch']
    
    return df[['pitch', 's_dur', 'diff_prev', 'diff_next', 's_on', 's_off']]


def get_beginner_position(target_midi_pitch):
    beginner_zone = {}
    for s_idx, open_midi in enumerate(OPEN_STRING_MIDI):
        for fret in range(5): 
            p = open_midi + fret
            if p not in beginner_zone:
                beginner_zone[p] = (s_idx, fret)

    if target_midi_pitch in beginner_zone:
        return beginner_zone[target_midi_pitch]

    target_pc = target_midi_pitch % 12
    candidates = []

    for p_available, pos in beginner_zone.items():
        if p_available % 12 == target_pc:
            dist = abs(p_available - target_midi_pitch)
            candidates.append((dist, pos))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    return 0, 0 

