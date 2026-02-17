import pandas as pd
import numpy as np
from basic_pitch.inference import predict

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

