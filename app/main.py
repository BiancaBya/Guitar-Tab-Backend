from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import torch
import numpy as np
import os
import shutil
import json

from app.model_defs import EmbeddingCRNN
from app.utils import (
    audio_to_model_input,
    get_beginner_position,
    transpose_song_to_beginner
)
from app import models, database, auth

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/GuitarSet_CNN_best_86_37%.pth"
UPLOAD_DIR = "uploads"

OPEN_STRINGS = [64, 59, 55, 50, 45, 40]

MAX_FRET = 24
SEQ_LEN = 64
STRIDE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmbeddingCRNN()
try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint

    model.to(DEVICE)
    model.eval()
    print(f"Model Loaded Successfully on {DEVICE}!")
except Exception as e:
    print(f"Failed to load model: {e}")

os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


def feasible_string_indices(pitch: int):
    """
    Returnează doar string-urile pe care nota respectivă poate fi cântată
    în intervalul 0..MAX_FRET.
    """
    valid = []
    for idx, open_pitch in enumerate(OPEN_STRINGS):
        fret = pitch - open_pitch
        if 0 <= fret <= MAX_FRET:
            valid.append(idx)
    return valid


def predict_logits_windowed(model, pitch_arr, rest_arr, device):
    """
    Inferență pe ferestre glisante, mai apropiată de train.
    Returnează logits [T, 6].
    """
    T = len(pitch_arr)

    if T == 0:
        return np.zeros((0, 6), dtype=np.float32)

    if T <= SEQ_LEN:
        pad = SEQ_LEN - T
        if pad > 0:
            pitch_pad = np.pad(pitch_arr, (0, pad), mode="edge")
            rest_pad = np.pad(rest_arr, ((0, pad), (0, 0)), mode="edge")
        else:
            pitch_pad = pitch_arr
            rest_pad = rest_arr

        with torch.no_grad():
            p = torch.LongTensor(pitch_pad).unsqueeze(0).to(device)
            r = torch.tensor(rest_pad, dtype=torch.float32).unsqueeze(0).to(device)
            logits = model((p, r))[0].permute(1, 0).cpu().numpy()[:T]

        return logits.astype(np.float32)

    agg = np.zeros((T, 6), dtype=np.float32)
    cnt = np.zeros(T, dtype=np.float32)

    starts = list(range(0, T - SEQ_LEN + 1, STRIDE))
    if starts[-1] != T - SEQ_LEN:
        starts.append(T - SEQ_LEN)

    with torch.no_grad():
        for start in starts:
            end = start + SEQ_LEN

            p = torch.LongTensor(pitch_arr[start:end]).unsqueeze(0).to(device)
            r = torch.tensor(rest_arr[start:end], dtype=torch.float32).unsqueeze(0).to(device)

            logits = model((p, r))[0].permute(1, 0).cpu().numpy()  

            agg[start:end] += logits
            cnt[start:end] += 1.0

    cnt[cnt == 0] = 1.0
    agg /= cnt[:, None]

    return agg.astype(np.float32)


def decode_with_viterbi(logits, pitches):
    """
    Alege un traseu de string-uri plauzibil și cântabil.
    """
    T = len(pitches)
    if T == 0:
        return []

    candidates_per_note = []
    for pitch in pitches:
        valid = feasible_string_indices(int(pitch))
        if not valid:
            valid = [0]
        candidates_per_note.append(valid)

    dp = []
    back = []

    first_states = {}
    first_back = {}

    for s in candidates_per_note[0]:
        fret = int(pitches[0]) - OPEN_STRINGS[s]
        cost = -float(logits[0, s])
        cost += 0.08 * fret
        first_states[s] = cost
        first_back[s] = None

    dp.append(first_states)
    back.append(first_back)

    for t in range(1, T):
        curr_states = {}
        curr_back = {}

        for s in candidates_per_note[t]:
            curr_fret = int(pitches[t]) - OPEN_STRINGS[s]
            emission_cost = -float(logits[t, s]) + 0.08 * curr_fret

            best_cost = None
            best_prev_s = None

            for prev_s, prev_cost in dp[t - 1].items():
                prev_fret = int(pitches[t - 1]) - OPEN_STRINGS[prev_s]

                transition_cost = 0.0
                transition_cost += 0.35 * abs(curr_fret - prev_fret)
                transition_cost += 0.20 * abs(s - prev_s)

                total_cost = prev_cost + emission_cost + transition_cost

                if best_cost is None or total_cost < best_cost:
                    best_cost = total_cost
                    best_prev_s = prev_s

            curr_states[s] = best_cost
            curr_back[s] = best_prev_s

        dp.append(curr_states)
        back.append(curr_back)

    last_state = min(dp[-1], key=dp[-1].get)

    path = [last_state]
    for t in range(T - 1, 0, -1):
        last_state = back[t][last_state]
        path.append(last_state)

    path.reverse()
    return path


@app.post("/register", status_code=status.HTTP_201_CREATED)
def register_user(user_data: auth.UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(models.User).filter(models.User.username == user_data.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    hashed_pwd = auth.get_password_hash(user_data.password)
    new_user = models.User(username=user_data.username, password_hash=hashed_pwd)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully"}


@app.post("/token")
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = auth.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/predict-tab/")
async def predict_tablature(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    # current_user: models.User = Depends(auth.get_current_user)
):
    temp_filename = os.path.join(UPLOAD_DIR, file.filename)

    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        df = audio_to_model_input(temp_filename)

        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="No notes detected in audio.")

        pitches = df["pitch"].values.astype(np.int64)
        rest_features = df[["s_dur", "diff_prev", "diff_next"]].values.astype(np.float32)

        print("Running windowed model inference...")
        logits = predict_logits_windowed(model, pitches, rest_features, DEVICE)

        print("Running constrained decoding...")
        pred_string_indices = decode_with_viterbi(logits, pitches)

        beginner_strings, beginner_frets = transpose_song_to_beginner(pitches)

        result_tab_original = []
        result_tab_beginner = []

        for i, row in df.iterrows():
            pitch = int(row["pitch"])

            string_idx_orig = int(pred_string_indices[i])
            fret_orig = pitch - OPEN_STRINGS[string_idx_orig]

            if fret_orig < 0 or fret_orig > MAX_FRET:
                valid = feasible_string_indices(pitch)
                if valid:
                    best_valid = max(valid, key=lambda s: logits[i, s])
                    string_idx_orig = best_valid
                    fret_orig = pitch - OPEN_STRINGS[string_idx_orig]
                else:
                    string_idx_orig = 0
                    fret_orig = max(0, pitch - OPEN_STRINGS[0])

            beg_s = beginner_strings[i]
            beg_f = beginner_frets[i]

            common_data = {
                "time": float(row["s_on"]),
                "duration": float(row["s_dur_raw"]),   
                "pitch": pitch
            }

            result_tab_original.append({
                **common_data,
                "string": string_idx_orig + 1,
                "fret": int(fret_orig)
            })

            if beg_s is not None and beg_f is not None:
                result_tab_beginner.append({
                    **common_data,
                    "string": int(beg_s) + 1,
                    "fret": int(beg_f)
                })
            else:
                result_tab_beginner.append({
                    **common_data,
                    "string": None,
                    "fret": None
                })

        return {
            "filename": file.filename,
            "tablature": result_tab_original,
            "tablature_beginner": result_tab_beginner,
            "message": "Tablatures generated successfully!"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


@app.get("/history")
def get_user_history(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    tabs = db.query(models.Tablature).filter(models.Tablature.user_id == current_user.id).all()
    results = []

    for t in tabs:
        results.append({
            "id": t.id,
            "filename": t.filename,
            "preview": json.loads(t.json_content)[:5]
        })

    return results

