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
from app.utils import audio_to_model_input
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

OPEN_STRINGS = [40, 45, 50, 55, 59, 64]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = EmbeddingCRNN()
try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
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
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
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
    # current_user: models.User = Depends(auth.get_current_user) # Comentat pt testare rapida
):
    temp_filename = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        df = audio_to_model_input(temp_filename)
        
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="No notes detected in audio.")
            
        pitch_tensor = torch.LongTensor(df['pitch'].values.astype(int)).unsqueeze(0).to(DEVICE)
        rest_features = df[['s_dur', 'diff_prev', 'diff_next']].values.astype(np.float32)
        rest_tensor = torch.tensor(rest_features).unsqueeze(0).to(DEVICE)
        
        print("Running Model Prediction...")
        with torch.no_grad():
            logits = model((pitch_tensor, rest_tensor))
            pred_strings = torch.argmax(logits, dim=1).cpu().numpy().flatten()
            
        result_tab = []
        for i, row in df.iterrows():
            string_idx = int(pred_strings[i])
            pitch = int(row['pitch'])
            
            if 0 <= string_idx < 6:
                open_pitch = OPEN_STRINGS[string_idx]
                fret = pitch - open_pitch

                if fret < 0:
                    fret = 0
            else:
                fret = 0 

            result_tab.append({
                "time": float(row['s_on']),
                "duration": float(row['s_dur']),
                "string": string_idx + 1, 
                "fret": fret,
                "pitch": pitch
            })
            
        json_content_str = json.dumps(result_tab)
        new_tabEntry = models.Tablature(
            filename=file.filename,
            json_content=json_content_str,
            # user_id=current_user.id  
        )
        db.add(new_tabEntry)
        db.commit()

        return {
            "filename": file.filename, 
            "tablature": result_tab,
            "message": "Tablature generated successfully using Notebook logic!"
        }

    except Exception as e:
        print(f"Error processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.get("/history")
def get_user_history(db: Session = Depends(get_db), current_user: models.User = Depends(auth.get_current_user)):
    tabs = db.query(models.Tablature).filter(models.Tablature.user_id == current_user.id).all()
    results = []
    for t in tabs:
        results.append({
            "id": t.id,
            "filename": t.filename,
            "preview": json.loads(t.json_content)[:5] 
        })
    return results

