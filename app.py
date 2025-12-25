from fastapi import FastAPI, UploadFile, File
import os
import shutil
import joblib
import numpy as np
from feature_extraction import extract_features

app = FastAPI(title="AI Body Measurement Estimation")

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

model = joblib.load("measurement_model.pkl")

@app.post("/predict")
async def predict_measurements(
    front: UploadFile = File(...),
    side: UploadFile = File(...),
    stand: UploadFile = File(...)
):
    paths = {}

    for img, name in zip([front, side, stand], ["front", "side", "stand"]):
        path = f"{UPLOAD_DIR}/{name}.jpg"
        with open(path, "wb") as buffer:
            shutil.copyfileobj(img.file, buffer)
        paths[name] = path

    data = extract_features(paths["front"], paths["stand"])

    X = np.array([[data["shoulder"], data["hip"], data["arm"], data["leg"]]])
    pred = model.predict(X)[0]

    height = data["height_cm"]

    # 🔹 Step 1: Raw measurements
    shoulder_cm = pred[0] * height
    hip_cm = pred[1] * height
    arm_cm = pred[2] * height
    leg_cm = pred[3] * height

    # 🔹 Step 2: Apply correction factors (ONLY here)
    shoulder_cm *= 1.08   # compensate inner shoulder landmarks
    arm_cm *= 0.90        # reduce arm overestimation

    # 🔹 Step 3: Return corrected results
    return {
        "Estimated Height (cm)": round(height, 2),
        "Shoulder (cm)": round(shoulder_cm, 2),
        "Hip (cm)": round(hip_cm, 2),
        "Arm Length (cm)": round(arm_cm, 2),
        "Leg / Inseam (cm)": round(leg_cm, 2),
    }
