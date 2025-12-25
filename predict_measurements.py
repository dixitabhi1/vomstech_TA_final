import joblib
import numpy as np
from feature_extraction import extract_features

model = joblib.load("measurement_model.pkl")

front_img = "static/uploads/front.jpg"
stand_img = "static/uploads/stand.jpg"

data = extract_features(front_img, stand_img)

X = np.array([[data["shoulder"], data["hip"], data["arm"], data["leg"]]])
pred = model.predict(X)[0]

height = data["height_cm"]

print("\n📏 Estimated Measurements\n")
print(f"Height (cm): {height}")
print(f"Shoulder (cm): {round(pred[0] * height, 2)}")
print(f"Hip (cm): {round(pred[1] * height, 2)}")
print(f"Arm Length (cm): {round(pred[2] * height, 2)}")
print(f"Leg / Inseam (cm): {round(pred[3] * height, 2)}")
