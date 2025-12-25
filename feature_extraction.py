import numpy as np
from pose_utils import extract_landmarks

# Landmark indices
NOSE = 0
L_SHOULDER = 11
R_SHOULDER = 12
L_HIP = 23
R_HIP = 24
L_WRIST = 15
L_ANKLE = 27

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def estimate_height_cm(landmarks):
    nose = landmarks[NOSE]
    shoulder = landmarks[L_SHOULDER]
    hip = landmarks[L_HIP]
    ankle = landmarks[L_ANKLE]

    # Pixel distances
    hip_to_ankle = dist(hip, ankle)
    shoulder_to_ankle = dist(shoulder, ankle)
    nose_to_ankle = dist(nose, ankle)

    # Anthropometric ratios (approx)
    h1 = hip_to_ankle / 0.53
    h2 = shoulder_to_ankle / 0.82
    h3 = nose_to_ankle / 0.93

    height_px = (h1 + h2 + h3) / 3

    # Normalize to average adult height
    height_cm = (height_px / nose_to_ankle) * 170

    # Clamp to human range
    height_cm = max(140, min(height_cm, 200))

    return round(height_cm, 2)

def extract_features(front_img, stand_img):
    front = extract_landmarks(front_img)
    stand = extract_landmarks(stand_img)

    if front is None or stand is None:
        raise ValueError("Pose not detected properly")

    height_px = dist(stand[NOSE], stand[L_ANKLE])

    features = {
        "shoulder": dist(front[L_SHOULDER], front[R_SHOULDER]) / height_px,
        "hip": dist(front[L_HIP], front[R_HIP]) / height_px,
        "arm": dist(front[L_SHOULDER], front[L_WRIST]) / height_px,
        "leg": dist(stand[L_HIP], stand[L_ANKLE]) / height_px,
        "height_cm": estimate_height_cm(stand)
    }

    return features
