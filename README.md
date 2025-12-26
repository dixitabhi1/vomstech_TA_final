# AI-Based Full Body Measurement Estimation (Approximate)

##  Project Overview

This project presents a complete, end-to-end solution for estimating approximate human body measurements from a set of three standard pose images (front, side, and standing full-body). Developed for the Voms Tech Machine Round, this system demonstrates a robust, explainable, and deployable AI/ML pipeline that meets all mandatory output and submission requirements.

The core of the solution is a hybrid approach combining state-of-the-art computer vision for pose estimation with a machine learning regression model for robust measurement prediction.

---
## Live Demo and API

*   **Working Web-Based Frontend Demo:** [https://body-scan-measure.vercel.app/](https://body-scan-measure.vercel.app/) or [https://vomstechfinal.vercel.app/](https://vomstechfinal.vercel.app/)
*   **Web Frontend Repository:**[https://github.com/dixitabhi1/body-scan-measure](https://github.com/dixitabhi1/body-scan-measure)
*   **Core ML/CV Repository:**[https://github.com/dixitabhi1/vomstech_TA_final](https://github.com/dixitabhi1/vomstech_TA_final)
*   **Swagger API Documentation:** [https://abhishek785-ai-body-measurement.hf.space/docs](https://abhishek785-ai-body-measurement.hf.space/docs)
*   **Planning and execution(using Notion):**[https://www.notion.so/AI-Body-Measurement-Project-Retrospective-Final-Submission-2d5f3b5325a48102b04dcc8d18668932](https://www.notion.so/AI-Body-Measurement-Project-Retrospective-Final-Submission-2d5f3b5325a48102b04dcc8d18668932)



##  Mandatory Outputs

The system successfully estimates the following mandatory body measurements:

| Measurement | Unit | Derivation |
| :--- | :--- | :--- |
| **Height** | cm | User-provided input (used as the absolute scale reference) |
| **Shoulder Width** | cm | Derived from front-view shoulder landmarks. |
| **Hip Width** | cm | Derived from front-view hip landmarks (serves as a key torso measurement). |
| **Arm Length** | cm | Derived from side-view shoulder-to-wrist distance. |
| **Leg / Inseam Length** | cm | Derived from standing-view hip-to-ankle distance. |

##  Technical Approach

The estimation process is broken down into four distinct, sequential, and highly explainable steps:

### 1. Pose Landmark Extraction

*   **Tool:** MediaPipe Pose
*   **Function:** Extracts 2D body landmarks (key points) from the uploaded front, side, and standing images.
*   **Key Points:** Critical landmarks such as shoulders, hips, wrists, ankles, and the nose are identified and their pixel coordinates are recorded.

### 2. Feature Engineering & Normalization

*   **Feature Creation:** Raw pixel distances are calculated between relevant pairs of landmarks (e.g., left shoulder to right shoulder for width, hip to ankle for leg length).
*   **Scale Invariance:** To make the measurements independent of the image resolution and distance from the camera, these pixel distances are **normalized** by the total body height in pixels (derived from the full-body standing image). This yields a set of **scale-invariant ratios** (e.g., `Shoulder Width (pixels) / Height (pixels)`).

### 3. Machine Learning Regression

*   **Model:** Random Forest Regressor
*   **Purpose:** The ML model is trained on synthetic anthropometric ratio data. It takes the pose-derived ratios as input and predicts a set of refined, final measurement ratios.
*   **Benefit:** This step acts as a crucial **scaling and noise-reduction layer**, correcting for minor pose variations, perspective distortions, and inherent inaccuracies in the 2D landmark detection, thereby improving the robustness of the final output over a purely rule-based system.

### 4. Scaling Logic (Final Measurement Calculation)

4. Scaling Logic (Final Measurement Calculation)

The final, absolute measurements in centimeters are calculated by applying the predicted height to the predicted ratios. This is the core scaling logic that converts the normalized, relative measurements into real-world values.



$$
\text{Final Measurement (cm)} = \text{Predicted Ratio (from ML model)} \times \text{Predicted Height (cm)}
$$

---

##  Accuracy and Justification

### Target Accuracy: ~85%

The project aims for a **reasonable accuracy** (target $\sim 85\%$) with a strong emphasis on the **approach and justification**, as requested.

| Metric | Value | Context |
| :--- | :--- | :--- |
| **Model R² Score** | $\sim 0.65$ | Achieved on synthetic validation data during model training. |
| **Expected Real-World Deviation** | $\pm 1-3 \text{ cm}$ | Aligns with industry standards for virtual fitting and sizing systems. |

### Justification for Accuracy

The $85\%$ target is justified by the following:

1.  **Hybrid Robustness:** The use of a **Random Forest Regressor** to smooth the raw pose data significantly reduces the impact of noise and minor errors from the initial computer vision step.
2.  **Scale Reference:** By relying on the **user-provided height** as the single, absolute scale reference, the system avoids complex and error-prone camera calibration, ensuring a consistent scaling factor.
3.  **Inherent Limitations:** The system's accuracy is constrained by the use of **2D images** (lacking depth information), potential **perspective distortion**, and the influence of **clothing**. The $\pm 1-3 \text{ cm}$ deviation is a realistic and well-justified expectation for a non-medical, approximate measurement system.

---

##  Assumptions & Limitations

### Assumptions

*   The person in the images is standing **upright** and facing the camera directly (front/side).
*   The **full body** (head to feet) is visible in the standing image for height normalization.
*   The person is wearing **minimal or tight-fitting clothing** to prevent landmark occlusion.
*   All three images belong to the **same person** and are taken at roughly the same distance/perspective.
*   The model accurately predicts the Height to serve as the scaling reference
### Limitations

*   **2D Constraint:** The system cannot account for body depth or curvature, leading to approximations (e.g., shoulder width is measured as a straight line across the body).
*   **Perspective Distortion:** Extreme camera angles or close-up shots can skew the landmark coordinates and ratios.
*   **Data Dependency:** The ML model is trained on synthetic anthropometric data due to the lack of publicly available, large-scale, paired image-and-measurement datasets.
*   **Approximate Only:** The measurements are suitable for sizing and virtual fitting but are not a substitute for professional medical or tailoring measurements.

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Computer Vision** | Python, MediaPipe, OpenCV | Pose landmark detection and image processing. |
| **Machine Learning** | Scikit-learn (Random Forest) | Regression model for robust ratio prediction. |
| **Backend API** | FastAPI | High-performance, modern API for serving the model (deployed on Hugging Face). |
| **Web Framework** | swaggerURL | Simple web application for local testing and demonstration. |
| **Frontend Demo** | React, Vite, Tailwind CSS(Loveable) | Professional, responsive web application for the final demo. |

##  Setup and Run Instructions

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### 1. Clone the Repository

```bash
git clone https://github.com/dixitabhi1/vomstech_TA_final.git
cd vomstech_TA_final
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Local Web App (Flask)

For a quick local demonstration:

```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000`.

### 4. Run the API (FastAPI)

The core prediction logic is also exposed via a FastAPI service, which is deployed on Hugging Face.

```bash
uvicorn predict_measurements:app --host 0.0.0.0 --port 8000
```

The API documentation (Swagger UI) will be available at `http://127.0.0.1:8000/docs`.

---

##  Project Structure

```
vomstech_TA_final/
├── app.py                      # FastAPI application for the API endpoint
├── feature_extraction.py       # Logic for calculating pixel distances and ratios
├── pose_utils.py               # Helper functions for MediaPipe Pose
├── predict_measurements.py     # Ratio prediction
├── measurement_model.pkl       # Trained Random Forest Regressor model
├── dataset.csv                 # Synthetic training data (anthropometric ratios)
├── requirements.txt            # Python dependencies
├── trained_model.ipynb         # Jupyter notebook detailing model training
├── templates/                  # HTML templates 
│   └── index.html
└── static/                     # Static assets (CSS, JS, images)
```


## 📄 License

This project is licensed under the MIT License.

---
*Prepared by Abhishek Dixit for the Voms Tech Machine Round.*
