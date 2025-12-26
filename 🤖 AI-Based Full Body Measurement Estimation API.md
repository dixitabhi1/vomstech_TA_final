# 🤖 AI-Based Full Body Measurement Estimation API

## 🚀 Overview

This Hugging Face Space hosts the **FastAPI** backend for the **AI-Based Full Body Measurement Estimation** project. It provides a robust, high-performance API endpoint that accepts three pose images and returns a set of approximate body measurements in centimeters.

This API serves as the core prediction engine for the live web application.

---

## ⚙️ API Endpoint

The prediction service is exposed via a single `POST` endpoint:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **POST** | `/predict` | Estimates body measurements from three uploaded images. |

### Request Details

*   **Content Type:** `multipart/form-data`
*   **Required Fields:**
    *   `front`: Front-view image file.
    *   `side`: Side-view image file.
    *   `stand`: Standing full-body image file.
    *   *Note: The user's height is expected to be provided as a feature within the image processing pipeline (or a separate field if the API were modified to accept it directly). For this implementation, the height is an implicit scaling factor used in the feature engineering step.*

### API Documentation (Swagger UI)

You can interact with the live API and view the full schema documentation here:
[https://abhishek785-ai-body-measurement.hf.space/docs](https://abhishek785-ai-body-measurement.hf.space/docs)

---

## 🧠 Approach Used (The Pipeline)

The measurement estimation follows a four-stage hybrid pipeline:

1.  **Pose Estimation:** Uses MediaPipe Pose to extract 2D landmarks from the three input images.
2.  **Feature Engineering:** Calculates pixel distances between key landmarks (e.g., shoulder-to-shoulder, hip-to-ankle). These distances are normalized by the total body height in pixels to create **scale-invariant ratios**.
3.  **ML Regression:** A pre-trained **Random Forest Regressor** model takes the pose-derived ratios and predicts a set of refined, final measurement ratios. This step corrects for noise and improves robustness.
4.  **Scaling Logic:** The predicted ratios are multiplied by the user-provided height (which is an input to the overall system, though not explicitly in the API call for this specific deployment) to yield the final measurements in centimeters.

---

## 📐 Scaling Logic

The final measurement is derived by scaling the predicted ratio (a value between 0 and 1) by the user's known height. This ensures the output is in the correct unit (cm) and is personalized to the individual.

$$
\text{Final Measurement (cm)} = \text{Predicted Ratio (from ML model)} \times \text{User Height (cm)}
$$

---

## 📊 Estimated Measurements

The API returns the following approximate body measurements:

*   **Height** (cm)
*   **Shoulder Width** (cm)
*   **Hip** (cm)
*   **Arm Length** (cm)
*   **Leg / Inseam Length** (cm)

---

## ⚠️ Assumptions & Limitations

### Assumptions

*   **Standard Pose:** The person is in a standard, upright pose in all three images.
*   **Clear View:** Full body landmarks are clearly visible, with minimal loose clothing.
*   **Accurate Height:** The user-provided height (used for scaling) is accurate.

### Limitations

*   **2D Input:** Measurements are approximations due to the lack of 3D depth information.
*   **Perspective:** Accuracy is sensitive to camera angle and distance.
*   **Synthetic Data:** The ML model is trained on synthetic anthropometric data, which may not perfectly capture all real-world variations.

---

## 🔗 Project Links

*   **Source Code (ML/API):** [https://github.com/dixitabhi1/vomstech_TA_final](https://github.com/dixitabhi1/vomstech_TA_final)
*   **Web-Based Frontend Demo:** [https://body-scan-measure.vercel.app/](https://body-scan-measure.vercel.app/)
*   **Frontend Source Code:** [https://github.com/dixitabhi1/body-scan-measure](https://github.com/dixitabhi1/body-scan-measure)
