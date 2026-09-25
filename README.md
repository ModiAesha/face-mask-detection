# Mask Detection and Face Safety Compliance Checker

A Computer Vision + Deep Learning system that detects whether a person is wearing a face mask correctly, incorrectly, or not at all, and computes a Compliance Score with a final safety status (SAFE / VIOLATION / UNCERTAIN).

---

## 1. Project Overview

Pipeline:

```
Image → Face Detection (OpenCV DNN) → Face Cropping →
CNN Classification (MobileNetV2) → Compliance Scoring → Final Output
```

For every detected face, the app displays: bounding box, predicted label, confidence percentage, compliance score (0-100), and final status.

---

## 2. Dataset Used

- **Source:** Kaggle — Face Mask Detection by andrewmvd
- **Format:** 853 images with PASCAL VOC-style XML annotations (bounding box + label per face; images can contain multiple faces)
- **Classes (raw → mapped):**
  - `with_mask` → `mask_proper`
  - `without_mask` → `no_mask`
  - `mask_weared_incorrect` → `mask_improper`
---

## 3. Model Architecture

**Base:** MobileNetV2 (pretrained on ImageNet, `include_top=False`)
**Head:** GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3) → Dense(3, Softmax)

**Why MobileNetV2 (transfer learning) over a custom CNN:**
- The dataset (~5,100 training images after augmentation) is small for training a CNN from scratch reliably
- MobileNetV2 is lightweight (~3.4M base parameters) — fast to train and suitable for real-time/webcam deployment
- Pretrained ImageNet features (edges, textures, shapes) transfer well to face-based classification, reducing the amount of new data needed

**Fine-tuning:** After initial training, the top 30 layers of MobileNetV2 were unfrozen (BatchNormalization layers kept frozen throughout) and retrained with a low learning rate (1e-5) to adapt pretrained features more specifically to mask-related visual patterns.

**Signs of overfitting observed:** After the first training phase, training accuracy (~93%) pulled ahead of validation accuracy (~87.85%), a ~5-point gap indicating mild overfitting. This was expected given the small `mask_improper` class and addressed via fine-tuning and additional targeted augmentation.

---

## 4. Face Detection Module

**Detector used:** OpenCV DNN face detector 

**Why face cropping improves classification accuracy:** cropping isolates the face region and removes irrelevant background, clothing, and clutter that would otherwise dilute the CNN's learned features. It also matches the exact preprocessing used during training (faces cropped via ground-truth XML bounding boxes), keeping train/inference conditions consistent.

**Why OpenCV DNN over Haarcascade (initial choice):** Haarcascade was tried first, since it's simple and ships built into OpenCV. However, testing revealed a critical limitation — Haarcascade relies on recognizing a complete facial pattern (eyes, nose, mouth together), and **failed to detect any faces in photos where subjects wore masks**, since the mask obscures the nose/mouth region it depends on. This is a serious flaw for a mask-detection tool specifically. Switching to the OpenCV DNN face detector (a deep-learning-based detector) resolved this, since it is far more robust to partial occlusion.

**Remaining limitations of the DNN detector:**
- Some accuracy degradation observed on real-world crowd photos with angled faces, motion blur, or small/distant faces — a domain gap between the clean, front-facing training data and messy real-world scenes
- Performance depends on a fixed confidence threshold (0.5); very low-light or extreme angle conditions may still miss detections

---

## 5. Compliance Scoring System (Intelligent Feature)

### Per-face scoring logic

```
IF confidence < 60%:
    status = "UNCERTAIN"
    compliance_score = confidence

ELSE:
    IF predicted_class == "mask_proper":
        compliance_score = confidence
    IF predicted_class == "no_mask":
        compliance_score = 100 - confidence
    IF predicted_class == "mask_improper":
        compliance_score = 50 - (confidence * 0.5)

    status = "SAFE" if compliance_score >= 70 else "VIOLATION"
```

**Explanation:**
- **Uncertainty gate (confidence < 60%):** runs first — if the model itself isn't confident, the prediction shouldn't be trusted enough to issue a safety verdict at all.
- **Confidence ≠ compliance:** these measure different things. High confidence in `no_mask` means high confidence in an *unsafe* outcome, so the score must invert (`100 - confidence`), not copy the raw confidence value.
- **`mask_improper` is capped at a maximum of 50:** an improperly worn mask is always at least a partial violation and should never be scored as fully "safe," regardless of how confident the model is.
- **SAFE threshold of 70 (not 50):** builds in a safety margin — borderline scores (60-69) are still treated as VIOLATION, reflecting a deliberately conservative design appropriate for a safety-compliance tool.
---

## 6. Challenges Faced

1. **Class imbalance (~26:1)** in `mask_improper` — fixed via targeted augmentation + class weighting, though the rare class's recall (42%) remained limited by genuine data scarcity (~90 real images), not model design.
2. **Model version mismatch** — model trained in Colab (TensorFlow 2.20.0) failed to load locally (2.15.0) due to a Keras config incompatibility; fixed by matching versions.
3. **Local environment issue** — Python 3.14 only supported an unstable TensorFlow build, causing a Windows DLL crash; fixed by using Python 3.11 in a dedicated virtual environment.
4. **Haarcascade detector failed on masked faces** entirely (relies on seeing the full face); switched to OpenCV's DNN face detector, which handles occlusion correctly.
5. **Real-world generalization gap** — accuracy dropped on unrelated real-world photos (e.g., crowd scenes) compared to the clean test set, due to angle, lighting, and face-size differences.

## 7. Possible Improvements

- Collect or source additional real (non-augmented) images of incorrectly-worn masks to directly address the `mask_improper` data scarcity, rather than relying on synthetic augmentation of a small base set.
- Train or fine-tune on a more diverse, real-world dataset (varied angles, distances, lighting, crowd scenes) to close the generalization gap observed in real-world testing.
- Experiment with EfficientNetB0 as an alternative backbone for comparison against MobileNetV2.
- Add a rolling/temporal compliance score across video frames for live webcam deployment (partially designed via the frame-level scoring function, not yet extended to video).
- Add a sound/visual alert triggered on VIOLATION status for live deployment scenarios.
- Containerize the application (Docker) for easier deployment.

---

## 8. How to Run

```bash
# 1. Create and activate a virtual environment (Python 3.11 recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the web app
cd modules
streamlit run app.py
```

Upload an image via the web interface to see detected faces, predicted mask status, compliance scores, and final safety status for each person.

---

## 9. Accuracy Metrics Summary

- **Test Accuracy:** 92.35%
- **Test Loss:** 0.3456 (pre-final-augmentation-round baseline, final round improved per-class recall)
- **Macro-average F1-score:** 0.74
- Full confusion matrix and classification report available in `notebooks/face_detection.ipynb` and the accompanying documentation report.
