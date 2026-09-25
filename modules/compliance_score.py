def score_face(predicted_class, confidence):
    """
    Computes a per-face Compliance Score (0-100) and safety status,
    based on the model's predicted class and confidence.

    predicted_class: one of "mask_proper", "mask_improper", "no_mask"
    confidence: float, 0-100 (softmax probability as a percentage)

    Returns: dict with predicted_class, confidence, compliance_score, status
    """

    if confidence < 60:
        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "compliance_score": round(confidence, 2),
            "status": "UNCERTAIN"
        }

    if predicted_class == "mask_proper":
        compliance_score = confidence

    elif predicted_class == "no_mask":
        compliance_score = 100 - confidence

    elif predicted_class == "mask_improper":
        compliance_score = 50 - (confidence * 0.5)

    else:
        raise ValueError(f"Unknown predicted_class: {predicted_class}")

    compliance_score = max(0, min(100, compliance_score))  # keep within 0-100 bounds

    status = "SAFE" if compliance_score >= 70 else "VIOLATION"

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "compliance_score": round(compliance_score, 2),
        "status": status
    }


def score_frame(face_results):
    """
    Computes a frame-level (bonus) compliance score by aggregating
    per-face results from score_face().

    face_results: list of dicts, each one the output of score_face()

    Returns: dict with total_faces, safe_count, violation_count,
             uncertain_count, and frame_compliance_score
    """
    total_faces = len(face_results)
    safe_count = sum(1 for f in face_results if f["status"] == "SAFE")
    violation_count = sum(1 for f in face_results if f["status"] == "VIOLATION")
    uncertain_count = sum(1 for f in face_results if f["status"] == "UNCERTAIN")

    confidently_detected = safe_count + violation_count  # excludes UNCERTAIN

    if confidently_detected == 0:
        frame_compliance_score = 0  # avoid division by zero when all faces are uncertain
    else:
        frame_compliance_score = (safe_count / confidently_detected) * 100

    return {
        "total_faces": total_faces,
        "safe_count": safe_count,
        "violation_count": violation_count,
        "uncertain_count": uncertain_count,
        "frame_compliance_score": round(frame_compliance_score, 2)
    }