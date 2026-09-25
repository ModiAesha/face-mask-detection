import streamlit as st
import cv2
import numpy as np
from PIL import Image

from predict import process_frame


st.set_page_config(
    page_title="Mask Detection",
    page_icon="😷",
    layout="centered"
)


st.title("😷 Mask Detection & Compliance Checker")

st.caption(
    "Upload an image to detect faces and check mask compliance."
)

st.divider()


uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    pil_image = Image.open(uploaded_file).convert("RGB")

    img_array = np.array(pil_image)

    img_bgr = cv2.cvtColor(
        img_array,
        cv2.COLOR_RGB2BGR
    )

    with st.spinner("Analyzing..."):

        annotated_img, face_results, frame_summary = process_frame(
            img_bgr
        )

    annotated_rgb = cv2.cvtColor(
        annotated_img,
        cv2.COLOR_BGR2RGB
    )


    st.subheader("Detection Result")

    st.image(
        annotated_rgb,
        width="stretch"
    )


    if not face_results:

        st.warning("No faces detected in this image.")

    else:

        st.subheader("Face Results")

        # Table data
        table_data = []

        for i, result in enumerate(face_results, start=1):

            status = result["status"]

            if status == "SAFE":
                status_display = "✅ Safe"

            elif status == "VIOLATION":
                status_display = "🚫 Violation"

            else:
                status_display = "⚠️ Uncertain"


            prediction = result["predicted_class"].replace(
                "_",
                " "
            ).title()


            table_data.append({
                "Face": i,
                "Prediction": prediction,
                "Confidence": f"{result['confidence']}%",
                "Status": status_display,
                "Score": f"{result['compliance_score']}/100"
            })

        st.table(table_data)

        st.subheader("Overall Result")

        score = frame_summary["frame_compliance_score"]

        st.metric(
            "Compliance",
            f"{score}%"
        )

        st.progress(
            min(max(score / 100, 0.0), 1.0)
        )

        st.caption(
            f"{frame_summary['total_faces']} Face(s)  •  "
            f"{frame_summary['safe_count']} Safe  •  "
            f"{frame_summary['violation_count']} Violation(s)  •  "
            f"{frame_summary['uncertain_count']} Uncertain"
        )