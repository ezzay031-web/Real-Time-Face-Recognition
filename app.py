# ===============================
# FACE RECOGNITION & ATTENDANCE
# STREAMLIT + DEEPFACE (CLOUD SAFE)
# ===============================

import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from deepface import DeepFace

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Face Attendance", layout="centered")

KNOWN_FACES_DIR = "faces_db"
ATTENDANCE_FILE = "attendance.csv"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# ---------------- UI ----------------
st.title("🧑‍💼 Face Recognition Attendance System")

menu = st.sidebar.selectbox(
    "Select Option",
    ["Register New Face", "Mark Attendance", "View Attendance"]
)

# ---------------- REGISTER FACE ----------------
if menu == "Register New Face":
    st.header("➕ Register New Face")

    name = st.text_input("Enter person name")
    uploaded_file = st.file_uploader("Upload face image", type=["jpg", "png", "jpeg"])

    if st.button("Save Face"):
        if name and uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            save_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
            img.save(save_path)
            st.success(f"✅ {name} registered successfully!")
        else:
            st.error("❌ Name and image both required")

# ---------------- MARK ATTENDANCE ----------------
elif menu == "Mark Attendance":
    st.header("📸 Mark Attendance")

    uploaded_file = st.file_uploader("Upload image for recognition", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        unknown_img = Image.open(uploaded_file).convert("RGB")
        st.image(unknown_img, caption="Uploaded Image", width=300)

        unknown_path = "temp_unknown.jpg"
        unknown_img.save(unknown_path)

        identified_name = "Unknown"

        for file in os.listdir(KNOWN_FACES_DIR):
            known_path = os.path.join(KNOWN_FACES_DIR, file)

            try:
                result = DeepFace.verify(
                    img1_path=unknown_path,
                    img2_path=known_path,
                    model_name="Facenet",
                    detector_backend="retinaface",
                    enforce_detection=False
                )

                if result["verified"]:
                    identified_name = os.path.splitext(file)[0]
                    break

            except Exception:
                continue

        st.subheader(f"🧠 Identified: **{identified_name}**")

        # Save attendance
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = pd.DataFrame([[identified_name, time]], columns=["Name", "Time"])
        df.to_csv(
            ATTENDANCE_FILE,
            mode="a",
            index=False,
            header=not os.path.exists(ATTENDANCE_FILE)
        )

        st.success("✅ Attendance marked successfully!")

# ---------------- VIEW ATTENDANCE ----------------
elif menu == "View Attendance":
    st.header("📊 Attendance Records")

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(df)
    else:
        st.info("No attendance recorded yet.")


