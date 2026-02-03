import streamlit as st
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime
from PIL import Image

KNOWN_FACES_DIR = "faces_db"
ATTENDANCE_FILE = "attendance.csv"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# -------- Load known faces --------
def load_known_faces():
    encodings = []
    names = []

    for person in os.listdir(KNOWN_FACES_DIR):
        person_path = os.path.join(KNOWN_FACES_DIR, person)
        if not os.path.isdir(person_path):
            continue

        for img_file in os.listdir(person_path):
            if img_file.endswith(("jpg", "png")):
                img_path = os.path.join(person_path, img_file)
                img = face_recognition.load_image_file(img_path)
                enc = face_recognition.face_encodings(img)
                if enc:
                    encodings.append(enc[0])
                    names.append(person)
    return encodings, names

st.title("🎯 Face Recognition Attendance System")

menu = st.sidebar.selectbox("Menu", ["Register Face", "Mark Attendance", "View Attendance"])

known_encodings, known_names = load_known_faces()

# -------- Register --------
if menu == "Register Face":
    name = st.text_input("Enter Name")
    images = st.file_uploader("Upload images", type=["jpg","png"], accept_multiple_files=True)

    if st.button("Save"):
        if name and images:
            person_dir = os.path.join(KNOWN_FACES_DIR, name)
            os.makedirs(person_dir, exist_ok=True)

            for i, img_file in enumerate(images):
                img = Image.open(img_file)
                img.save(f"{person_dir}/{i}.jpg")

            st.success("✅ Images saved successfully")

# -------- Attendance --------
elif menu == "Mark Attendance":
    img_file = st.file_uploader("Upload Image", type=["jpg","png"])

    if img_file:
        img = face_recognition.load_image_file(img_file)
        encs = face_recognition.face_encodings(img)

        if encs:
            enc = encs[0]
            distances = face_recognition.face_distance(known_encodings, enc)
            name = "Unknown"

            if len(distances) > 0:
                best = np.argmin(distances)
                if distances[best] < 0.48:
                    name = known_names[best]

            st.success(f"🧠 Identified: {name}")

            time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df = pd.DataFrame([[name, time]], columns=["Name","Time"])
            df.to_csv(ATTENDANCE_FILE, mode="a", index=False, header=not os.path.exists(ATTENDANCE_FILE))

# -------- View Attendance --------
else:
    if os.path.exists(ATTENDANCE_FILE):
        st.dataframe(pd.read_csv(ATTENDANCE_FILE))
    else:
        st.info("No attendance yet")

