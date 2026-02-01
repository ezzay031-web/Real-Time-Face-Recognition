import streamlit as st
import face_recognition
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from PIL import Image

st.set_page_config(page_title="Face Recognition System", layout="centered")
st.title("🧠 Face Recognition & Attendance System")

KNOWN_FACES_DIR = "faces_db"
ATTENDANCE_FILE = "attendance.csv"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# ---------------- LOAD KNOWN FACES ----------------
known_encodings = []
known_names = []

for file in os.listdir(KNOWN_FACES_DIR):
    if file.endswith(".jpg") or file.endswith(".png"):
        img = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, file))
        enc = face_recognition.face_encodings(img)
        if enc:
            known_encodings.append(enc[0])
            known_names.append(os.path.splitext(file)[0])

# ---------------- REGISTER NEW FACE ----------------
st.subheader("➕ Register New Face")
name = st.text_input("Enter person name")
new_face = st.file_uploader("Upload face image", type=["jpg", "png"], key="register")

if st.button("Add Face"):
    if name and new_face:
        img = Image.open(new_face)
        img.save(f"{KNOWN_FACES_DIR}/{name}.jpg")
        st.success(f"{name} added successfully! Restart app to load.")
    else:
        st.warning("Name and image both required")

st.divider()

# ---------------- FACE RECOGNITION ----------------
st.subheader("🔍 Face Recognition & Attendance")
uploaded_file = st.file_uploader("Upload image", type=["jpg", "png"], key="detect")

if uploaded_file:
    image = np.array(Image.open(uploaded_file))
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    attendance = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            idx = matches.index(True)
            name = known_names[idx]

            time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            attendance.append([name, time])

        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if attendance:
        df = pd.DataFrame(attendance, columns=["Name", "Time"])
        df.to_csv(ATTENDANCE_FILE, mode="a", index=False, header=not os.path.exists(ATTENDANCE_FILE))
        st.success("Attendance marked!")

    st.image(image, use_column_width=True)

# ---------------- VIEW ATTENDANCE ----------------
st.divider()
st.subheader("📊 Attendance Records")

if os.path.exists(ATTENDANCE_FILE):
    st.dataframe(pd.read_csv(ATTENDANCE_FILE))
else:
    st.info("No attendance recorded yet.")
