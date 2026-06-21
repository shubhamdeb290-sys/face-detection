import streamlit as st
import cv2
import face_recognition
import numpy as np
from datetime import datetime
import pandas as pd

st.title("Face Detection Attendance System (No CSV)")

# Upload reference image
uploaded_file = st.file_uploader("Upload Reference Photo", type=["jpg", "png"])
if uploaded_file:
    ref_image = face_recognition.load_image_file(uploaded_file)
    ref_encoding = face_recognition.face_encodings(ref_image)[0]
    ref_name = "Person1"  # Replace with actual name if needed
    st.success("Reference photo uploaded successfully!")

# Attendance DataFrame (in memory only)
attendance = pd.DataFrame(columns=["Name", "Time"])

# Start webcam verification
if st.button("Start Attendance") and uploaded_file:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces([ref_encoding], face_encoding)
            name = "Unknown"

            if matches[0]:
                name = ref_name
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                attendance.loc[len(attendance)] = [name, now]

            # Draw rectangle and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        stframe.image(frame, channels="BGR")

    cap.release()

# Show attendance log (only in app, not saved)
if not attendance.empty:
    st.subheader("Attendance Log")
    st.dataframe(attendance)
