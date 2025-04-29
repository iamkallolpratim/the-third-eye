import streamlit as st
import cv2
import numpy as np
import datetime
import json
import os
from collections import defaultdict
from deepface import DeepFace
from PIL import Image
import mediapipe as mp

# === CONFIG ===
st.set_page_config(page_title="Emotion Tracker AI", layout="wide")
st.title("Emotion Tracker AI — Multi-face Engagement & Emotion Tracking")

# === SESSION STATE INIT ===
if "emotion_sessions" not in st.session_state:
    st.session_state.emotion_sessions = defaultdict(list)
if "current_emotions" not in st.session_state:
    st.session_state.current_emotions = {}
if "start_times" not in st.session_state:
    st.session_state.start_times = {}

# === TRACKING FUNCTION ===


def track_emotion(user, emotion, activity, age=None, gender=None):
    now = datetime.datetime.now()
    ce = st.session_state.current_emotions
    stt = st.session_state.start_times
    log = st.session_state.emotion_sessions

    if user not in ce or ce[user]["emotion"] != emotion or ce[user]["activity"] != activity:
        if user in ce:
            session = {
                "emotion": ce[user]["emotion"],
                "activity": ce[user]["activity"],
                "age": ce[user].get("age", "N/A"),
                "gender": ce[user].get("gender", "N/A"),
                "start_time": stt[user].strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": now.strftime("%Y-%m-%d %H:%M:%S")
            }
            log[user].append(session)

        ce[user] = {
            "emotion": emotion,
            "activity": activity,
            "age": age,
            "gender": gender
        }
        stt[user] = now

# === SAVE LOG ===


def save_log():
    os.makedirs("emotion_logs", exist_ok=True)
    log_data = {k: v for k, v in st.session_state.emotion_sessions.items()}
    with open("emotion_logs/emotion_log.json", "w") as f:
        json.dump(log_data, f, indent=4)
    st.success("Emotion log saved successfully!")

# === ESTIMATE ACTIVITY ===


def estimate_activity(landmarks):
    left_eye = landmarks[159].y - landmarks[145].y
    right_eye = landmarks[386].y - landmarks[374].y
    eye_open = (left_eye + right_eye) / 2
    mouth_open = landmarks[14].y - landmarks[13].y

    if eye_open < 0.015:
        return "Drowsy"
    elif mouth_open > 0.03:
        return "Speaking or Yawning"
    else:
        return "Attentive"


# === UI ===
run = st.checkbox("Start Tracking", value=False)
frame_placeholder = st.empty()

# === ANALYSIS LOOP ===
if run:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera error")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        h, w, _ = frame.shape

        if results.multi_face_landmarks:
            for i, face_landmarks in enumerate(results.multi_face_landmarks):
                # Bounding box from landmarks
                xs = [lm.x for lm in face_landmarks.landmark]
                ys = [lm.y for lm in face_landmarks.landmark]
                x_min, x_max = int(min(xs) * w), int(max(xs) * w)
                y_min, y_max = int(min(ys) * h), int(max(ys) * h)

                # Crop face
                face_crop = rgb[y_min:y_max, x_min:x_max]

                try:
                    result = DeepFace.analyze(
                        face_crop, actions=["emotion", "age", "gender"], enforce_detection=False)
                    if isinstance(result, list):
                        result = result[0]
                    emotion = result["dominant_emotion"]
                    age = result["age"]
                    gender = result["gender"]
                except Exception as e:
                    emotion, age, gender = "Unknown", "N/A", "N/A"
                    print("DeepFace error:", e)

                # Estimate activity
                activity = estimate_activity(face_landmarks.landmark)

                # Track user
                user_name = f"User_{i+1}"
                track_emotion(user_name, emotion, activity, age, gender)

                # Draw green box and labels
                cv2.rectangle(frame, (x_min, y_min),
                              (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"{user_name}: {emotion}, {activity}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"{gender}, {age}y", (x_min, y_max + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Display frame
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_placeholder.image(img)

    cap.release()

# === LOG SAVE BUTTON ===
if st.button("Save Emotion Log"):
    save_log()
    with open("emotion_logs/emotion_log.json", "rb") as f:
        st.download_button(
            "Download JSON", f, file_name="emotion_log.json", mime="application/json")
