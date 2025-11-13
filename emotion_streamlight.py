import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import datetime
import json
import os
from collections import defaultdict
import google.generativeai as genai
from deepface import DeepFace

# === STREAMLIT UI ===
st.set_page_config(page_title="Emotion Tracker AI", layout="wide")
st.title("Emotion Tracker AI — Streamlit Version")


# === SESSION STATE INIT ===
from collections import defaultdict

if "emotion_sessions" not in st.session_state:
    st.session_state.emotion_sessions = defaultdict(list)

if "current_emotions" not in st.session_state:
    st.session_state.current_emotions = {}

if "start_times" not in st.session_state:
    st.session_state.start_times = {}

# === CONFIG ===
USE_GEMINI = True
GEMINI_API_KEY = "GEMINI_API_KEY"
genai.configure(api_key=GEMINI_API_KEY) if USE_GEMINI else None
gemini = genai.GenerativeModel("gemini-2.0-flash") if USE_GEMINI else None

# === INIT MEDIAPIPE ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True)

# === STATE TRACKING ===
emotion_sessions = defaultdict(list)
current_emotions = {}
start_times = {}

def track_emotion(user, emotion):
    now = datetime.datetime.now()
    current_emotions = st.session_state.current_emotions
    start_times = st.session_state.start_times
    sessions = st.session_state.emotion_sessions

    if user not in current_emotions or current_emotions[user] != emotion:
        if user in current_emotions:
            session = {
                "emotion": current_emotions[user],
                "start_time": start_times[user].strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": now.strftime("%Y-%m-%d %H:%M:%S")
            }
            sessions[user].append(session)
        current_emotions[user] = emotion
        start_times[user] = now

def describe_expression(landmarks):
    try:
        # Extracting the necessary facial landmarks
        top_lip = landmarks[13]
        bottom_lip = landmarks[14]
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        right_eye_top = landmarks[386]
        right_eye_bottom = landmarks[374]
        left_eyebrow = landmarks[70]
        right_eyebrow = landmarks[300]

        # Calculating mouth openness and eye openness
        mouth_open = abs(bottom_lip.y - top_lip.y)
        eye_open_left = abs(left_eye_bottom.y - left_eye_top.y)
        eye_open_right = abs(right_eye_bottom.y - right_eye_top.y)
        eyebrow_distance = abs(left_eyebrow.y - right_eyebrow.y)

        # Emotion detection based on facial features
        if mouth_open > 0.05 and eye_open_left > 0.03 and eye_open_right > 0.03:
            return "Surprised"
        elif mouth_open < 0.02 and eye_open_left < 0.02 and eye_open_right < 0.02:
            return "Sad"
        elif mouth_open > 0.04 and eyebrow_distance < 0.01:
            return "Happy"
        elif mouth_open > 0.03 and eyebrow_distance > 0.05:
            return "Anger"
        elif eye_open_left > 0.03 and eye_open_right > 0.03 and eyebrow_distance > 0.04:
            return "Fear"
        elif mouth_open < 0.02 and eyebrow_distance > 0.02:
            return "Disgust"
        else:
            return "Neutral"  # Default emotion if no clear pattern is detected

    except Exception as e:
        print(f"Error in describe_expression: {e}")
        return "Neutral expression."  # Default to neutral if anything goes wrong


def analyze_emotion_deepface(frame, x, y, w, h):
    face_crop = frame[y:y+h, x:x+w]
    try:
        result = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list):
            return result[0]['dominant_emotion']
        return result['dominant_emotion']
    except Exception as e:
        print("DeepFace error:", e)
        return "Unknown"

def get_emotion_from_gemini(desc):
    try:
        response = gemini.generate_content(f"What emotion is being expressed: '{desc}'?")
        return response.text.strip()
    except Exception as e:
        return f"Error getting emotion from Gemini: {str(e)}"

def save_log():
    os.makedirs("emotion_logs", exist_ok=True)
    file_path = "emotion_logs/emotion_log.json"
    with open(file_path, "w") as f:
        json.dump(st.session_state.emotion_sessions, f, indent=4)
    return file_path


run = st.checkbox("Activate Camera", value=True)
use_gemini = st.checkbox("Use Gemini AI", value=USE_GEMINI)
frame_display = st.empty()
status_box = st.empty()

if run:
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            status_box.error("Camera frame not found")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            for i, landmarks in enumerate(result.multi_face_landmarks):
                name = f"User_{i+1}"
                desc = describe_expression(landmarks.landmark)
                emotion = get_emotion_from_gemini(desc) if use_gemini else desc
                track_emotion(name, emotion)

                # Label on frame
                pt = landmarks.landmark[10]
                h, w, _ = frame.shape
                x, y = int(pt.x * w), int(pt.y * h)
                cv2.putText(frame, f"{name}: {emotion}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Display to Streamlit
        frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

# === Save log button ===
if st.button("Save Emotion Log"):
    path = save_log()
    with open(path, "rb") as f:
        st.download_button("Download JSON", f, file_name="emotion_log.json", mime="application/json")

