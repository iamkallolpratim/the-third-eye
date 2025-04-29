import streamlit as st
from deepface import DeepFace
import numpy as np
from PIL import Image
import json
import os
import datetime
from collections import defaultdict
import google.generativeai as genai

# === STREAMLIT UI ===
st.set_page_config(page_title="Emotion Tracker AI", layout="wide")
st.title("Emotion Tracker AI — DeepFace Version")

# === CONFIG ===
USE_GEMINI = True
GEMINI_API_KEY = "AIzaSyBANef6_oJ4nbzHy9lSHZujHaacGzCI974"
genai.configure(api_key=GEMINI_API_KEY) if USE_GEMINI else None
gemini = genai.GenerativeModel("gemini-2.0-flash") if USE_GEMINI else None



# === SESSION STATE INIT ===
if "emotion_sessions" not in st.session_state:
    st.session_state.emotion_sessions = defaultdict(list)

if "current_emotions" not in st.session_state:
    st.session_state.current_emotions = {}

if "start_times" not in st.session_state:
    st.session_state.start_times = {}

# === FUNCTIONS ===
def infer_interest(emotion):
    return "interested" if emotion.lower() in ['happy', 'surprise', 'neutral'] else "bored"

def track_emotion(user, emotion, age=None, gender=None):
    now = datetime.datetime.now()
    ce = st.session_state.current_emotions
    stt = st.session_state.start_times
    log = st.session_state.emotion_sessions

    if user not in ce or ce[user]["emotion"] != emotion:
        if user in ce:
            session = {
                "emotion": ce[user]["emotion"],
                "age": ce[user].get("age", "N/A"),
                "gender": ce[user].get("gender", "N/A"),
                "interest_level": infer_interest(ce[user]["emotion"]),
                "start_time": stt[user].strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": now.strftime("%Y-%m-%d %H:%M:%S")
            }
            log[user].append(session)

        ce[user] = {"emotion": emotion, "age": age, "gender": gender}
        stt[user] = now

def get_emotion_from_gemini(desc):
    try:
        response = gemini.generate_content(f"What emotion is being expressed: '{desc}'?")
        return response.text.strip()
    except Exception as e:
        return f"Gemini error: {e}"

def save_log():
    os.makedirs("emotion_logs", exist_ok=True)
    file_path = "emotion_logs/emotion_log.json"
    with open(file_path, "w") as f:
        json.dump(st.session_state.emotion_sessions, f, indent=4)
    return file_path

# === UI CONTROLS ===
use_gemini = st.checkbox("Use Gemini AI for Description Analysis", value=USE_GEMINI)
img_data = st.camera_input("Capture Audience Expression")

if img_data:
    img = Image.open(img_data)
    rgb_img = np.array(img)

    try:
        result = DeepFace.analyze(rgb_img, actions=['age', 'gender', 'emotion'], enforce_detection=False)
        if isinstance(result, list):
            result = result[0]

        # Extract results
        emotion = result["dominant_emotion"]
        age = result["age"]
        gender = result["gender"]

        # Optional Gemini interpretation
        if use_gemini:
            emotion = get_emotion_from_gemini(emotion)

        user_name = f"User_{len(st.session_state.current_emotions) + 1}"
        track_emotion(user_name, emotion, age, gender)

        # Display summary
        st.subheader("Detected Expression")
        st.write(f"**User:** {user_name}")
        st.write(f"**Emotion:** {emotion}")
        st.write(f"**Age:** {age}")
        st.write(f"**Gender:** {gender}")
        st.image(img, caption="Captured Frame")

    except Exception as e:
        st.error(f"DeepFace failed to analyze the image: {str(e)}")

# === SAVE LOG BUTTON ===
if st.button("Save Emotion Log"):
    path = save_log()
    with open(path, "rb") as f:
        st.download_button("Download JSON", f, file_name="emotion_log.json", mime="application/json")