import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import time
import openai

# ---------------- Config ----------------
st.set_page_config(page_title="🌱 Plant Doctor", layout="wide", page_icon="🌿")

MODEL_PATH = "models/plant_disease_model.keras"
CLASS_NAMES_PATH = "models/class_names.json"
TREATMENTS_PATH = "models/disease_treatments.json"

# ---------------- API Key Management ----------------
with st.sidebar:
    st.header("API Configuration")
    # Use the key from the widget's state directly to ensure it persists across reruns
    st.text_input("Enter your OpenAI API Key:", type="password", key="api_key_input")
    if st.session_state.get("api_key_input"):
        st.success("API Key set successfully!")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_model()

# ---------------- Load Class Names & Treatments ----------------
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

if os.path.exists(TREATMENTS_PATH):
    with open(TREATMENTS_PATH, "r") as f:
        treatments = json.load(f)
else:
    treatments = {}

# ---------------- Helper Function ----------------
def predict_disease(image):
    if model is None:
        return "Model not available", "Please train and save the model first."

    img = image.resize((128, 128))
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds)]
    pred_treatment = treatments.get(pred_class, "No treatment info available.")

    return pred_class, pred_treatment

# ---------------- CSS Styling ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #e8f5e9, #f1f8e9);
    font-family: 'Arial', sans-serif;
}
.stApp {
    background: linear-gradient(to right, #d0f0c0, #f0fff4);
}
.card {
    background-color: white;
    padding: 1rem;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}
.chat-bubble { padding: 0.5rem 1rem; border-radius: 12px; margin: 0.3rem 0; max-width: 70%; }
.user { background-color: #81c784; color: white; align-self: flex-end; }
.bot { background-color: #ffffff; color: #2e7d32; border: 1px solid #c8e6c9; align-self: flex-start; }
.chat-container { display: flex; flex-direction: column; }
.tip { margin-bottom: 0.7rem; padding: 0.8rem; border-radius: 10px; background: white; box-shadow: 0 2px 6px rgba(0,0,0,0.08); display: flex; align-items: center; }
.tip-icon { font-size: 1.3rem; margin-right: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ---------------- UI Layout ----------------
st.title("🌱 Plant Disease Detector & Care Assistant")

tabs = st.tabs(["🩺 Detection", "💬 Chatbot", "💡 Tips"])

# ---------------- Detection Tab ----------------
with tabs[0]:
    st.header("Upload Leaf Image")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        uploaded_file = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="📷 Uploaded Leaf", use_column_width=True)

    with col2:
        if uploaded_file and st.button("🔍 Predict Disease"):
            with st.spinner("Analyzing the leaf... 🌿"):
                disease, treatment = predict_disease(image)

                # Save last detected disease for chatbot reference
                st.session_state.last_disease = disease

                st.markdown(f"""
                <div class="card">
                    <h3>🦠 Detected Disease: <span style="color:#2e7d32">{disease}</span></h3>
                    <p><b>💊 Recommended Treatment:</b> {treatment}</p>
                </div>
                """, unsafe_allow_html=True)

# ---------------- Agentic Chatbot Tab ----------------
with tabs[1]:
    st.header("Agentic Plant Chatbot 💬")

    # Standardized chat history format (list of dicts)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if 'last_disease' not in st.session_state:
        st.session_state.last_disease = "None"

    # Display chat messages from history
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        if role == "user":
            st.markdown(f'<div class="chat-bubble user">👩‍🌾 {content}</div>', unsafe_allow_html=True)
        elif role == "assistant":
            st.markdown(f'<div class="chat-bubble bot">🤖 {content}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    if not st.session_state.get("api_key_input"):
        st.warning("Please enter your OpenAI API key in the sidebar to use the chatbot.")
    else:
        user_msg = st.text_input("Ask your plant question here:")
        if st.button("Send") and user_msg.strip() != "":
            st.session_state.chat_history.append({"role": "user", "content": user_msg})

            try:
                client = openai.OpenAI(api_key=st.session_state.api_key_input)

                system_prompt = f"""
                You are a friendly and knowledgeable 'Plant Doctor' AI assistant.
                Your goal is to help users identify plant diseases and provide clear, concise, and actionable advice.
                Current context: The last detected disease was '{st.session_state.last_disease}'.
                If the user asks about this disease, use the known treatment: '{treatments.get(st.session_state.last_disease, "No specific treatment info available.")}'.
                Keep your answers brief and to the point. Use emojis to make the conversation more engaging.
                """

                messages_for_api = [{"role": "system", "content": system_prompt}] + st.session_state.chat_history

                # Use the new client and updated method
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages_for_api
                )

                bot_response = response.choices[0].message.content
                st.session_state.chat_history.append({"role": "assistant", "content": bot_response})

                # Rerun the app to display the new message immediately
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.chat_history.append({"role": "assistant", "content": "Sorry, an error occurred. Please check your API key and try again."})
                st.rerun()

# ---------------- Tips Tab ----------------
with tabs[2]:
    st.header("💡 Plant Care Tips")

    tips = [
        ("💧", "Check soil moisture before watering."),
        ("☀️", "Ensure enough sunlight for your plant type."),
        ("🍃", "Regularly inspect leaves for signs of disease."),
        ("🌱", "Use fertilizers according to plant needs.")
    ]

    for icon, tip in tips:
        st.markdown(f'<div class="tip"><span class="tip-icon">{icon}</span>{tip}</div>', unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("<center>🌿 Developed with ❤️ | AI-Powered Plant Doctor 🌿</center>", unsafe_allow_html=True)