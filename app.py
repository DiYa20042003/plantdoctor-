import streamlit as st
from PIL import Image
import json
import os
from predict import predict_image
import openai

# ---------------- Config ----------------
st.set_page_config(page_title="🌱 Plant Doctor", layout="wide", page_icon="🌿")

TREATMENTS_PATH = "models/disease_treatments.json"

# ---------------- Load Treatments ----------------
if os.path.exists(TREATMENTS_PATH):
    with open(TREATMENTS_PATH, "r") as f:
        treatments = json.load(f)
else:
    treatments = {}

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

# ---------------- Sidebar for API Key ----------------
st.sidebar.title("Settings")
st.session_state.api_key_input = st.sidebar.text_input("Enter your OpenAI API Key (Optional)", type="password")


tabs = st.tabs(["🩺 Detection", "💬 Chatbot", "💡 Tips"])

# ---------------- Detection Tab ----------------
with tabs[0]:
    st.header("Upload or Capture a Leaf Image")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        uploaded_file = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])
        camera_photo = st.camera_input("Or, take a picture with your camera")

        # Determine which input to use
        image_to_predict = None
        if uploaded_file:
            image_to_predict = Image.open(uploaded_file).convert("RGB")
            st.image(image_to_predict, caption="📷 Uploaded Leaf", use_column_width=True)
        elif camera_photo:
            image_to_predict = Image.open(camera_photo).convert("RGB")
            st.image(image_to_predict, caption="📷 Captured Leaf", use_column_width=True)


    with col2:
        if image_to_predict and st.button("🔍 Predict Disease"):
            with st.spinner("Analyzing the leaf... 🌿"):
                disease, confidence = predict_image(image_to_predict)
                treatment = treatments.get(disease, "No treatment info available.")

                # Save last detected disease for chatbot reference
                st.session_state.last_disease = disease

                st.markdown(f"""
                <div class="card">
                    <h3>🦠 Detected Disease: <span style="color:#2e7d32">{disease}</span></h3>
                    <p><b>Confidence:</b> {confidence:.2f}%</p>
                    <p><b>💊 Recommended Treatment:</b> {treatment}</p>
                </div>
                """, unsafe_allow_html=True)

# ---------------- Chatbot Tab ----------------
with tabs[1]:
    st.header("Plant Chatbot 💬")
    st.info("The advanced chatbot uses OpenAI. Enter your API key in the sidebar for AI-powered answers. Without a key, it will use a basic keyword-based system.", icon="💡")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'last_disease' not in st.session_state:
        st.session_state.last_disease = None

    user_msg = st.text_input("Ask your plant question here:")
    if st.button("Send") and user_msg.strip() != "":
        # Use OpenAI if API key is provided
        if st.session_state.api_key_input:
            openai.api_key = st.session_state.api_key_input
            try:
                # Context-aware prompt
                prompt = f"You are a helpful plant care assistant. A user has a question about their plant. The last detected disease was '{st.session_state.last_disease if st.session_state.last_disease else 'None'}'. The user's question is: '{user_msg}'. Provide a concise and helpful answer."

                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=150
                )
                answer = response.choices[0].text.strip()
            except Exception as e:
                answer = f"🤖 Error connecting to OpenAI: {e}"
        # Fallback to simple keyword-based bot if no API key
        else:
            fallback = {
                "watering": "💧 Water the plant 2-3 times a week depending on soil moisture.",
                "sunlight": "☀️ Most plants need 6-8 hours of sunlight daily.",
                "fertilizer": "🌱 Use a balanced NPK fertilizer every 3-4 weeks.",
                "leaf": "🍃 Yellow leaves may indicate overwatering or nutrient deficiency."
            }

            # Check if the question relates to the last detected disease
            if st.session_state.last_disease and st.session_state.last_disease.lower() in user_msg.lower():
                answer = treatments.get(st.session_state.last_disease, f"I don't have specific treatment info for {st.session_state.last_disease}.")
            else:
                # Simple keyword matching
                found_keyword = False
                for key, val in fallback.items():
                    if key in user_msg.lower():
                        answer = val
                        found_keyword = True
                        break
                if not found_keyword:
                    answer = "🤖 I can answer questions about watering, sunlight, fertilizer, or the last detected disease. For more advanced questions, please provide an OpenAI API key."

        # Save to history
        st.session_state.chat_history.append((user_msg, answer))

    # Display chat
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for chat in st.session_state.chat_history:
        st.markdown(f'<div class="chat-bubble user">👩‍🌾 {chat[0]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-bubble bot">🤖 {chat[1]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

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