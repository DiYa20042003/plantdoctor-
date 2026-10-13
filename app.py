"""Plant Doctor - Streamlit Web UI for Plant Disease Detection and Care Guidance"""
import html
import json
import os
import streamlit as st
from PIL import Image
from predict import PlantDiseasePredictor, IMG_SIZE, DEFAULT_MODEL_PATH, DEFAULT_CLASS_FILE

# ---------------- Page Config ----------------
st.set_page_config(page_title="🌱 Plant Doctor", layout="wide", page_icon="🌿")

TREATMENTS_PATH = os.path.join("models", "disease_treatments.json")

# Default care treatments for common crop conditions
DEFAULT_TREATMENTS = {
    "Apple___Apple_scab": "Apply fungicide such as captan or myclobutanil early in the growing season. Prune trees to increase air circulation.",
    "Apple___Black_rot": "Remove and destroy mummified fruit, dead wood, and cankers. Apply protective copper-based fungicide spray.",
    "Apple___healthy": "Leaves look healthy! Maintain regular watering at base and avoid overhead irrigation.",
    "Corn___Common_rust": "Apply foliar fungicides containing azoxystrobin or propiconazole at early sign of pustules. Rotate crops annually.",
    "Corn___healthy": "Crop is healthy! Ensure adequate soil nitrogen and consistent moisture during silking.",
    "Grape___Black_rot": "Prune affected vines during dormancy. Apply mancozeb or captan from bud break until veraison.",
    "Grape___healthy": "Vines are healthy! Ensure adequate sun exposure and trellis canopy management.",
    "Potato___Early_blight": "Apply chlorothalonil or copper-based sprays every 7-10 days. Avoid overhead irrigation and remove infected foliage.",
    "Potato___Late_blight": "Urgent: Destroy infected plants immediately to prevent spread. Apply systemic metalaxyl fungicides preventatively.",
    "Potato___healthy": "Foliage is healthy! Hill soil around stems and maintain balanced potassium and phosphorus.",
    "Tomato___Bacterial_spot": "Apply copper-mancozeb combination spray. Avoid working with wet plants to prevent bacterial spread.",
    "Tomato___Early_blight": "Remove lower infected leaves. Apply preventative copper fungicide or neem oil. Mulch base to prevent soil splash.",
    "Tomato___Late_blight": "Remove and discard affected plants. Use certified disease-free seeds and keep foliage dry.",
    "Tomato___healthy": "Vibrant and healthy foliage! Continue balanced watering and support with tomato stakes or cages."
}


@st.cache_resource
def get_predictor() -> PlantDiseasePredictor:
    """Cache model predictor instance"""
    return PlantDiseasePredictor(
        model_path=DEFAULT_MODEL_PATH,
        class_file=DEFAULT_CLASS_FILE,
        autoload=True
    )


def load_treatments() -> dict:
    """Load disease treatments mapping"""
    if os.path.exists(TREATMENTS_PATH):
        try:
            with open(TREATMENTS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_TREATMENTS


predictor = get_predictor()
treatments = load_treatments()

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
    padding: 1.25rem;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
    border-left: 5px solid #2e7d32;
}
.chat-bubble { padding: 0.6rem 1.2rem; border-radius: 12px; margin: 0.4rem 0; max-width: 75%; }
.user { background-color: #2e7d32; color: white; align-self: flex-end; }
.bot { background-color: #ffffff; color: #1b5e20; border: 1px solid #c8e6c9; align-self: flex-start; }
.chat-container { display: flex; flex-direction: column; }
.tip { margin-bottom: 0.7rem; padding: 0.8rem; border-radius: 10px; background: white; box-shadow: 0 2px 6px rgba(0,0,0,0.08); display: flex; align-items: center; }
.tip-icon { font-size: 1.3rem; margin-right: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ---------------- UI Layout ----------------
st.title("🌱 Plant Disease Detector & Care Assistant")
st.caption("AI-powered diagnostic assistant for crop health, disease classification, and treatment recommendations.")

if predictor.model is None:
    st.info(
        "ℹ️ **Demo/Pre-Training Mode Active:** Model weights (`models/plant_disease_model.keras`) are not currently placed in the repo. "
        "Run `python train.py` on your dataset to train the EfficientNetB0 model, or run `pytest` to verify the pipeline."
    )

tabs = st.tabs(["🩺 Detection", "💬 Care Chatbot", "💡 Best Practice Tips"])

# ---------------- Detection Tab ----------------
with tabs[0]:
    st.header("Upload Leaf Image for Analysis")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        uploaded_file = st.file_uploader("Upload a clear leaf photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="📷 Uploaded Leaf Image", use_container_width=True)

    with col2:
        if uploaded_file and st.button("🔍 Run Disease Diagnostics"):
            if predictor.model is None:
                st.warning("⚠️ Cannot run live inference: model weights file not found. Train the model using `python train.py`.")
            else:
                with st.spinner("Analyzing plant leaf features... 🌿"):
                    try:
                        disease, confidence = predictor.predict(image)
                        treatment = treatments.get(disease, "Consult local agricultural extension service for specialized treatment.")

                        st.session_state.last_disease = disease

                        # Sanitize outputs to prevent XSS
                        safe_disease = html.escape(disease)
                        safe_treatment = html.escape(treatment)
                        safe_confidence = f"{confidence:.1f}%"

                        st.markdown(f"""
                        <div class="card">
                            <h3>🦠 Diagnosis: <span style="color:#2e7d32">{safe_disease}</span></h3>
                            <p><b>🎯 Confidence:</b> {safe_confidence}</p>
                            <p><b>💊 Recommended Treatment:</b> {safe_treatment}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Inference error: {html.escape(str(e))}")

# ---------------- Chatbot Tab ----------------
with tabs[1]:
    st.header("Plant Care Chatbot 💬")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'last_disease' not in st.session_state:
        st.session_state.last_disease = None

    user_msg = st.text_input("Ask a plant care question:")
    if st.button("Send Message") and user_msg.strip() != "":
        raw_msg = user_msg.strip()
        fallback_knowledge = {
            "watering": "💧 Water early in the morning so foliage dries before nightfall, reducing fungal spore germination.",
            "sunlight": "☀️ Most fruit-bearing plants need 6-8 hours of direct sunlight daily for healthy disease resistance.",
            "fertilizer": "🌱 Avoid excessive nitrogen fertilizer as tender new growth is more susceptible to bacterial and fungal infections.",
            "yellow": "🍃 Yellowing leaves may indicate overwatering, nitrogen deficiency, or early root rot.",
            "blight": "🍄 For blight, immediately remove infected foliage, apply preventative copper-based spray, and avoid overhead water.",
            "spots": "🔬 Leaf spots are commonly caused by fungal spores splashing from soil. Mulch the soil bed and apply fungicide."
        }

        # Check if user asks about the most recently diagnosed disease
        if st.session_state.last_disease and st.session_state.last_disease.lower() in raw_msg.lower():
            bot_reply = treatments.get(st.session_state.last_disease, f"Follow standard sanitary pruning for {st.session_state.last_disease}.")
        else:
            bot_reply = None
            for kw, ans in fallback_knowledge.items():
                if kw in raw_msg.lower():
                    bot_reply = ans
                    break
            if not bot_reply:
                bot_reply = "🤖 I can help with plant diseases, watering schedules, sunlight needs, fertilizers, and treatments. Try asking about watering, blight, or yellow leaves!"

        st.session_state.chat_history.append((raw_msg, bot_reply))

    # Display sanitized chat
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for user_turn, bot_turn in st.session_state.chat_history:
        safe_user = html.escape(user_turn)
        safe_bot = html.escape(bot_turn)
        st.markdown(f'<div class="chat-bubble user">👩‍🌾 {safe_user}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-bubble bot">🤖 {safe_bot}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Tips Tab ----------------
with tabs[2]:
    st.header("💡 Preventative Plant Care Tips")

    tips = [
        ("💧", "Water at the soil level rather than over foliage to prevent humid conditions favoring fungal spore germination."),
        ("☀️", "Ensure adequate spacing between plants to maximize airflow and sunlight penetration."),
        ("🍃", "Regularly inspect the undersides of leaves where pests and early fungal pustules first develop."),
        ("🌱", "Sanitize pruning shears with 70% isopropyl alcohol between cuts to prevent cross-plant disease transmission.")
    ]

    for icon, tip in tips:
        safe_tip = html.escape(tip)
        st.markdown(f'<div class="tip"><span class="tip-icon">{icon}</span>{safe_tip}</div>', unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("<br><center><small>🌱 Plant Doctor AI — EfficientNetB0 Diagnostic System</small></center>", unsafe_allow_html=True)
