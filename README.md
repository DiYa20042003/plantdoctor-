# 🌱 Plant Doctor: Transfer-Learning Crop Disease Diagnostic System

An AI-powered computer vision and diagnostic assistant for detecting crop leaf diseases and recommending targeted agricultural care using EfficientNetB0 transfer learning.

---

## 📌 Overview

Agricultural leaf diseases cause significant yield loss worldwide. **Plant Doctor** provides an end-to-end computer vision pipeline and web application designed to:
- **Classify crop leaf pathologies** across 14+ conditions (including early blight, late blight, apple scab, black rot, rust, and healthy crops).
- **Deliver immediate care and treatment protocols** for detected diseases.
- **Provide an interactive care chatbot** offering watering, sunlight, soil, and preventative guidance.

---

## 🏗️ Architecture & Model Pipeline

```mermaid
flowchart LR
    A[Leaf Image 128x128x3] --> B[EfficientNet Preprocessing]
    B --> C[EfficientNetB0 Base (ImageNet Frozen)]
    C --> D[GlobalAveragePooling2D]
    D --> E[Dropout 0.3]
    E --> F[Dense Softmax Classifier]
    F --> G[Diagnosis + Confidence Score]
    G --> H[Treatment Recommendation & Care Guidance]
```

### Key Engineering Decisions:
1. **Transfer Learning Backbone:** Pretrained `EfficientNetB0` on ImageNet provides high feature representation with lower parameter overhead compared to ResNet-50.
2. **Double-Softmax Elimination:** Ensures single-stage probability computation preserving honest confidence intervals rather than compressing probability distribution.
3. **Defensive Web UI:** Streamlit interface features HTML entity escaping against XSS injection, responsive layout, and graceful degradation when running without pre-cached model weights.
4. **Reproducible Evaluation Pipeline:** Dedicated `evaluate.py` calculating macro precision, recall, F1-score, and top-1 accuracy on independent test splits.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Virtual environment (`venv`)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/DiYa20042003/plantdoctor-.git
cd plantdoctor-

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

Organize your leaf image dataset (e.g. [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)) into class subdirectories:
```
data/
  dataset/
    Apple___Apple_scab/
    Apple___Black_rot/
    Apple___healthy/
    Tomato___Early_blight/
    Tomato___healthy/
    ...
```

Run training:
```bash
python train.py --data-dir data/dataset --epochs 15 --batch-size 16 --img-size 128
```
This saves:
- Model checkpoint to `models/plant_disease_model.keras`
- Class labels mapping to `models/class_names.json`

---

## 📊 Evaluation & Verification

Evaluate model performance on an independent evaluation set:
```bash
python evaluate.py --data-dir data/test --output-json evaluation_results.json
```

Outputs:
- **Accuracy**
- **Macro Precision**
- **Macro Recall**
- **Macro F1-Score**

---

## 💻 Running the Web Application

Launch the Streamlit interactive dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Automated Testing

Run the test suite verifying inference preprocessing, probability scaling, error boundaries, and input sanitization:
```bash
pytest tests/ -v
```

---

## 🛡️ Limitations & Honest Considerations

- **Dataset Bias:** Performance is dependent on training dataset lighting and laboratory vs. field-condition leaf captures.
- **Microbial Confirmation:** Computer vision diagnoses should be corroborated by local agricultural extension officers before applying expensive chemical fungicides.
- **Model Storage:** Large binary `.keras` checkpoints are excluded from Git version control via `.gitignore` and should be stored via Git LFS or object storage.

---

## 📄 License
MIT License
