import joblib
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
from app.model_utils import clean_text, embed_text, infer_location

# === Model globals ===
category_model = None
category_le = None
urgency_model = None
urgency_le = None
summary_tokenizer = None
summary_model = None

# === Base path to models directory ===
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
SUMMARY_DIR = os.path.join(MODELS_DIR, "summary_t5_model")

# === Helper to safely load .joblib files ===
def safe_load_joblib(filename):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

# === Load models (called on startup) ===
def load_models():
    global category_model, category_le, urgency_model, urgency_le
    global summary_tokenizer, summary_model

    print("🔄 Loading AI models...")

    category_model = safe_load_joblib("category_model.joblib")
    category_le = safe_load_joblib("category_label_encoder.joblib")
    urgency_model = safe_load_joblib("urgency_model.joblib")
    urgency_le = safe_load_joblib("urgency_label_encoder.joblib")

    if not os.path.exists(SUMMARY_DIR):
        raise FileNotFoundError(f"T5 model directory not found: {SUMMARY_DIR}")

    summary_tokenizer = T5Tokenizer.from_pretrained(SUMMARY_DIR)
    summary_model = T5ForConditionalGeneration.from_pretrained(SUMMARY_DIR)

    print("✅ All AI models loaded successfully.")

# === Prediction function ===
def predict_complaint_metadata(room_number: str, text: str):
    if not all([category_model, category_le, urgency_model, urgency_le, summary_tokenizer, summary_model]):
        raise RuntimeError("❌ AI models not loaded. Call load_models() first.")

    input_text = clean_text(f"{room_number} {text}")
    embedded = embed_text([input_text])

    category = category_le.inverse_transform(category_model.predict(embedded))[0]
    urgency = urgency_le.inverse_transform(urgency_model.predict(embedded))[0]
    location = infer_location(room_number, text)

    input_ids = summary_tokenizer("summarize: " + text, return_tensors="pt", truncation=True).input_ids
    output_ids = summary_model.generate(input_ids, max_length=30, num_beams=2, early_stopping=True)
    summary = summary_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return category, urgency, location, summary
