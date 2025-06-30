import joblib
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
from app.model_utils import clean_text, embed_text, infer_location

# === Load Category and Urgency Models ===
category_model = joblib.load("app/models/category_model.joblib")
category_le = joblib.load("app/models/category_label_encoder.joblib")

urgency_model = joblib.load("app/models/urgency_model.joblib")
urgency_le = joblib.load("app/models/urgency_label_encoder.joblib")

# === Load T5 Summary Model ===
summary_tokenizer = T5Tokenizer.from_pretrained("app/models/summary_t5_model")
summary_model = T5ForConditionalGeneration.from_pretrained("app/models/summary_t5_model")

# === Prediction Function ===
def predict_complaint_metadata(room_number: str, text: str):
    input_text = clean_text(f"{room_number} {text}")
    embedded = embed_text([input_text])

    category = category_le.inverse_transform(category_model.predict(embedded))[0]
    urgency = urgency_le.inverse_transform(urgency_model.predict(embedded))[0]
    location = infer_location(room_number, text)

    # Generate summary with T5
    input_ids = summary_tokenizer("summarize: " + text, return_tensors="pt", truncation=True).input_ids
    output_ids = summary_model.generate(input_ids, max_length=30, num_beams=2, early_stopping=True)
    summary = summary_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return category, urgency, location, summary


def load_models():
    print("Models loaded.")  # Already loaded at import time
