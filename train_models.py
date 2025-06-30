import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

from app.model_utils import clean_text, embed_text

# 📂 Load dataset
df = pd.read_csv("final_complaints_dataset.csv")

# 🧹 Clean and Embed Text
print("\n🧹 Preprocessing and embedding...")
df["cleaned"] = df.apply(lambda row: clean_text(f"{row['room_number']} {row['text']}"), axis=1)
X_embed = embed_text(df["cleaned"].tolist())

# ======================= CATEGORY MODEL ========================
print("\n📦 Training Category Model")
y_category = df["category"]
category_le = joblib.load("app/models/category_label_encoder.joblib") if \
    os.path.exists("app/models/category_label_encoder.joblib") else LabelEncoder()
y_category = category_le.fit_transform(y_category)
joblib.dump(category_le, "app/models/category_label_encoder.joblib")

X_train, X_test, y_train, y_test = train_test_split(X_embed, y_category, test_size=0.2, random_state=42)
category_model = LogisticRegression(max_iter=1000)
category_model.fit(X_train, y_train)
joblib.dump(category_model, "app/models/category_model.joblib")
print(classification_report(y_test, category_model.predict(X_test), zero_division=0))

# ======================= URGENCY MODEL ========================
print("\n📦 Training Urgency Model")
y_urgency = df["urgency"]
urgency_le = joblib.load("app/models/urgency_label_encoder.joblib") if \
    os.path.exists("app/models/urgency_label_encoder.joblib") else LabelEncoder()
y_urgency = urgency_le.fit_transform(y_urgency)
joblib.dump(urgency_le, "app/models/urgency_label_encoder.joblib")

X_train, X_test, y_train, y_test = train_test_split(X_embed, y_urgency, test_size=0.2, random_state=42)
urgency_model = LogisticRegression(max_iter=1000)
urgency_model.fit(X_train, y_train)
joblib.dump(urgency_model, "app/models/urgency_model.joblib")
print(classification_report(y_test, urgency_model.predict(X_test), zero_division=0))

# ======================= SUMMARY MODEL ========================
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ✅ Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# ✅ Dataset class
class SummaryDataset(Dataset):
    def __init__(self, texts, summaries, tokenizer):
        self.inputs = tokenizer(["summarize: " + t for t in texts], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        self.targets = tokenizer(summaries, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
        self.targets["input_ids"][self.targets["input_ids"] == tokenizer.pad_token_id] = -100

    def __len__(self):
        return len(self.inputs.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs.input_ids[idx],
            "attention_mask": self.inputs.attention_mask[idx],
            "labels": self.targets.input_ids[idx]
        }

# ✅ Prepare dataset
dataset = SummaryDataset(df["text"].tolist(), df["summary"].tolist(), tokenizer)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# ✅ Setup training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

EPOCHS = 10
PATIENCE = 3
best_val_loss = float("inf")
patience_counter = 0

print("\n📦 Training T5 Summary Model")
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_train_loss += loss.item()

    # ✅ Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_val_loss += outputs.loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"📊 Epoch {epoch+1}/{EPOCHS} — Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # ✅ Show some generated samples
    if epoch % 1 == 0:
        test_text = df["text"].iloc[epoch % len(df)]
        input_ids = tokenizer("summarize: " + test_text, return_tensors="pt", truncation=True).input_ids.to(device)
        summary_ids = model.generate(input_ids, max_length=30)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print(f"📝 Sample [{epoch+1}]: {test_text[:50]}... → {summary}")

    # ✅ Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save_pretrained("app/models/summary_t5_model")
        tokenizer.save_pretrained("app/models/summary_t5_model")
        print("✅ New best model saved.")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("⛔ Early stopping triggered.")
            break

print("✅ T5 summary training complete.")

print("\n✅ All models trained and saved in app/models/")
