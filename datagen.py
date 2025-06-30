import pandas as pd
from transformers import pipeline

# Load your dataset
df = pd.read_csv("final_complaints_dataset.csv")

# Load the T5 paraphrasing model
paraphraser = pipeline("text2text-generation", model="ramsrigouthamg/t5_paraphraser", framework="pt")

# Function to generate paraphrases
def generate_paraphrases(text, num_return_sequences=2):
    prompt = f"paraphrase: {text} </s>"
    outputs = paraphraser(prompt, max_length=128, num_return_sequences=num_return_sequences, do_sample=True)
    return [o['generated_text'] for o in outputs]

# Generate augmented complaints
augmented = []
for _, row in df.iterrows():
    try:
        new_texts = generate_paraphrases(row['text'], num_return_sequences=2)
        for new_text in new_texts:
            augmented.append({
                "room_number": row["room_number"],
                "text": new_text,
                "category": row["category"],
                "urgency": row["urgency"],
                "location": row["location"],
                "summary": row["summary"]
            })
        if len(augmented) >= 1000:
            break
    except Exception as e:
        continue

# Save to CSV
pd.DataFrame(augmented[:1000]).to_csv("augmented_complaints.csv", index=False)
print("✅ Augmented complaints saved.")
