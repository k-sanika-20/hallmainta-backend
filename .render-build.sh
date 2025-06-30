#!/usr/bin/env bash
# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create directories if not present
mkdir -p app/models/summary_t5_model

# Download model files from Google Drive
curl -L -o app/models/category_model.joblib "https://drive.google.com/uc?export=download&id=1mVNeebLrFRtsOHP-k93ARfw6ZwSq03Pq"
curl -L -o app/models/category_label_encoder.joblib "https://drive.google.com/uc?export=download&id=1pG6ShUpZkBJ8wB57oGPVx0c51xUI2Xh7"
curl -L -o app/models/urgency_model.joblib "https://drive.google.com/uc?export=download&id=1urFZejqvFZBZZM46wyhNAXe5Ux1DW0uB"
curl -L -o app/models/urgency_label_encoder.joblib "https://drive.google.com/uc?export=download&id=1_Q9R8Cy6j6JZecKSaveSFpvUSdO2k6-E"

# T5 Summary model files
curl -L -o app/models/summary_t5_model/config.json "https://drive.google.com/uc?export=download&id=1za7mucMjMLmOkT7M2gyA_Yjcymvj02jY"
curl -L -o app/models/summary_t5_model/pytorch_model.bin "https://drive.google.com/uc?export=download&id=1cn6l5208oU9XbdmT1in9wdVgIm9hDK6q"
curl -L -o app/models/summary_t5_model/spiece.model "https://drive.google.com/uc?export=download&id=119U0-s0bGHlMM7d8JUcmgeganRm5k4_N"
curl -L -o app/models/summary_t5_model/tokenizer_config.json "https://drive.google.com/uc?export=download&id=1eYiCmWxy-wO3W3ZfvAc8cadTX_mnHmLU"
curl -L -o app/models/summary_t5_model/tokenizer.json "https://drive.google.com/uc?export=download&id=1ts42zhrP0I845m7UMOGllmrzZ6CozH98"
curl -L -o app/models/summary_t5_model/special_tokens_map.json "https://drive.google.com/uc?export=download&id=1wzu8Uk0qBdTj_8AtJZwCfPpJhQQ5X-bA"
curl -L -o app/models/summary_t5_model/generation_config.json "https://drive.google.com/uc?export=download&id=1FIf4DLPQVGOBQLeW9dSEZLXVCv9s-yjW"
