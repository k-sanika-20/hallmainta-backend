#!/usr/bin/env bash

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install gdown

# Create necessary folders
mkdir -p app/models/summary_t5_model

# Download joblib models
gdown --id 1mVNeebLrFRtsOHP-k93ARfw6ZwSq03Pq -O app/models/urgency_model.joblib
gdown --id 1_Q9R8Cy6j6JZecKSaveSFpvUSdO2k6-E -O app/models/category_label_encoder.joblib
gdown --id 1urFZejqvFZBZZM46wyhNAXe5Ux1DW0uB -O app/models/category_model.joblib
gdown --id 1pG6ShUpZkBJ8wB57oGPVx0c51xUI2Xh7 -O app/models/urgency_label_encoder.joblib

# Download T5 summarizer model files
gdown --id 1wzu8Uk0qBdTj_8AtJZwCfPpJhQQ5X-bA -O app/models/summary_t5_model/config.json
gdown --id 1ts42zhrP0I845m7UMOGllmrzZ6CozH98 -O app/models/summary_t5_model/generation_config.json
gdown --id 119U0-s0bGHlMM7d8JUcmgeganRm5k4_N -O app/models/summary_t5_model/special_tokens_map.json
gdown --id 1cn6l5208oU9XbdmT1in9wdVgIm9hDK6q -O app/models/summary_t5_model/spiece.model
gdown --id 1za7mucMjMLmOkT7M2gyA_Yjcymvj02jY -O app/models/summary_t5_model/tokenizer_config.json
gdown --id 119U0-s0bGHlMM7d8JUcmgeganRm5k4_N -O app/models/summary_t5_model/special_tokens_map.json
gdown --id 1FIf4DLPQVGOBQLeW9dSEZLXVCv9s-yjW -O app/models/summary_t5_model/added_tokens.json

