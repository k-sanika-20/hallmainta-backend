#!/usr/bin/env bash

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install gdown

# Create necessary folders
mkdir -p app/models/summary_t5_model

# Download joblib models
gdown --id 1ADgcS7CGdYE39JJW791pd-kxZWD428EB -O app/models/urgency_model.joblib
gdown --id 1UFJeaMjZzeUopg0NrxL6hfkKBLml5hfZ -O app/models/category_label_encoder.joblib
gdown --id 1rh81AX_DLsdNUp_-1DEU2m95MhVxtA6a -O app/models/category_model.joblib
gdown --id 1if2jSI5RuSxulNdWxZLj3Fa3LQde4fai -O app/models/urgency_label_encoder.joblib

# Download T5 summarizer model files
gdown --id 1ekGuNrEYtmsdhfo1aTwrWzRZBqkvWulR -O app/models/summary_t5_model/config.json
gdown --id 19rm02Yw5DHsuCXtM9iYsWe3I-0zR2U5L -O app/models/summary_t5_model/pytorch_model.bin
gdown --id 1JI5OXncSI3DD_PSbZb7e3k3hUvvdHqNm -O app/models/summary_t5_model/special_tokens_map.json
gdown --id 14IGT4EcR4RsD2ORQMVCvuYYf5Qzx7vhM -O app/models/summary_t5_model/spiece.model
gdown --id 1-9iT9rcPntIEcMA2OoAbBdGKzJTU4SgU -O app/models/summary_t5_model/tokenizer_config.json
gdown --id 1yqVwHCRYTuvNAv-1j-PCA07xDQ4FTbjA -O app/models/summary_t5_model/added_tokens.json

# ✅ Debug: List all files in app/models after download
echo "✅ Files in app/models:"
ls -R app/models


