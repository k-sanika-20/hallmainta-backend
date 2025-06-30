import re
import numpy as np
from sentence_transformers import SentenceTransformer

# Load sentence transformer model
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Clean and preprocess text
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ✅ Generate sentence embeddings
def embed_text(texts: list[str]) -> np.ndarray:
    return sentence_model.encode(texts)

# ✅ Infer location from room and complaint description
def infer_location(room_number: str, description: str) -> str:
    description = description.lower().strip()
    room_match = re.match(r"([A-Z]+)[- ]?(\d{2,3})$", room_number.upper())

    if not room_match:
        return "Unknown"

    block = room_match.group(1)
    number = room_match.group(2)

    if len(number) <= 2:
        floor_label = None  # No floor information
    else:
        floor_num = int(number[0])
        floor_map = {1: "ground", 2: "1st", 3: "2nd", 4: "3rd"}
        floor_label = floor_map.get(floor_num, f"{floor_num}th")

    room_keywords = ["fan", "bulb", "light", "wall", "painting", "ceiling", "cupboard", "bed", "cot", "table", "wardrobe"]
    washroom_keywords = ["washroom", "geyser", "basin", "sink", "toilet", "flush","lavatory"]
    corridor_keywords = ["corridor"]
    common_area_keywords = [
        "gym", "music room", "common room", "study room", "zen lounge",
        "library", "manager office", "warden office", "mess", "basketball court"
    ]

    if any(word in description for word in room_keywords):
        return room_number.upper()

    if any(word in description for word in washroom_keywords):
        if floor_label:
            return f"{block} Block {floor_label} floor washroom"
        return f"{block} Block washroom"

    if any(word in description for word in corridor_keywords):
        if floor_label:
            return f"{block} Block {floor_label} floor corridor"
        return f"{block} Block corridor"

    for area in common_area_keywords:
        if area in description:
            return area.title()

    if floor_label:
        return f"{block} Block {floor_label} floor"
    return f"{block} Block"
