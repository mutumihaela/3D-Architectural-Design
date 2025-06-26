
import os
import json
import re
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import subprocess

# --- CONFIG ---
CSV_PATH = "/Users/mutumihaela/Desktop/blender/clip_scores_finetuned.csv"
OBJ_FOLDER = "/Users/mutumihaela/Desktop/blender/obj"
OUTPUT_JSON = "/Users/mutumihaela/Desktop/blender/placement_plan.json"
# sentence = "Put a sleek black office chair on the left and a wooden table on the right"
# sentence = "Place a fabric dining chair on the left and a round table in the corner."
sentence = "Put a coffee table in the center and a white fabric sofa near the window."
# sentence = "Put an armchair on the left and a rectangular coffee table in the corner."
# sentence = "Put an leather armchair on the right and a black table in the middle."
# sentence = "Place a wooden bunk bed near the wall and a vibrant colored bed with frame in the corner."
# sentence = "Put a red painted vintage bookshelf on the left and a high back executive chair in the center."


ssentence = "Place a black office desk in the center and a leather office chair on the left."

BLENDER_PATH = "/Applications/Blender.app/Contents/MacOS/Blender"
BLENDER_SCRIPT = "/Users/mutumihaela/Desktop/blender/render_scene.py"

# KEYWORDS 
POSITION_KEYWORDS = {
    "center": ["center", "middle"],
    "left": ["left"],
    "right": ["right"],
    "corner": ["corner"],
    "near wall": ["near wall"],
    "near window": ["near window"]
}

# Load model
print(" Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

df = pd.read_csv(CSV_PATH)
prompt_texts = df["prompt"].tolist()

# Pre-encode all prompts once
print(" Encoding all prompts...")
prompt_inputs = clip_processor(text=prompt_texts, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    prompt_embeddings = clip_model.get_text_features(**prompt_inputs)

# --- Extract (object, position) pairs from sentence ---
def extract_pairs(sentence):
    sentence = sentence.lower()
    parts = re.split(r"\s+and\s+", sentence)  # split by and

    pairs = []
    for part in parts:
        for pos, keywords in POSITION_KEYWORDS.items():
            for keyword in keywords:
                match = re.search(rf"([\w\s]{{3,40}}?)\s+(?:in|on)\s+the\s+{re.escape(keyword)}", part)
                if match:
                    obj_desc = match.group(1).replace("put", "").replace("place", "").strip()
                    pairs.append((obj_desc, pos))
                    break
    return pairs

# Pick best match using CLIP semantic score 
def select_best_file(description):
    print(f" Looking for: {description}")
    inputs = clip_processor(text=[description], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        desc_embed = clip_model.get_text_features(**inputs)

    similarities = cosine_similarity(desc_embed.numpy(), prompt_embeddings.numpy())[0]
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    if best_score < 0.2:
        print(f"[!] Low score ({best_score:.2f}) → skip")
        return None
    best_file = df.iloc[best_idx]["file"]
    print(f" Matched with '{df.iloc[best_idx]['prompt']}' ({best_score:.2f})")
    return best_file

# Main 
print(" Parsing sentence...")
pairs = extract_pairs(sentence)

print("📦 Extracted pairs:")
for d, p in pairs:
    print(f" - '{d}' → {p}")

result = []
for desc, pos in pairs:
    best_file = select_best_file(desc)
    if best_file:
        full_path = os.path.join(OBJ_FOLDER, best_file)
        result.append({"file": full_path, "position": pos})

with open(OUTPUT_JSON, "w") as f:
    json.dump(result, f, indent=2)

print(f"Saved placement plan with {len(result)} object(s)")
