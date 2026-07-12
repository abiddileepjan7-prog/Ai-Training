
import warnings
warnings.filterwarnings("ignore")

from transformers import pipeline
from sentence_transformers import SentenceTransformer, util


# ---------------------------------------------------------------------------
# 1. SLM - Small Language Model -> predict sentiment of a sentence
# ---------------------------------------------------------------------------
print("\n=== SLM: sentiment prediction ===")
slm = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
text = "I really loved how smooth this product felt to use."
result = slm(text)[0]
print("input:", text)
print("predicted label:", result["label"], "| score:", round(result["score"], 4))


# ---------------------------------------------------------------------------
# 2. MLM - Masked Language Model -> predict the missing word
# ---------------------------------------------------------------------------
print("\n=== MLM: masked word prediction ===")
mlm = pipeline("fill-mask", model="bert-base-uncased")
text = "The capital of France is [MASK]."
result = mlm(text)[0]
print("input:", text)
print("top prediction:", result["token_str"], "| score:", round(result["score"], 4))


# ---------------------------------------------------------------------------
# 3. SAM - Segment Anything Model -> segment an image (categorize pixels)
# ---------------------------------------------------------------------------
print("\n=== SAM: image segmentation ===")
from transformers import SamModel, SamProcessor
from PIL import Image, ImageDraw
import torch

# Simple synthetic image so the demo runs without needing a real file.
image = Image.new("RGB", (256, 256), color=(200, 200, 200))
ImageDraw.Draw(image).ellipse((60, 60, 180, 180), fill=(30, 120, 200))

sam_model = SamModel.from_pretrained("facebook/sam-vit-base")
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

input_points = [[[128, 128]]]  # click at the center of the image
inputs = sam_processor(image, input_points=input_points, return_tensors="pt")
with torch.no_grad():
    outputs = sam_model(**inputs)

scores = outputs.iou_scores.squeeze().tolist()
print("image size:", image.size)
print("mask quality scores:", [round(s, 4) for s in scores])


# ---------------------------------------------------------------------------
# 4. LAM - Large Action Model -> categorize an instruction into an action
# ---------------------------------------------------------------------------
print("\n=== LAM: instruction -> action categorization ===")
lam = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
instruction = "Remind me to call mom at 6pm"
actions = ["open_app", "send_message", "search_web", "set_reminder", "play_music", "unknown"]
result = lam(instruction, candidate_labels=actions)
print("instruction:", instruction)
print("predicted action:", result["labels"][0], "| score:", round(result["scores"][0], 4))


# ---------------------------------------------------------------------------
# 5. LCM - Large / Latent Concept Model -> categorize text into a concept
# ---------------------------------------------------------------------------
print("\n=== LCM: text -> concept categorization ===")
lcm = SentenceTransformer("all-MiniLM-L6-v2")
text = "The central bank raised interest rates to curb inflation."
concepts = ["finance", "health", "technology", "sports", "politics", "entertainment"]

text_emb = lcm.encode(text, convert_to_tensor=True)
concept_embs = lcm.encode(concepts, convert_to_tensor=True)
scores = util.cos_sim(text_emb, concept_embs)[0]
best_idx = int(scores.argmax())

print("input:", text)
print("predicted concept:", concepts[best_idx], "| score:", round(float(scores[best_idx]), 4))