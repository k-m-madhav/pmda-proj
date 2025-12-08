from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "railsem19" / "jpgs"
OUT_CSV = ROOT / "weather_scenarios_snowclip.csv"

labels = ["foggy scene", "clear sunny scene", "night scene", "rainy weather","snowy scene"]

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def classify_image(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
    except:
        return None, None

    inputs = processor(text=labels, images=img, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]
    best_idx = probs.argmax().item()

    return labels[best_idx], probs[best_idx].detach().item()

def run_classification():
    results = []

    img_files = list(IMG_DIR.rglob("*.jpg"))

    print(f"Found {len(img_files)} images.")

    for img_path in img_files:
        label, confidence = classify_image(img_path)
        results.append({
            "image_rel_path": str(img_path.relative_to(IMG_DIR)),
            "scenario": label,
            "confidence": confidence
        })

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved to {OUT_CSV}")

if __name__ == "__main__":
    run_classification()