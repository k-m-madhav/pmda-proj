from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch


class HazardCLIP:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Hazard detection labels
        self.labels = [
            "clear railway track",
            "tree blocking the track",
            "fallen tree across the railway track",
            "railway track obstruction",
            "object blocking railway track",
            "snow blocking the track",
            "train ahead on the track",
            "vegetation covering the railway track",
            "car/vehicle on track"
        ]

    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")
        return self._predict_from_image(img)

    def predict_image(self, img: Image.Image):
        """
        Run CLIP hazard detection on an in-memory PIL image.
        """
        return self._predict_from_image(img.convert("RGB"))

    def _predict_from_image(self, img: Image.Image):
        inputs = self.processor(
            text=self.labels,
            images=img,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
        best_idx = probs.argmax().item()

        return {
            "label": self.labels[best_idx],
            "confidence": float(probs[best_idx])
        }
