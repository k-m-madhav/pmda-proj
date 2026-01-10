from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch


class HazardCLIP:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Hazard detection labels - focus on BLOCKING hazards, not proximity
        self.labels = [
            "clear railway track with vegetation on the sides",
            "tree fallen across and blocking the railway track",
            "large tree trunk blocking the railway track",
            "railway track completely blocked by obstruction",
            "object directly blocking railway track",
            "flooded railway track with water covering rails",
            "landslide debris blocking railway track",
            "light snow on the track",
            "heavy snow completely covering and blocking the track",
            "train directly ahead blocking the track",
            "vegetation overgrown blocking the railway track",
            "car or vehicle stopped on railway track",
            "people standing on railway track"
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
