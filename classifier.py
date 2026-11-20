"""Hugging Face sketch model loading and inference helpers."""

from __future__ import annotations

import os

import numpy as np
from PIL import Image


HUGGINGFACE_MODEL_ID = "kmewhort/beit-sketch-classifier"

# Common QuickDraw classes translated for the German app. Other categories are
# kept in the model's original English wording so every output stays traceable.
GERMAN_LABELS = {
    "The Eiffel Tower": "Eiffelturm",
    "The Great Wall of China": "Chinesische Mauer",
    "The Mona Lisa": "Mona Lisa",
    "airplane": "Flugzeug",
    "apple": "Apfel",
    "bench": "Bank",
    "bicycle": "Fahrrad",
    "bridge": "Brücke",
    "building": "Gebäude",
    "bus": "Bus",
    "car": "Auto",
    "castle": "Schloss",
    "chair": "Stuhl",
    "cloud": "Wolke",
    "flower": "Blume",
    "forest": "Wald",
    "house": "Haus",
    "mountain": "Berg",
    "park": "Park",
    "person": "Person",
    "river": "Fluss",
    "road": "Straße",
    "star": "Stern",
    "tree": "Baum",
    "umbrella": "Regenschirm",
    "walking": "Gehweg",
    "wheel": "Rad",
}


def load_model(model_id: str = HUGGINGFACE_MODEL_ID):
    """Download and load a public sketch classifier from Hugging Face."""
    os.environ.setdefault("USE_TF", "0")
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()
    return processor, model


def classify_crop(model, crop_rgb: np.ndarray) -> tuple[str, float]:
    """Classify a single RGB drawing crop and return its label and confidence."""
    if crop_rgb.size == 0 or crop_rgb.shape[0] < 8 or crop_rgb.shape[1] < 8:
        raise ValueError("Der erkannte Ausschnitt ist zu klein.")

    processor, network = model
    image = Image.fromarray(np.asarray(crop_rgb[:, :, :3], dtype=np.uint8), mode="RGB")
    inputs = processor(images=image, return_tensors="pt")

    import torch

    with torch.inference_mode():
        logits = network(**inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1)
    if probabilities.numel() != len(network.config.id2label):
        raise ValueError("Das Hugging-Face-Modell liefert eine unerwartete Anzahl an Klassen.")
    if not torch.isfinite(probabilities).all():
        raise ValueError("Das Modell hat ungültige Konfidenzwerte geliefert.")

    index = int(torch.argmax(probabilities).item())
    raw_label = network.config.id2label[index]
    label = GERMAN_LABELS.get(raw_label, raw_label)
    return label, float(probabilities[index].item())
