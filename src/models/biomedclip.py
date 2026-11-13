"""Frozen official BiomedCLIP zero-shot inference."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import torch
from PIL import Image


MODEL_IDENTIFIER = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CLASS_MAPPING = {"Normal": 0, "Benign": 1, "Malignant": 2}


@dataclass(frozen=True)
class BiomedCLIPOutput:
    sample_id: str
    condition: str
    ground_truth: int
    predicted_class: str
    scores: dict[str, float]
    confidence: float
    model_identifier: str
    prompt_version: str


def load_prompts(path):
    import yaml
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    classes = config["classes"]
    if list(classes) != list(CLASS_MAPPING) or len({len(v) for v in classes.values()}) != 1:
        raise ValueError("BiomedCLIP prompts must contain the three classes equally")
    if not all(classes.values()):
        raise ValueError("Each BiomedCLIP class needs at least one prompt")
    return config


class FrozenBiomedCLIP:
    def __init__(self, config, device=None):
        from open_clip import create_model_from_pretrained, get_tokenizer
        self.config = config
        self.model_identifier = config["model"]["identifier"]
        if self.model_identifier != MODEL_IDENTIFIER:
            raise ValueError("Only the official BiomedCLIP model is supported")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = create_model_from_pretrained(self.model_identifier)
        self.tokenizer = get_tokenizer(self.model_identifier)
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.prompt_version = config["model"]["prompt_version"]
        self.class_embeddings = self._encode_prompts(config["classes"])
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("BiomedCLIP parameters were not fully frozen")

    def _encode_prompts(self, prompts):
        embeddings = []
        with torch.inference_mode():
            for class_name in CLASS_MAPPING:
                tokens = self.tokenizer(prompts[class_name]).to(self.device)
                values = self.model.encode_text(tokens)
                values = values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                aggregate = values.mean(dim=0)
                embeddings.append(aggregate / aggregate.norm().clamp_min(1e-12))
        return torch.stack(embeddings)

    def predict_batch(self, images, rows, condition):
        with torch.inference_mode():
            inputs = torch.stack([self.preprocess(image) for image in images]).to(self.device)
            image_embeddings = self.model.encode_image(inputs)
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            scores = image_embeddings @ self.class_embeddings.T
            predictions = scores.argmax(dim=1).cpu().tolist()
            score_values = scores.cpu().tolist()
        output = []
        names = list(CLASS_MAPPING)
        for row, prediction, values in zip(rows, predictions, score_values):
            output.append(BiomedCLIPOutput(
                row.get("sample_id", row["source_relative_path"]), condition,
                int(row["class_index"]), names[prediction],
                {name: float(value) for name, value in zip(names, values)},
                float(max(values)), self.model_identifier, self.prompt_version,
            ))
        return output

    def metadata(self):
        return {
            "model_identifier": self.model_identifier,
            "prompt_version": self.prompt_version,
            "class_mapping": CLASS_MAPPING,
            "aggregation": self.config["model"]["aggregation"],
            "device": str(self.device),
            "parameters_frozen": all(not p.requires_grad for p in self.model.parameters()),
            "training_mode": self.model.training,
        }


def output_records(outputs):
    return [
        {
            "sample_id": item.sample_id, "condition": item.condition,
            "ground_truth": item.ground_truth, "predicted_class": item.predicted_class,
            "normal_score": item.scores["Normal"], "benign_score": item.scores["Benign"],
            "malignant_score": item.scores["Malignant"], "confidence": item.confidence,
            "model_identifier": item.model_identifier, "prompt_version": item.prompt_version,
        }
        for item in outputs
    ]
