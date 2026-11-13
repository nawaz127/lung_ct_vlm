"""Configurable CNN training engine for Sector 05."""

from dataclasses import dataclass
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from models import create_model
from ..metrics import classification_metrics


@dataclass
class EarlyStopping:
    patience: int = 7
    min_delta: float = 0.0001
    best: float = float("-inf")
    bad_epochs: int = 0

    def update(self, value):
        if value > self.best + self.min_delta:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def build_optimizer(model, config):
    optimizer = config.get("optimizer", {})
    if optimizer.get("name", "adamw").lower() != "adamw":
        raise ValueError("Only AdamW is supported by the Sector 05 default engine")
    return torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(optimizer.get("learning_rate", 0.0001)),
        weight_decay=float(optimizer.get("weight_decay", 0.0001)),
        betas=tuple(optimizer.get("betas", [0.9, 0.999])),
        eps=float(optimizer.get("eps", 1e-8)),
    )


def build_loss(model, config, class_counts=None):
    loss_config = config.get("loss", {})
    if loss_config.get("name", "cross_entropy") != "cross_entropy":
        raise ValueError("Only cross_entropy is supported by the Sector 05 engine")
    weights = None
    if loss_config.get("class_weighted", False):
        if not class_counts:
            raise ValueError("class_counts from training records are required for weighted loss")
        total = sum(class_counts)
        weights = torch.tensor(
            [total / max(1, count) for count in class_counts], dtype=torch.float32
        )
        weights = weights / weights.mean()
    return nn.CrossEntropyLoss(weight=weights)


def _autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def run_epoch(model, loader, loss_fn, optimizer=None, device=None, scaler=None, amp_enabled=False):
    training = optimizer is not None
    model.train(training)
    total_loss, targets, predictions = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        try:
            with _autocast(device, amp_enabled):
                logits = model(images)
                loss = loss_fn(logits, labels)
        except RuntimeError as error:
            if "out of memory" in str(error).lower() and device.type == "cuda":
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA out of memory during training. Lower the configured batch_size "
                    "and restart or resume explicitly; batch size was not changed automatically."
                ) from error
            raise
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss encountered")
        if training:
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        total_loss += loss.detach().float().item() * labels.size(0)
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(logits.detach().float().argmax(1).cpu().tolist())
    metrics = classification_metrics(targets, predictions, getattr(model, "num_classes", 3))
    metrics["loss"] = total_loss / max(1, len(targets))
    return metrics


class TrainingEngine:
    def __init__(self, model, config, device=None, output_dir="experiments", checkpoint_metadata=None):
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = build_optimizer(model, config)
        scheduler_config = config.get("scheduler", {})
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode=scheduler_config.get("mode", "max"),
            factor=float(scheduler_config.get("factor", 0.5)),
            patience=int(scheduler_config.get("patience", 2)),
            min_lr=float(scheduler_config.get("min_lr", 1e-6)),
        )
        self.amp_enabled = bool(config.get("mixed_precision", True) and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.early_stopping = EarlyStopping(
            int(config.get("early_stopping_patience", 7)),
            float(config.get("early_stopping_min_delta", 0.0001)),
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_metadata = checkpoint_metadata or {}

    def save_checkpoint(self, name, epoch, history):
        path = self.output_dir / name
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "epoch": epoch,
            "best_val_macro_f1": self.early_stopping.best,
            "early_stopping": self.early_stopping.__dict__,
            "history": history,
            "model_name": self.model.model_name,
            "num_classes": self.model.num_classes,
            "config": self.config,
            "metadata": self.checkpoint_metadata,
        }, path)
        return path

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("model_name") != self.model.model_name or checkpoint.get("num_classes") != self.model.num_classes:
            raise ValueError("Checkpoint is incompatible with the current model")
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.early_stopping = EarlyStopping(**checkpoint["early_stopping"])
        return checkpoint["epoch"], checkpoint.get("history", [])

    def fit(self, train_loader, validation_loader, loss_fn, start_epoch=0, history=None):
        max_epochs = int(self.config.get("max_epochs", 20))
        history = list(history or [])
        for epoch in range(start_epoch, max_epochs):
            train = run_epoch(self.model, train_loader, loss_fn, self.optimizer, self.device, self.scaler, self.amp_enabled)
            with torch.no_grad():
                validation = run_epoch(self.model, validation_loader, loss_fn, device=self.device, amp_enabled=self.amp_enabled)
            self.scheduler.step(validation["macro_f1"])
            record = {"epoch": epoch + 1, "train": train, "validation": validation,
                      "learning_rate": self.optimizer.param_groups[0]["lr"]}
            history.append(record)
            self.save_checkpoint("last.pt", epoch + 1, history)
            if validation["macro_f1"] >= self.early_stopping.best:
                self.save_checkpoint("best.pt", epoch + 1, history)
            if self.early_stopping.update(validation["macro_f1"]):
                break
        (self.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        return history
