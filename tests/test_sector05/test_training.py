import torch
from torch import nn

from training.engine import EarlyStopping, TrainingEngine, build_loss
from training.metrics import classification_metrics


class TinyModel(nn.Module):
    model_name = "tiny"
    num_classes = 3

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 3)

    def forward(self, x):
        return self.layer(x)


def test_metrics_and_early_stopping():
    metrics = classification_metrics([0, 1, 2], [0, 1, 1])
    assert metrics["accuracy"] == 2 / 3
    stopping = EarlyStopping(patience=2, min_delta=0.1)
    assert not stopping.update(0.5)
    assert not stopping.update(0.51)
    assert stopping.update(0.52)


def test_training_engine_checkpoint_and_resume(tmp_path):
    config = {
        "max_epochs": 1,
        "mixed_precision": True,
        "optimizer": {"name": "adamw", "learning_rate": 0.001},
        "early_stopping_patience": 3,
    }
    engine = TrainingEngine(TinyModel(), config, device=torch.device("cpu"), output_dir=tmp_path)
    train = [(torch.randn(4, 3), torch.tensor([0, 1, 2, 1]))]
    validation = [(torch.randn(4, 3), torch.tensor([0, 1, 2, 1]))]
    history = engine.fit(train, validation, nn.CrossEntropyLoss())
    assert history and (tmp_path / "best.pt").is_file() and (tmp_path / "last.pt").is_file()
    restored = TrainingEngine(TinyModel(), config, device=torch.device("cpu"), output_dir=tmp_path)
    epoch, _ = restored.load_checkpoint(tmp_path / "last.pt")
    assert epoch == 1


def test_weighted_loss_requires_training_counts():
    try:
        build_loss(TinyModel(), {"loss": {"class_weighted": True}})
    except ValueError as error:
        assert "class_counts" in str(error)
    else:
        raise AssertionError("weighted loss accepted without training counts")
