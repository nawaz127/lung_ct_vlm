import torch

from models.biomedclip import CLASS_MAPPING, FrozenBiomedCLIP, load_prompts


def test_prompt_configuration_is_balanced():
    config = load_prompts("configs/biomedclip.yaml")
    assert list(config["classes"]) == list(CLASS_MAPPING)
    assert {len(prompts) for prompts in config["classes"].values()} == {3}
    assert config["model"]["prompt_version"] == "sector07_v1"


def test_frozen_wrapper_with_mock_model():
    wrapper = object.__new__(FrozenBiomedCLIP)
    wrapper.model_identifier = "test"
    wrapper.prompt_version = "test"
    wrapper.device = torch.device("cpu")
    wrapper.config = {"model": {"aggregation": "mean_then_normalize"}}
    wrapper.model = torch.nn.Linear(2, 2)
    wrapper.model.eval()
    for parameter in wrapper.model.parameters():
        parameter.requires_grad = False
    metadata = wrapper.metadata()
    assert metadata["parameters_frozen"]
    assert metadata["training_mode"] is False
