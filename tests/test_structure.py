"""Validate the required project scaffold without requiring data or model weights."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_root_files_exist() -> None:
    required = {
        "README.md",
        "PROJECT_STRUCTURE.md",
        "SECTOR_STATUS.md",
        "requirements.txt",
        "environment.yml",
        ".gitignore",
        ".env.example",
        "pyproject.toml",
        "pytest.ini",
        "LICENSE",
        "CHANGELOG.md",
    }
    assert {path.name for path in ROOT.iterdir() if path.is_file()} >= required


def test_all_sector_directories_exist() -> None:
    for sector in range(1, 11):
        assert (ROOT / "tests" / f"test_sector{sector:02d}").is_dir()


def test_configuration_templates_exist() -> None:
    names = {
        "project",
        "dataset",
        "preprocessing",
        "models",
        "training",
        "corruptions",
        "vlm",
        "evaluation",
        "xai",
        "logging",
    }
    assert {path.stem for path in (ROOT / "configs").glob("*.yaml")} >= names