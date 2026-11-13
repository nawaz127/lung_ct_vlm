"""Report the configured Python, PyTorch, and CUDA runtime."""

import argparse
import json
from pathlib import Path

from utils.configuration import load_project_config
from utils.environment import validate_environment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-config", default="configs/project.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_project_config(root / args.project_config)
    report = validate_environment(
        tuple(int(value) for value in str(config["python_min_version"]).split(".")),
        config["device"],
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
