"""Deterministic, read-only discovery of class-organized image files."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageCandidate:
    path: Path
    class_name: str
    relative_path: str


@dataclass(frozen=True)
class DiscoveryResult:
    candidates: list[ImageCandidate]
    missing_classes: list[str]
    unexpected_classes: list[str]
    empty_classes: list[str]
    unsupported_files: list[str]
    dataset_root_exists: bool


def discover_dataset(
    root: Path,
    expected_classes: list[str],
    supported_extensions: set[str],
    recursive: bool = True,
    class_aliases: dict[str, str] | None = None,
) -> DiscoveryResult:
    """Discover supported images without modifying the dataset."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        return DiscoveryResult([], list(expected_classes), [], [], [], False)

    directories = [item for item in root.iterdir() if item.is_dir() and not item.name.startswith(".")]
    expected_by_key = {name.casefold(): name for name in expected_classes}
    aliases = {
        alias.casefold(): canonical.casefold()
        for alias, canonical in (class_aliases or {}).items()
    }
    matched: dict[str, Path] = {}
    unexpected = []
    for directory in sorted(directories, key=lambda item: item.name.casefold()):
        key = aliases.get(directory.name.casefold(), directory.name.casefold())
        if key in expected_by_key:
            matched[key] = directory
        else:
            unexpected.append(directory.name)

    candidates: list[ImageCandidate] = []
    unsupported: list[str] = []
    empty: list[str] = []
    for expected in expected_classes:
        directory = matched.get(expected.casefold())
        if directory is None:
            continue
        iterator = directory.rglob("*") if recursive else directory.iterdir()
        files = sorted((item for item in iterator if item.is_file() and not item.name.startswith(".")), key=lambda item: str(item).casefold())
        if not files:
            empty.append(expected)
        for path in files:
            if path.suffix.casefold() not in supported_extensions:
                unsupported.append(path.relative_to(root).as_posix())
                continue
            candidates.append(ImageCandidate(path, expected, path.relative_to(root).as_posix()))

    return DiscoveryResult(
        candidates=sorted(candidates, key=lambda item: item.relative_path.casefold()),
        missing_classes=[name for name in expected_classes if name.casefold() not in matched],
        unexpected_classes=sorted(unexpected, key=str.casefold),
        empty_classes=sorted(set(empty), key=str.casefold),
        unsupported_files=sorted(unsupported, key=str.casefold),
        dataset_root_exists=True,
    )