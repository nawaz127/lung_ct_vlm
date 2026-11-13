"""Image decoding, metadata, and exact SHA-256 integrity checks."""

from hashlib import sha256
from pathlib import Path

from PIL import Image, UnidentifiedImageError


VALID_MODES = {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "YCbCr", "I", "F"}


def inspect_image(path: Path, root: Path, class_name: str) -> dict:
    """Return a complete metadata record, including failures."""
    relative_path = path.resolve().relative_to(root.resolve()).as_posix()
    record = {
        "relative_path": relative_path,
        "absolute_path": str(path.resolve()),
        "class_name": class_name,
        "extension": path.suffix.casefold(),
        "format": "",
        "width": "",
        "height": "",
        "mode": "",
        "channels": "",
        "file_size_bytes": path.stat().st_size if path.exists() else 0,
        "file_size_kb": round(path.stat().st_size / 1024, 3) if path.exists() else 0,
        "sha256": "",
        "validation_status": "invalid",
        "validation_error": "",
    }
    if not path.exists():
        record["validation_error"] = "file_missing"
        return record
    try:
        digest = sha256(path.read_bytes()).hexdigest()
        record["sha256"] = digest
        if record["file_size_bytes"] == 0:
            raise ValueError("zero_byte_file")
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            record.update(
                format=image.format or "",
                width=width,
                height=height,
                mode=image.mode,
                channels=len(image.getbands()),
            )
        if width <= 0 or height <= 0:
            raise ValueError("non_positive_dimensions")
        if record["mode"] not in VALID_MODES:
            raise ValueError("unsupported_image_mode")
        record["validation_status"] = "valid"
    except (OSError, ValueError, UnidentifiedImageError) as error:
        record["validation_error"] = str(error) or error.__class__.__name__
    return record