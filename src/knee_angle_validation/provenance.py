"""Recording identity and image geometry shared by extraction and annotation."""

import hashlib
import re
from pathlib import Path
from uuid import uuid4

import pandas as pd


PROVENANCE_KEYS = ("analysis_id", "source_sha256", "crop_mode", "resize_height")


def new_analysis_id() -> str:
    return uuid4().hex


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recording_metadata(data: pd.DataFrame) -> dict | None:
    """Require one complete recording identity; legacy CSV returns None."""
    present = set(PROVENANCE_KEYS).intersection(data.columns)
    if not present:
        return None
    required = (*PROVENANCE_KEYS, "video_id", "person_id", "side")
    if data.empty or not set(required).issubset(data.columns):
        raise ValueError("Niekompletne metadane pochodzenia danych.")
    result = {}
    for key in required:
        if data[key].isna().any() or data[key].nunique() != 1:
            raise ValueError(f"Dane muszą dotyczyć jednej wartości {key}.")
        result[key] = data[key].iloc[0]
    for key, length in (("analysis_id", 32), ("source_sha256", 64)):
        if not re.fullmatch(r"[a-f0-9]{%d}" % length, str(result[key])):
            raise ValueError(f"Nieprawidłowe pole {key}.")
    if result["crop_mode"] not in ("none", "left", "right", "center"):
        raise ValueError("Nieprawidłowe kadrowanie w metadanych.")
    height = float(result["resize_height"])
    if not 0 <= height <= 100000 or not height.is_integer():
        raise ValueError("Nieprawidłowa wysokość przeskalowania w metadanych.")
    result["resize_height"] = int(height)
    for key in ("video_id", "person_id"):
        if not str(result[key]).strip():
            raise ValueError(f"Puste pole {key}.")
        result[key] = str(result[key])
    if result["side"] not in ("left", "right"):
        raise ValueError("Nieprawidłowa strona ciała w metadanych.")
    return result


def require_source_match(data: pd.DataFrame, source_sha256: str | None) -> dict:
    metadata = recording_metadata(data)
    if metadata is None:
        raise ValueError("CSV nie zawiera identyfikacji filmu. Wykonaj ponowną ekstrakcję, aby użyć obrazu i anotacji.")
    if metadata["source_sha256"] != source_sha256:
        raise ValueError("Film nie odpowiada danym CSV (inna suma SHA-256). Wgraj właściwe nagranie.")
    return metadata
