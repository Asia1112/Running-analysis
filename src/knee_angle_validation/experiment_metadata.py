"""Map the study's semicolon-separated recording table to analysis records."""

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METADATA_PATH = Path("data/metadata/videos_metadata.csv")
EXPERIMENT_KEYS = ("original_filename", "speed_kmh", "inclusion_status", "metadata_notes", "metadata_table_sha256")


def normalize_experiment_metadata(metadata: dict, person_id: str, side: str) -> dict:
    """Validate one recording's study metadata without guessing missing speed."""
    if not str(person_id).strip() or side not in ("left", "right"):
        raise ValueError("Podaj identyfikator osoby i stronę left/right.")
    if metadata.get("person_id", person_id) != person_id or metadata.get("side", side) != side:
        raise ValueError("Osoba lub strona nie zgadza się z tabelą metadanych.")
    filename = str(metadata.get("original_filename", "")).strip()
    if not filename:
        raise ValueError("Metadane wymagają oryginalnej nazwy nagrania.")
    speed = metadata.get("speed_kmh")
    if speed is None or pd.isna(speed) or str(speed).strip() == "":
        speed = None
    else:
        try:
            speed = float(str(speed).replace(",", "."))
        except (ValueError, TypeError) as error:
            raise ValueError("Prędkość musi być dodatnią liczbą w km/h.") from error
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("Prędkość musi być dodatnią liczbą w km/h.")
    status = str(metadata.get("inclusion_status", "INCLUDED")).strip().upper()
    if status not in ("INCLUDED", "EXCLUDED", "PENDING"):
        raise ValueError("Status nagrania musi mieć wartość INCLUDED, EXCLUDED albo PENDING.")
    return {"original_filename": filename, "speed_kmh": speed, "inclusion_status": status,
            "metadata_notes": str(metadata.get("metadata_notes", "")),
            "metadata_table_sha256": str(metadata.get("metadata_table_sha256", ""))}


def load_video_metadata(path: str | Path = DEFAULT_METADATA_PATH) -> pd.DataFrame:
    """Read and validate the actual study schema; preserve participant IDs."""
    content = Path(path).read_bytes()
    table = pd.read_csv(io.BytesIO(content), sep=";", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    required = {"original_filename", "participant_id", "speed_kmh", "analyzed_side", "inclusion_status", "notes"}
    if not required.issubset(table.columns):
        raise ValueError("Tabela metadanych wymaga kolumn: " + ", ".join(sorted(required)) + " (separator ;).")
    rows = []
    for _, row in table.iterrows():
        person = row.participant_id.strip()
        side = row.analyzed_side.strip().lower()
        values = normalize_experiment_metadata({
            "original_filename": row.original_filename, "speed_kmh": row.speed_kmh,
            "inclusion_status": row.inclusion_status, "metadata_notes": row.notes,
            "metadata_table_sha256": hashlib.sha256(content).hexdigest(),
        }, person, side)
        if values["speed_kmh"] is None:
            raise ValueError("W tabeli metadanych brakuje prędkości nagrania.")
        rows.append({**values, "person_id": person, "side": side})
    normalized = pd.DataFrame(rows, columns=[*EXPERIMENT_KEYS, "person_id", "side"])
    if normalized.original_filename.str.casefold().duplicated().any():
        raise ValueError("Tabela metadanych zawiera powielone nazwy nagrań.")
    return normalized


def metadata_for_video(filename: str, table: pd.DataFrame) -> dict | None:
    matches = table.loc[table.original_filename.str.casefold().eq(Path(filename).name.casefold())]
    return matches.iloc[0].to_dict() if not matches.empty else None


def experiment_values(data: pd.DataFrame) -> dict | None:
    """Read one complete snapshot; older CSV without study metadata stays legacy."""
    if not set(EXPERIMENT_KEYS).intersection(data.columns):
        return None
    if data.empty or not {*EXPERIMENT_KEYS, "person_id", "side"}.issubset(data.columns):
        raise ValueError("Niekompletne metadane eksperymentu w CSV.")
    values = {}
    for key in EXPERIMENT_KEYS:
        column = data[key].fillna("")
        if column.nunique() != 1:
            raise ValueError(f"Analiza wymaga jednej wartości {key}.")
        values[key] = column.iloc[0]
    return normalize_experiment_metadata(values, str(data.person_id.iloc[0]), str(data.side.iloc[0]))
