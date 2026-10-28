"""Append-only session annotations with duplicate protection and atomic CSV writes."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from .provenance import PROVENANCE_KEYS, RANGE_KEYS, recording_metadata
from .experiment_metadata import EXPERIMENT_KEYS, experiment_values
from .reference_quality import normalize_reference_quality


ANNOTATION_KEYS = ["video_id", "person_id", "frame_number", "side", "annotator_id", "annotation_session"]
TEXT_KEYS = [key for key in ANNOTATION_KEYS if key != "frame_number"]


def annotation_identity(frame: object, annotator_id: str, session: str) -> dict[str, object]:
    """Return the complete, normalized identity of one independent measurement."""
    identity = {key: frame[key] for key in ("video_id", "person_id", "frame_number", "side")}
    identity.update(annotator_id=annotator_id, annotation_session=session)
    for key in TEXT_KEYS:
        if pd.isna(identity[key]) or not str(identity[key]).strip():
            raise ValueError(f"Pole {key} nie może być puste.")
        identity[key] = str(identity[key]).strip()
    number = float(identity["frame_number"])
    if not np.isfinite(number) or number < 0 or number != int(number):
        raise ValueError("Numer klatki musi być nieujemną liczbą całkowitą.")
    identity["frame_number"] = int(number)
    if identity["side"] not in ("left", "right"):
        raise ValueError("Strona musi mieć wartość left albo right.")
    return identity


def annotation_widget_key(identity: dict[str, object], image_sha256: str) -> str:
    """Keep drafts separate by all identity fields and the exact displayed image."""
    payload = [identity[key] for key in ANNOTATION_KEYS] + [identity.get(key) for key in (*PROVENANCE_KEYS, *RANGE_KEYS)] + [image_sha256]
    return "annotation_" + hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def read_reference_annotations(source: object) -> pd.DataFrame:
    """Preserve textual IDs such as session '001' when reading CSV."""
    return pd.read_csv(source, dtype={key: str for key in [*TEXT_KEYS, "analysis_id", "source_sha256"]}, keep_default_na=False)


def annotation_exists(data: pd.DataFrame, identity: dict[str, object]) -> bool:
    if data.empty:
        return False
    missing = set(ANNOTATION_KEYS).difference(data.columns)
    if missing:
        raise ValueError(f"CSV anotacji nie zawiera kluczy: {', '.join(sorted(missing))}")
    mask = pd.Series(True, index=data.index)
    for key in ANNOTATION_KEYS:
        if key == "frame_number":
            mask &= pd.to_numeric(data[key], errors="raise").eq(identity[key])
        else:
            mask &= data[key].astype(str).str.strip().eq(identity[key])
    return bool(mask.any())


def select_reference_session(data: pd.DataFrame, annotator_id: str | None = None,
                             session: str | None = None) -> pd.DataFrame:
    """Require exactly one annotator/session for a primary comparison."""
    if not {"annotator_id", "annotation_session"}.issubset(data.columns):
        raise ValueError("Referencja musi zawierać anotatora i sesję pomiarową.")
    selected = data.copy()
    for key in ("annotator_id", "annotation_session"):
        if selected[key].isna().any():
            raise ValueError(f"Referencja: pole {key} nie może być puste.")
        selected[key] = selected[key].astype(str).str.strip()
        if selected[key].eq("").any():
            raise ValueError(f"Referencja: pole {key} nie może być puste.")
    if annotator_id is not None:
        selected = selected.loc[selected.annotator_id.eq(annotator_id.strip())]
    if session is not None:
        selected = selected.loc[selected.annotation_session.eq(session.strip())]
    if selected.empty:
        raise ValueError("Brak anotacji dla wybranego anotatora i sesji.")
    if len(selected[["annotator_id", "annotation_session"]].drop_duplicates()) != 1:
        raise ValueError("Wybierz jednego anotatora i jedną sesję referencyjną; nie łącz powtórzeń jako niezależnych klatek.")
    return selected


def save_reference_annotation(record: dict[str, object], path: str | Path) -> Path:
    """Append a measurement; never replace an earlier session or same-key record."""
    identity = annotation_identity(record, record["annotator_id"], record["annotation_session"])
    record = normalize_reference_quality(pd.DataFrame([record])).iloc[0].to_dict()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(output.name + ".lock")
    try:
        handle = lock.open("x")
    except FileExistsError as error:
        raise ValueError("Inny zapis anotacji jest w toku. Spróbuj ponownie.") from error
    temporary = None
    try:
        handle.close()
        existing = read_reference_annotations(output) if output.exists() else pd.DataFrame()
        if not existing.empty:
            existing = normalize_reference_quality(existing)
        if not existing.empty and (set(EXPERIMENT_KEYS).intersection(existing.columns) or
                                   set(EXPERIMENT_KEYS).intersection(record)):
            if experiment_values(existing) != experiment_values(pd.DataFrame([record])):
                raise ValueError("Plik anotacji ma inne metadane eksperymentu.")
        if not existing.empty and (set(PROVENANCE_KEYS).intersection(existing.columns) or
                                   set(PROVENANCE_KEYS).intersection(record)):
            if recording_metadata(existing) != recording_metadata(pd.DataFrame([record])):
                raise ValueError("Plik anotacji dotyczy innej analizy lub geometrii obrazu.")
        if annotation_exists(existing, identity):
            raise ValueError("Ta klatka jest już zapisana w tej sesji. Utwórz nową sesję, aby wykonać kolejny pomiar.")
        if not existing.empty and existing.duplicated(ANNOTATION_KEYS).any():
            raise ValueError("CSV zawiera powielone klucze anotacji. Zachowano plik bez zmian.")
        updated = pd.concat([existing, pd.DataFrame([{**record, **identity}])], ignore_index=True)
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", suffix=".tmp",
                                dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            updated.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
    return output
