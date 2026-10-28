"""Append-only session annotations with duplicate protection and atomic CSV writes."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd


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
    payload = [identity[key] for key in ANNOTATION_KEYS] + [image_sha256]
    return "annotation_" + hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def read_reference_annotations(source: object) -> pd.DataFrame:
    """Preserve textual IDs such as session '001' when reading CSV."""
    return pd.read_csv(source, dtype={key: str for key in TEXT_KEYS}, keep_default_na=False)


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
