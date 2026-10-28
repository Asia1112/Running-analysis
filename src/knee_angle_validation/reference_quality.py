"""Manual reference certainty, independent of MediaPipe detection confidence."""

import numpy as np
import pandas as pd


POINTS = ("hip", "knee", "ankle")
QUALITY_OPTIONS = ("CLEAR", "UNCERTAIN", "UNMARKABLE")
QUALITY_COLUMNS = tuple(f"{point}_quality" for point in POINTS)


def frame_quality(qualities: list[str]) -> str:
    if len(qualities) != 3:
        raise ValueError("Wymagane są trzy flagi jakości punktów.")
    if any(value not in (*QUALITY_OPTIONS, "UNASSESSED") for value in qualities):
        raise ValueError("Nieprawidłowa flaga jakości referencji.")
    if all(value == "UNASSESSED" for value in qualities):
        return "UNASSESSED"
    if "UNASSESSED" in qualities:
        raise ValueError("Oceń wszystkie trzy punkty albo pozostaw cały starszy pomiar jako UNASSESSED.")
    return "UNMARKABLE" if "UNMARKABLE" in qualities else "UNCERTAIN" if "UNCERTAIN" in qualities else "CLEAR"


def normalize_reference_quality(data: pd.DataFrame) -> pd.DataFrame:
    """Preserve legacy uncertainty and mask points explicitly marked unmarkable."""
    result = data.copy()
    present = set(QUALITY_COLUMNS).intersection(result.columns)
    if present and len(present) != 3:
        raise ValueError("Referencja wymaga flag jakości biodra, kolana i kostki.")
    if not present:
        for key in QUALITY_COLUMNS:
            result[key] = "UNASSESSED"
    else:
        flags = result[list(QUALITY_COLUMNS)].fillna("").astype(str).apply(lambda col: col.str.strip().str.upper())
        legacy = flags.eq("").all(axis=1)
        flags.loc[legacy, :] = "UNASSESSED"
        result[list(QUALITY_COLUMNS)] = flags
    derived = pd.Series([frame_quality(list(values)) for values in result[list(QUALITY_COLUMNS)].itertuples(index=False, name=None)],
                        index=result.index, dtype=str)
    if "frame_quality" in result:
        saved = result.frame_quality.fillna("").astype(str).str.strip().str.upper()
        if (saved.ne("") & saved.ne(derived)).any():
            raise ValueError("Ocena klatki nie zgadza się z flagami punktów.")
    result["frame_quality"] = derived
    for point, quality_column in zip(POINTS, QUALITY_COLUMNS):
        for axis in ("x", "y"):
            column = f"{point}_{axis}_ref"
            if column not in result:
                raise ValueError(f"Brak współrzędnej referencji: {column}.")
            values = pd.to_numeric(result[column], errors="coerce")
            required = result[quality_column].isin(["CLEAR", "UNCERTAIN"])
            if (required & ~np.isfinite(values)).any():
                raise ValueError(f"Punkt {point} oznaczony jako CLEAR/UNCERTAIN wymaga poprawnych współrzędnych.")
            result[column] = values.mask(result[quality_column].eq("UNMARKABLE"))
    for column in ("knee_angle_reference", "knee_internal_angle_deg", "knee_flexion_deg"):
        if column in result:
            result[column] = result[column].mask(result.frame_quality.eq("UNMARKABLE"))
    return result
