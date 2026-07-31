"""Reference-frame selection and manual annotation data helpers."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .frame_quality import classify_visibility
from .knee_angle import calculate_knee_angle


REFERENCE_COLUMNS = [
    "video_id", "person_id", "frame_number", "side", "annotator_id",
    "annotation_session", "hip_x_ref", "hip_y_ref", "knee_x_ref", "knee_y_ref",
    "ankle_x_ref", "ankle_y_ref", "knee_angle_reference", "notes",
]


def select_reference_frames(data: pd.DataFrame, count: int, random_state: int = 42) -> pd.DataFrame:
    """Select diverse frames stratified by visibility and knee-angle terciles."""
    if count < 1:
        raise ValueError("count must be positive")
    candidates = data.loc[data["pose_found"].astype(bool) & data["knee_angle_raw"].notna()].copy()
    if candidates.empty:
        return candidates
    candidates["visibility_group"] = candidates["min_visibility"].map(classify_visibility)
    ranks = candidates["knee_angle_raw"].rank(method="first")
    candidates["angle_group"] = pd.qcut(ranks, q=min(3, len(candidates)), labels=False, duplicates="drop")
    candidates["stratum"] = candidates["visibility_group"].astype(str) + ":" + candidates["angle_group"].astype(str)
    groups = list(candidates.groupby("stratum", sort=True))
    selected: list[pd.DataFrame] = []
    remaining = count
    rng = np.random.default_rng(random_state)
    while remaining and groups:
        next_groups = []
        for _, group in groups:
            if remaining == 0:
                break
            unused = group.drop(index=pd.concat(selected).index if selected else [])
            if unused.empty:
                continue
            index = rng.choice(unused.index.to_numpy())
            selected.append(unused.loc[[index]])
            remaining -= 1
            if len(unused) > 1:
                next_groups.append(("", unused.drop(index=index)))
        groups = next_groups
    return pd.concat(selected).sort_values("frame_number").head(count).reset_index(drop=True)


def build_reference_annotation(
    frame: pd.Series,
    points: Sequence[tuple[float, float]],
    annotator_id: str,
    annotation_session: str,
    notes: str = "",
) -> dict[str, object]:
    """Build one reference record from ordered hip, knee, and ankle clicks."""
    if len(points) != 3:
        raise ValueError("Exactly three points are required: hip, knee, ankle")
    hip, knee, ankle = points
    return {
        "video_id": frame["video_id"], "person_id": frame["person_id"],
        "frame_number": int(frame["frame_number"]), "side": frame["side"],
        "annotator_id": annotator_id, "annotation_session": annotation_session,
        "hip_x_ref": hip[0], "hip_y_ref": hip[1],
        "knee_x_ref": knee[0], "knee_y_ref": knee[1],
        "ankle_x_ref": ankle[0], "ankle_y_ref": ankle[1],
        "knee_angle_reference": calculate_knee_angle(hip, knee, ankle), "notes": notes,
    }
