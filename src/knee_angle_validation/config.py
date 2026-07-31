"""Central configuration for knee-angle validation workflows."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ValidationConfig:
    """Configurable research thresholds that can be recalibrated later."""

    visibility_bins: tuple[float, float, float] = (0.60, 0.80, 0.90)
    jump_thresholds_deg: tuple[float, float, float] = (10.0, 20.0, 30.0)
    smoothing_window: int = 5
    savgol_window: int = 7
    savgol_order: int = 2
    residual_review_deg: float = 10.0
    residual_unreliable_deg: float = 20.0
    point_acceleration_review: float = 0.04
    point_acceleration_unreliable: float = 0.08
    review_visibility: float = 0.80
    unreliable_visibility: float = 0.60
    required_columns: tuple[str, ...] = field(default=(
        "video_id", "person_id", "frame_number", "time_s", "fps",
        "image_width", "image_height", "side", "knee_angle_raw", "pose_found",
    ))


DEFAULT_CONFIG = ValidationConfig()
