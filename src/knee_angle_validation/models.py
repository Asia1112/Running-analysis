"""Typed models shared by extraction, quality, and annotation modules."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Point2D:
    """A two-dimensional point in normalized or pixel coordinates."""

    x: float
    y: float


@dataclass
class FrameQuality:
    """Quality classification and its auditable reasons for one frame."""

    status: str
    reasons: list[str] = field(default_factory=list)
