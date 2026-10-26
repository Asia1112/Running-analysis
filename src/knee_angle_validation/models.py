"""Frame-quality result shared by extraction and quality assessment."""

from dataclasses import dataclass, field


@dataclass
class FrameQuality:
    """Quality classification and its auditable reasons for one frame."""

    status: str
    reasons: list[str] = field(default_factory=list)
