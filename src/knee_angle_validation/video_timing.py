"""Read and validate decoded presentation timestamps for CFR and VFR videos."""

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import math
import os
from pathlib import Path
import re
import subprocess

import imageio_ffmpeg
import numpy as np


TIME_SOURCE = "ffmpeg_pts_v1"


@dataclass(frozen=True)
class VideoTiming:
    pts_s: tuple[float, ...]
    time_base_s: float
    fps: float
    tolerance_s: float
    max_time_error_s: float
    max_interval_error_s: float
    is_cfr: bool = True
    last_duration_s: float | None = None

    @property
    def relative_times(self) -> np.ndarray:
        return np.round(np.asarray(self.pts_s) - self.pts_s[0], 12)

    @property
    def end_time_s(self) -> float:
        return float(self.relative_times[-1] + (self.last_duration_s or 1.0 / self.fps))

    def frame_range(self, start: float, end: float | None) -> tuple[int, int]:
        if end is not None and end > self.end_time_s + 1e-9:
            raise ValueError("Wybrany zakres przekracza koniec osi czasu PTS nagrania.")
        times = self.relative_times
        return int(np.searchsorted(times, start)), int(np.searchsorted(times, end)) if end is not None else len(times)

    def report(self) -> dict:
        return {"time_source": TIME_SOURCE, "verified_frame_count": len(self.pts_s),
                "pts_first_s": self.pts_s[0], "pts_time_base_s": self.time_base_s,
                "timing_tolerance_s": self.tolerance_s, "max_time_error_s": self.max_time_error_s,
                "max_interval_error_s": self.max_interval_error_s, "nominal_fps": self.fps,
                "frame_rate_mode": "CFR" if self.is_cfr else "VFR_or_nominal_mismatch",
                "end_duration_source": "decoded_frame_duration" if self.last_duration_s else "nominal_period_estimate"}


def inspect_timing(pts_s: list[float], time_base_s: float, fps: float,
                   last_duration_s: float | None = None) -> VideoTiming:
    """Validate real PTS and report whether nominal FPS also describes them."""
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(time_base_s) or time_base_s <= 0:
        raise ValueError("Nie można potwierdzić FPS ani podstawy czasu nagrania.")
    pts = np.asarray(pts_s, dtype=float)
    if len(pts) < 2 or not np.isfinite(pts).all() or (np.diff(pts) <= 0).any():
        raise ValueError("Czas filmu wymaga co najmniej dwóch klatek z rosnącymi, skończonymi PTS.")
    # Allow timestamp quantization, but never a tolerance above 1 ms or 5% of a frame.
    tolerance = min(0.001, 0.05 / fps, time_base_s * 1.01 + 1e-9)
    drift = float(np.max(np.abs((pts - pts[0]) - np.arange(len(pts)) / fps)))
    interval_error = float(np.max(np.abs(np.diff(pts) - 1.0 / fps)))
    constant = drift <= tolerance + 1e-9 and interval_error <= tolerance + 1e-9
    return VideoTiming(tuple(float(value) for value in pts), time_base_s, fps, tolerance, drift, interval_error,
                       constant, last_duration_s)


def verify_cfr(pts_s: list[float], time_base_s: float, fps: float) -> VideoTiming:
    result = inspect_timing(pts_s, time_base_s, fps)
    if not result.is_cfr:
        raise ValueError("Nie potwierdzono stałego FPS: zmienne odstępy PTS lub niezgodność z nominalnym FPS. "
                         f"Maks. odchylenie czasu: {result.max_time_error_s * 1000:.3f} ms.")
    return result


def audit_video_timing(source: str | Path, fps: float) -> VideoTiming:
    """Read every decoded frame's integer PTS via bundled FFmpeg, without conversion."""
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-nostdin", "-loglevel", "info",
               "-xerror", "-copyts", "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
               "-vf", "showinfo=checksum=0", "-fps_mode", "passthrough", "-f", "null", "-"]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    pts = []
    time_base = None
    last_duration = None
    tail = deque(maxlen=6)
    try:
        with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              text=True, encoding="utf-8", errors="replace", creationflags=flags) as process:
            try:
                for line in process.stderr:
                    tail.append(line.strip())
                    if "showinfo" not in line:
                        continue
                    config = re.search(r"config in time_base:\s*(\d+/\d+)", line)
                    if config:
                        parsed_base = float(Fraction(config.group(1)))
                        if time_base is not None and parsed_base != time_base:
                            raise ValueError("Podstawa czasu zmienia się w obrębie filmu.")
                        time_base = parsed_base
                    frame = re.search(r"\bn:\s*(\d+)\s+pts:\s*(\S+)", line)
                    if frame:
                        if time_base is None or int(frame.group(1)) != len(pts):
                            raise ValueError("Brak spójnego przypisania PTS do numerów klatek.")
                        pts.append(int(frame.group(2)) * time_base)
                        duration = re.search(r"\bduration:\s*(\d+)\b", line)
                        last_duration = int(duration.group(1)) * time_base if duration and int(duration.group(1)) > 0 else None
                if process.wait() != 0:
                    raise ValueError("FFmpeg nie odczytał poprawnie czasu filmu: " + " | ".join(tail))
            except BaseException:
                if process.poll() is None:
                    process.kill()
                process.wait()
                raise
    except OSError as error:
        raise ValueError("Nie można uruchomić FFmpeg do weryfikacji czasu filmu.") from error
    return inspect_timing(pts, time_base if time_base is not None else float("nan"), fps, last_duration)
