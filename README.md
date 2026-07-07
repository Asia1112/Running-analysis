# Running Biomechanics Analysis

Prototype for an engineering thesis system that analyzes running biomechanics from video.

The current pipeline is:

1. Detect pose landmarks from a running video.
2. Export selected anatomical points and knee angles to CSV.
3. Analyze the CSV to estimate cadence and quality/confidence.
4. Plot knee-angle curves for visual inspection.

## Project Structure

```text
pose_run/
  src/running_biomechanics/
    config.py             shared model paths, thresholds, landmark indexes
    pose_extraction.py    MediaPipe video processing and CSV export
    analysis.py           cadence and quality analysis from CSV
    plotting.py           knee-angle plotting
    utils.py              small shared helpers
  models/                 MediaPipe model files
  pose_quality_checks/    generated quality-check images and CSV summaries
  pose_video.py           CLI wrapper for pose extraction
  analyze_run_csv.py      CLI wrapper for CSV analysis
  plot_angles.py          CLI wrapper for plots
```

The root scripts are kept for convenience, but the implementation lives in
`src/running_biomechanics`.

## Commands

Detect pose landmarks:

```powershell
.\.venv\Scripts\python.exe pose_video.py --source path\to\video.mp4 --csv run_landmarks.csv --no-preview
```

For split-screen or screen-recorded videos:

```powershell
.\.venv\Scripts\python.exe pose_video.py --source path\to\video.mp4 --crop left --csv run_landmarks.csv --no-preview
```

Analyze the generated CSV:

```powershell
.\.venv\Scripts\python.exe analyze_run_csv.py --csv run_landmarks.csv
```

Plot knee angles:

```powershell
.\.venv\Scripts\python.exe plot_angles.py --csv run_landmarks.csv
```

## Future App Integration

After a user uploads a video, call `process_video()` from
`src.running_biomechanics.pose_extraction`. Then pass the generated CSV to
`analyze_csv()` from `src.running_biomechanics.analysis`.

The CSV contains `pose_found`, `core_quality`, and `core_visible`, which should
be used to reject weak frames or show a low-confidence report.

Large uploaded videos and generated CSV/image outputs are intentionally ignored
by Git. Keep reusable code, model configuration, and documentation in the
repository; keep test recordings outside version control unless a small sample
is specifically needed.
