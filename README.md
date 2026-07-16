# Running Biomechanics Analysis

Prototype for an engineering thesis system that analyzes running biomechanics from video.

The current pipeline is:

1. Detect pose landmarks from a running video.
2. Export selected anatomical points and knee angles to CSV.
3. Analyze the CSV to estimate cadence and quality/confidence.
4. Calculate report-ready technique metrics such as knee range of motion,
   trunk lean, symmetry, and vertical oscillation proxy.
5. Plot knee-angle curves for visual inspection.

## Project Structure

```text
pose_run/
  src/running_biomechanics/
    config.py             shared model paths, thresholds, landmark indexes
    pose_extraction.py    MediaPipe video processing and CSV export
    analysis.py           cadence and quality analysis from CSV
    report.py             user-facing report generation
    plotting.py           knee-angle plotting
    utils.py              small shared helpers
  models/                 MediaPipe model files
  pose_quality_checks/    generated quality-check images and CSV summaries
  pose_video.py           CLI wrapper for pose extraction
  analyze_run_csv.py      CLI wrapper for CSV analysis
  generate_report.py      CLI wrapper for report generation
  plot_angles.py          CLI wrapper for plots
```

The root scripts are kept for convenience, but the implementation lives in
`src/running_biomechanics`.

## Commands

Detect pose landmarks:

```powershell
.\.venv\Scripts\python.exe pose_video.py --source path\to\video.mp4 --csv run_landmarks.csv --no-preview
```

Save a browser-compatible H.264 video with the detected skeleton and knee angles:

```powershell
.\.venv\Scripts\python.exe pose_video.py --source path\to\video.mp4 --csv run_landmarks.csv --annotated-video annotated_run.mp4 --no-preview
```

For split-screen or screen-recorded videos:

```powershell
.\.venv\Scripts\python.exe pose_video.py --source path\to\video.mp4 --crop left --csv run_landmarks.csv --no-preview
```

Analyze the generated CSV:

```powershell
.\.venv\Scripts\python.exe analyze_run_csv.py --csv run_landmarks.csv
```

Save the analysis as JSON for a future app/report screen:

```powershell
.\.venv\Scripts\python.exe analyze_run_csv.py --csv run_landmarks.csv --json report.json
```

Generate a user-facing report:

```powershell
.\.venv\Scripts\python.exe generate_report.py --csv run_landmarks.csv --source-name video.mp4 --markdown report.md --json report.json
```

Start the upload app:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

The Streamlit report is organized into tabs for preview, detection quality,
cadence, knees, hips, trunk, technical proxy indicators, suspicious frames,
and exports. Each major metric has a separate reliability assessment. The
export tab provides PDF, landmark CSV, angle CSV, suspicious-frame CSV, and a
ZIP archive of diagnostic frames when they are available.

All reported values are technical indicators estimated from a side-view 2D
recording. They are not a medical diagnosis and should be verified against the
annotated video whenever landmark visibility or temporal stability is limited.

Plot knee angles:

```powershell
.\.venv\Scripts\python.exe plot_angles.py --csv run_landmarks.csv
```

## Knee ROM Validation

The upload app validates raw knee angles against a 5-frame moving average. It:

- plots raw and smoothed left/right knee angles,
- marks frame-to-frame angle jumps above 30 degrees,
- reports knee and ankle visibility for suspicious frames,
- lists the frames containing minimum and maximum knee angles,
- compares raw and smoothed ROM,
- flags a likely tracking error when smoothing reduces ROM substantially.

During pose extraction, frames with an angle jump above 30 degrees or knee/ankle
visibility below 0.60 are saved under `debug_frames/<video-and-timestamp>/` with
the skeleton, highlighted joints, knee angles, and visibility values overlaid.
After analysis, the upload app replaces the original preview with the generated
H.264 video containing the detected skeleton and knee-angle labels.

## Future App Integration

After a user uploads a video, call `process_video()` from
`src.running_biomechanics.pose_extraction`. Then pass the generated CSV to
`analyze_csv()` from `src.running_biomechanics.analysis`, or pass the analysis
result to `build_report()` from `src.running_biomechanics.report`.

The CSV contains `pose_found`, `core_quality`, and `core_visible`, which should
be used to reject weak frames or show a low-confidence report.

The analysis report currently includes cadence, left/right cadence difference,
knee range of motion, trunk lean, vertical oscillation proxy, symmetry score,
confidence, app-facing `report_status`, and plain-language flags. These metrics
are 2D side-view estimates, so they should be described as video-based proxy
metrics rather than laboratory ground-truth biomechanics.

Large uploaded videos and generated CSV/image outputs are intentionally ignored
by Git. Keep reusable code, model configuration, and documentation in the
repository; keep test recordings outside version control unless a small sample
is specifically needed.
