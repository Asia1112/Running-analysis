MODEL_PATH = r"models\pose_landmarker_full.task"
DEFAULT_SOURCE = r"run7.mp4"
DEFAULT_CSV_PATH = "run_landmarks.csv"

VISIBILITY_THRESHOLD = 0.60
ANALYSIS_VISIBILITY_THRESHOLD = 0.65
ROM_JUMP_THRESHOLD_DEG = 30.0
ROM_VISIBILITY_THRESHOLD = 0.60
ROM_DROP_ABS_THRESHOLD_DEG = 15.0
ROM_DROP_REL_THRESHOLD = 0.20
ROM_REL_DROP_MIN_RAW_DEG = 30.0

LANDMARKS_TO_SAVE = {
    "L_SHOULDER": 11,
    "R_SHOULDER": 12,
    "L_HIP": 23,
    "R_HIP": 24,
    "L_KNEE": 25,
    "R_KNEE": 26,
    "L_ANKLE": 27,
    "R_ANKLE": 28,
    "L_FOOT": 31,
    "R_FOOT": 32,
}

CORE_LANDMARKS = [
    "L_SHOULDER",
    "R_SHOULDER",
    "L_HIP",
    "R_HIP",
    "L_KNEE",
    "R_KNEE",
    "L_ANKLE",
    "R_ANKLE",
]

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (25, 26),
]
