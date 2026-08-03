import os

# ----------------------------
# Directories
# ----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "RAW Audio")
WORK_DIR = os.path.join(BASE_DIR, "working")
OUTPUT_DIR = os.path.join(BASE_DIR, "Transcripts")


# ----------------------------
# Whisper model (PyTorch ROCm)
# ----------------------------

# ROCm uses the CUDA device name in PyTorch
DEVICE = "cuda"

LANGUAGE = "en"
MODEL_NAME = "large-v3"

# Best accuracy model
if DEVICE == "cuda":
    COMPUTE_TYPE = "float16"
else:
    COMPUTE_TYPE = "int8"

BATCH_SIZE = 32

# ----------------------------
# Whisper accuracy settings
# ----------------------------

# RX 7900 XTX supports FP16 well
FP16 = True

# Beam search improves accuracy at the cost of speed
NUM_BEAMS = 5

# Multiple candidates for best result
BEST_OF = 5

TEMPERATURE = 0.0

# Helps maintain context between segments
CONDITION_ON_PREVIOUS_TEXT = True


# ----------------------------
# Diarization
# ----------------------------

DIARIZATION_TOKEN = os.getenv("DIARIZATION_TOKEN")

MIN_SPEAKERS = 1
MAX_SPEAKERS = 3


# ----------------------------
# Processing behaviour
# ----------------------------

SKIP_EXISTING = True

CACHE_MODELS = True


# ----------------------------
# Export
# ----------------------------

EXPORT_TXT = True
EXPORT_SRT = True
EXPORT_JSON = True
