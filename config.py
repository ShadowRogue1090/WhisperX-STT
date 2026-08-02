import os

# ----------------------------
# Directories
# ----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "RAW Audio")

WORK_DIR = os.path.join(BASE_DIR, "working")

OUTPUT_DIR = os.path.join(BASE_DIR, "Transcripts")


# ----------------------------
# Whisper model
# ----------------------------

MODEL_NAME = "large-v3"

DEVICE = "cpu"

COMPUTE_TYPE = "float32"

BATCH_SIZE = 16

LANGUAGE = "en"


# ----------------------------
# Diarization
# ----------------------------

# HuggingFace token
# Create one at:
# https://huggingface.co/settings/tokens

DIARIZATION_TOKEN = os.getenv("DIARIZATION_TOKEN")


MIN_SPEAKERS = 2
MAX_SPEAKERS = 2


# ----------------------------
# Processing behaviour
# ----------------------------

SKIP_EXISTING = True


# Keep models alive between files
# Recommended for batch processing

CACHE_MODELS = True


# ----------------------------
# Export
# ----------------------------

EXPORT_TXT = True

EXPORT_SRT = True

EXPORT_JSON = True
