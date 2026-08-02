import os
import gc
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import whisperx
from whisperx.diarize import DiarizationPipeline

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "cpu"
COMPUTE_TYPE = "float32"
MODEL_NAME = "large-v3"

AUDIO_FILE = r"C:\Users\Owenp\Downloads\Annabelle.wav"

# Set to None to let WhisperX determine speaker count
KNOWN_SPEAKERS = 3
# Example:
# KNOWN_SPEAKERS = 2

BATCH_SIZE = 16

# HF_TOKEN = os.getenv("HF_TOKEN")
HF_TOKEN = os.getenv("DIARIZATION-TOKEN")

if not HF_TOKEN:
    raise RuntimeError(
        "DIARIZATION-TOKEN environment variable not found.\n"
        "Create one with your Hugging Face Read token."
    )

# ============================================================
# OUTPUT SETUP
# ============================================================

start_time = time.time()

audio_path = Path(AUDIO_FILE)
output_dir = audio_path.parent / "Transcripts"
output_dir.mkdir(exist_ok=True)

json_output = output_dir / f"{audio_path.stem}.json"
markdown_output = output_dir / f"{audio_path.stem}.md"

print("=" * 60)
print(" WhisperX Interview Transcriber")
print("=" * 60)
print(f"Audio File : {audio_path}")
print(f"Model      : {MODEL_NAME}")
print(f"Device     : {DEVICE}")
print(f"Output Dir : {output_dir}")
print("=" * 60)

# ============================================================
# STEP 1 - LOAD MODEL
# ============================================================

print("\n[1/5] Loading Whisper model...")

model = whisperx.load_model(
    MODEL_NAME,
    DEVICE,
    compute_type=COMPUTE_TYPE,
)

print("Model loaded.")

# ============================================================
# STEP 2 - LOAD AUDIO
# ============================================================

print("\n[2/5] Loading audio...")

audio = whisperx.load_audio(str(audio_path))

print("Audio loaded.")

print("\nBeginning transcription...")
transcribe_start = time.time()

result = model.transcribe(
    audio,
    batch_size=BATCH_SIZE,
    language="en",
)

print(f"Transcription complete " f"({time.time()-transcribe_start:.1f} seconds)")

print(f"Detected language: {result['language']}")
print(f"Segments: {len(result['segments'])}")

# ============================================================
# FREE MEMORY
# ============================================================

print("\nReleasing Whisper model...")

del model
gc.collect()

# ============================================================
# STEP 3 - ALIGNMENT
# ============================================================

print("\n[3/5] Loading alignment model...")

model_a, metadata = whisperx.load_align_model(
    language_code=result["language"],
    device=DEVICE,
)

print("Aligning transcript...")

align_start = time.time()

result = whisperx.align(
    result["segments"],
    model_a,
    metadata,
    audio,
    DEVICE,
    return_char_alignments=False,
)

print(f"Alignment complete " f"({time.time()-align_start:.1f} seconds)")

del model_a
gc.collect()

# ============================================================
# STEP 4 - DIARIZATION
# ============================================================

print("\n[4/5] Running speaker diarization...")

diarize_model = DiarizationPipeline(
    token=HF_TOKEN,
    device=DEVICE,
)

if KNOWN_SPEAKERS is None:

    diarize_segments = diarize_model(audio)

else:

    diarize_segments = diarize_model(
        audio,
        min_speakers=KNOWN_SPEAKERS,
        max_speakers=KNOWN_SPEAKERS,
    )

print("Assigning speakers...")

result = whisperx.assign_word_speakers(
    diarize_segments,
    result,
)

print("Speaker assignment complete.")

# ============================================================
# STEP 5 - SAVE JSON
# ============================================================

print("\n[5/5] Saving outputs...")

metadata = {
    "model": MODEL_NAME,
    "device": DEVICE,
    "compute_type": COMPUTE_TYPE,
    "source_audio": str(audio_path),
    "generated": datetime.now().isoformat(),
    "language": result["language"],
}

output = {
    "metadata": metadata,
    "transcript": result,
}

with open(json_output, "w", encoding="utf-8") as f:
    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False,
    )

print(f"JSON saved -> {json_output}")

# ============================================================
# SAVE MARKDOWN
# ============================================================

with open(markdown_output, "w", encoding="utf-8") as f:

    f.write(f"# {audio_path.stem}\n\n")

    f.write(f"Generated: {metadata['generated']}\n\n")
    f.write(f"Model: {MODEL_NAME}\n\n")
    f.write("---\n\n")

    for segment in result["segments"]:

        speaker = segment.get("speaker", "Unknown")

        start = segment["start"]

        hours = int(start // 3600)
        minutes = int((start % 3600) // 60)
        seconds = start % 60

        timestamp = f"{hours:02}:{minutes:02}:{seconds:05.2f}"

        text = segment["text"].strip()

        f.write(f"## [{timestamp}] {speaker}\n\n")
        f.write(text)
        f.write("\n\n")

print(f"Markdown saved -> {markdown_output}")

# ============================================================
# FINISHED
# ============================================================

elapsed = time.time() - start_time

hours = int(elapsed // 3600)
minutes = int((elapsed % 3600) // 60)
seconds = int(elapsed % 60)

print("\n" + "=" * 60)
print("Processing Complete")
print("=" * 60)
print(f"Total runtime : {hours:02}:{minutes:02}:{seconds:02}")
print(f"JSON          : {json_output}")
print(f"Markdown      : {markdown_output}")
print("=" * 60)
