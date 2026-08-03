import os
import gc
import sys
import torch
import warnings

from config import INPUT_DIR

from stages.stage_00_metadata import run_metadata_stage
from stages.stage_01_transcribe import run_transcription_stage
from stages.stage_02_align import run_alignment_stage
from stages.stage_03_diarize import run_diarization_stage
from stages.stage_04_merge import run_merge_stage
from stages.stage_05_export import run_export_stage

warnings.filterwarnings("ignore", message="triton not found")

print("==============================")
print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip)
print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)

print("==============================")


def main():

    audio_files = [
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".wav")
    ]

    if not audio_files:
        print("No WAV files found")
        return

    for audio_file in audio_files:

        print("\n==============================")
        print(f"Processing: {audio_file}")
        print("==============================\n")

        try:

            run_metadata_stage(audio_file)
            run_transcription_stage(audio_file)
            run_alignment_stage(audio_file)
            run_diarization_stage(audio_file)
            run_merge_stage(audio_file)
            run_export_stage(audio_file)

        except Exception as e:

            print(f"\nFAILED: {audio_file}")
            print(e)

            continue

    print("\nALL COMPLETE")

    gc.collect()
    sys.exit(0)


if __name__ == "__main__":
    main()
