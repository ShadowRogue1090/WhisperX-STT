import os
import json
import torch
import whisperx

from config import DEVICE

from utils.files import stage_file


def run_alignment_stage(audio_file):

    output = stage_file(audio_file, "02_alignment.json")

    if os.path.exists(output):

        print("Skipping alignment")
        return

    transcription_file = stage_file(audio_file, "01_transcription.json")

    with open(transcription_file, encoding="utf8") as f:
        result = json.load(f)

    if not result.get("segments"):
        raise RuntimeError("No transcription segments available for alignment")

    audio = whisperx.load_audio(audio_file)

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=DEVICE,
    )

    aligned = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        DEVICE,
    )

    with open(output, "w", encoding="utf8") as f:
        json.dump(
            aligned,
            f,
            indent=4,
            ensure_ascii=False,
        )

    del model_a
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print("Alignment complete")
