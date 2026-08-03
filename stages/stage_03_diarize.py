import os
import json

import whisperx
from whisperx.diarize import DiarizationPipeline

from config import (
    DEVICE,
    DIARIZATION_TOKEN,
    MIN_SPEAKERS,
    MAX_SPEAKERS,
)

from utils.files import stage_file

_pipeline = None


def load_diarization_model():

    global _pipeline

    if _pipeline is None:

        print("Loading diarization model")

        _pipeline = DiarizationPipeline(
            token=DIARIZATION_TOKEN,
            device=DEVICE,
        )

        print("Diarization model loaded")

    return _pipeline


def run_diarization_stage(audio_file):

    output = stage_file(audio_file, "03_diarization.json")

    if os.path.exists(output):

        print("Skipping diarization")
        return

    print("Loading audio")

    audio = whisperx.load_audio(audio_file)

    pipeline = load_diarization_model()

    print("Running diarization")

    diarize_segments = pipeline(
        audio,
        min_speakers=MIN_SPEAKERS,
        max_speakers=MAX_SPEAKERS,
    )

    diarize_segments.to_json(output)

    print("Diarization complete")
