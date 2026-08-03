import os
import json

import whisperx

from config import MODEL_NAME, DEVICE, COMPUTE_TYPE, BATCH_SIZE, LANGUAGE

from utils.files import stage_file

_model = None


def load_model():

    global _model

    if _model is None:

        print("Loading Whisper model")

        _model = whisperx.load_model(
            MODEL_NAME,
            DEVICE,
            compute_type=COMPUTE_TYPE,
            language=LANGUAGE,
            batch_size=BATCH_SIZE,
        )

    return _model


def run_transcription_stage(audio_file):

    output = stage_file(audio_file, "01_transcription.json")

    if os.path.exists(output):

        print("Skipping transcription")

        return

    model = load_model()

    audio = whisperx.load_audio(audio_file)

    result = model.transcribe(audio, batch_size=BATCH_SIZE, language=LANGUAGE)

    with open(output, "w", encoding="utf8") as f:

        json.dump(result, f, indent=4, ensure_ascii=False)

    print("Transcription complete")
