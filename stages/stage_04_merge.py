import os
import json
import pandas as pd

import whisperx

from utils.files import stage_file


def run_merge_stage(audio_file):

    output = stage_file(audio_file, "04_final.json")

    if os.path.exists(output):

        print("Skipping merge")
        return

    # Load aligned Whisper output

    with open(stage_file(audio_file, "02_alignment.json"), encoding="utf8") as f:

        result = json.load(f)

    # Load diarization results

    diarize_segments = pd.read_json(stage_file(audio_file, "03_diarization.json"))

    # Assign speakers to words

    result = whisperx.assign_word_speakers(diarize_segments, result)

    with open(output, "w", encoding="utf8") as f:

        json.dump(result, f, indent=4, ensure_ascii=False)

    print("Merge complete")
