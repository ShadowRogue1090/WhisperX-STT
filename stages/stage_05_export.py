import json

from utils.files import stage_file


def run_export_stage(audio_file):

    with open(stage_file(audio_file, "04_final.json")) as f:

        result = json.load(f)

    output = stage_file(audio_file, "transcript.txt")

    with open(output, "w", encoding="utf8") as f:

        for segment in result["segments"]:

            speaker = segment.get("speaker", "UNKNOWN")

            f.write(f"{speaker}: {segment['text']}\n\n")

    print("Export complete")
