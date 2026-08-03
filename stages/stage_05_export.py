import json

from utils.files import stage_file


def run_export_stage(audio_file):

    with open(stage_file(audio_file, "04_final.json"), encoding="utf8") as f:

        result = json.load(f)

    output = stage_file(audio_file, "transcript.txt")

    with open(output, "w", encoding="utf8") as f:

        for segment in result["segments"]:

            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "").strip()

            start = segment.get("start")
            end = segment.get("end")

            if start is not None and end is not None:

                timestamp = f"[{start:08.2f} --> {end:08.2f}]"

            else:

                timestamp = "[NO TIMESTAMP]"

            f.write(f"{timestamp} {speaker}: {text}\n\n")

    print("Export complete")
