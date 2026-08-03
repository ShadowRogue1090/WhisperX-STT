# import json

# from utils.files import stage_file


# def run_export_stage(audio_file):

#     with open(stage_file(audio_file, "04_final.json"), encoding="utf8") as f:

#         result = json.load(f)

#     output = stage_file(audio_file, "transcript.txt")

#     with open(output, "w", encoding="utf8") as f:

#         for segment in result["segments"]:

#             speaker = segment.get("speaker", "UNKNOWN")
#             text = segment.get("text", "").strip()

#             start = segment.get("start")
#             end = segment.get("end")

#             if start is not None and end is not None:

#                 timestamp = f"[{start:08.2f} --> {end:08.2f}]"

#             else:

#                 timestamp = "[NO TIMESTAMP]"

#             f.write(f"{timestamp} {speaker}: {text}\n\n")

#     print("Export complete")
import json
import os

from utils.files import stage_file


def format_timestamp(seconds):
    """
    SRT timestamp format:
    HH:MM:SS,mmm
    """

    milliseconds = int((seconds % 1) * 1000)

    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    return f"{hours:02d}:" f"{minutes:02d}:" f"{secs:02d}," f"{milliseconds:03d}"


def export_srt(result, output):

    subtitle_id = 1

    MAX_WORDS = 8
    MAX_CHARS = 42

    with open(output, "w", encoding="utf8") as f:

        for segment in result["segments"]:

            speaker = segment.get("speaker", "UNKNOWN")

            words = segment.get("words", [])

            if not words:
                continue

            buffer = []

            start = None
            end = None

            for word in words:

                text = word.get("word", "").strip()

                if not text:
                    continue

                if start is None:
                    start = word["start"]

                end = word["end"]

                buffer.append(text)

                current_text = " ".join(buffer)

                if (
                    len(buffer) >= MAX_WORDS
                    or len(current_text) >= MAX_CHARS
                    or text.endswith((".", "?", "!"))
                ):

                    f.write(f"{subtitle_id}\n")

                    f.write(
                        f"{format_timestamp(start)} --> " f"{format_timestamp(end)}\n"
                    )

                    f.write(f"{speaker}: " f"{current_text}\n\n")

                    subtitle_id += 1

                    buffer = []

                    start = None
                    end = None

            if buffer:

                f.write(f"{subtitle_id}\n")

                f.write(f"{format_timestamp(start)} --> " f"{format_timestamp(end)}\n")

                f.write(f"{speaker}: " f"{' '.join(buffer)}\n\n")

                subtitle_id += 1


def export_markdown(result, output):

    with open(output, "w", encoding="utf8") as f:

        f.write("# Transcript\n\n")

        if "language" in result:

            f.write(f"**Language:** " f"{result['language']}\n\n")

        f.write("---\n\n")

        current_speaker = None

        for segment in result["segments"]:

            speaker = segment.get("speaker", "UNKNOWN")

            text = segment.get("text", "").strip()

            if not text:
                continue

            start = segment.get("start", 0)

            timestamp = format_timestamp(start)

            if speaker != current_speaker:

                current_speaker = speaker

                f.write(f"\n## {speaker}\n\n")

            f.write(f"**[{timestamp}]** " f"{text}\n\n")


def run_export_stage(audio_file):

    final_file = stage_file(audio_file, "04_final.json")

    with open(final_file, encoding="utf8") as f:
        result = json.load(f)

    # Get original filename without extension
    base_name = os.path.splitext(os.path.basename(audio_file))[0]

    # Save alongside the stage files using the original name
    output_dir = os.path.dirname(final_file)

    srt_output = os.path.join(output_dir, f"{base_name}.srt")
    md_output = os.path.join(output_dir, f"{base_name}.md")

    export_srt(result, srt_output)
    export_markdown(result, md_output)

    print(f"Export complete:\n{srt_output}\n{md_output}")
