import json
import os
from datetime import datetime

import soundfile as sf

from utils.files import stage_file


def run_metadata_stage(audio_file):

    output = stage_file(audio_file, "00_metadata.json")

    if os.path.exists(output):
        return

    info = sf.info(audio_file)

    metadata = {
        "filename": audio_file,
        "duration": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "format": info.format,
        "subtype": info.subtype,
        "created": datetime.now().isoformat(),
    }

    with open(output, "w", encoding="utf8") as f:
        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("Metadata complete")
