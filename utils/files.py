import os

from config import WORK_DIR


def get_job_name(audio_file):

    return os.path.splitext(os.path.basename(audio_file))[0]


def get_work_folder(audio_file):

    folder = os.path.join(WORK_DIR, get_job_name(audio_file))

    os.makedirs(folder, exist_ok=True)

    return folder


def stage_file(audio_file, filename):

    return os.path.join(get_work_folder(audio_file), filename)
