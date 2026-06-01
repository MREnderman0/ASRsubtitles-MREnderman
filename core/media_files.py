from __future__ import annotations

import glob
import os
import sys

from core.utils import load_key


def find_video_files(save_path: str = "output") -> str:
    video_files = [
        file
        for file in glob.glob(os.path.join(save_path, "*"))
        if os.path.splitext(file)[1][1:].lower() in load_key("allowed_video_formats")
    ]
    if sys.platform.startswith("win"):
        video_files = [file.replace("\\", "/") for file in video_files]
    video_files = [file for file in video_files if not file.startswith("output/output")]
    if len(video_files) != 1:
        raise ValueError(f"Number of videos found {len(video_files)} is not unique. Please check.")
    return video_files[0]
