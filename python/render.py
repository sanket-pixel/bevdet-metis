# SPDX-License-Identifier: AGPL-3.0-or-later
"""Render saved detections without running the network again."""

import argparse
import json
import subprocess
from pathlib import Path

from common import ROOT, load_config
from dataset import Dataset
from preprocessing import Preprocessing
from visualize import Renderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    records = [json.loads(line) for line in args.detections.read_text().splitlines()]
    if not records or records[-1]["type"] != "summary" or not records[-1].get("completed", True):
        raise ValueError("Detections file is incomplete")
    metadata = records[0]
    if metadata["scene"] != config.scene:
        raise ValueError("Detections and config select different scenes")
    frames = {
        frame["token"]: frame for frame in Dataset(config.dataset_directory).frames(config.scene)
    }
    detections = [record for record in records if record["type"] == "frame"]
    if not detections or len(detections) != records[-1]["frames"]:
        raise ValueError("Detections count differs from the summary")
    is_video = args.output.suffix == ".mp4"
    if not is_video and (len(detections) != 1 or args.output.suffix != ".png"):
        raise ValueError("Choose PNG for one frame or MP4 for a video")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    renderer = Renderer(config.dataset_directory, metadata["backend"])
    preprocessing = Preprocessing(config.dataset_directory)
    writer = None
    try:
        if is_video:
            writer = subprocess.Popen(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    "1920x1080",
                    "-framerate",
                    "2",
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(args.output),
                ],
                stdin=subprocess.PIPE,
            )
        for index, record in enumerate(detections):
            frame = frames[record["token"]]
            prepared = {"camera_to_keyego": preprocessing.camera_transforms(frame)}
            image = renderer.Process(
                frame, record["boxes"], prepared, record["timings_ms"], index, len(detections)
            )
            if writer:
                writer.stdin.write(image.tobytes())
            else:
                image.save(args.output)
        if writer:
            writer.stdin.close()
            if writer.wait():
                raise RuntimeError("FFmpeg encoding failed")
    finally:
        preprocessing.Close()
        if writer and writer.poll() is None:
            writer.kill()
            writer.wait()
    print(f"Saved visualization: {args.output}", flush=True)


if __name__ == "__main__":
    main()
