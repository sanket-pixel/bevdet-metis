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
    parser.add_argument("--detections", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=2, help="Video playback frame rate (default: 2)")
    args = parser.parse_args()
    if args.fps < 1:
        parser.error("--fps must be positive")
    config = load_config(args.config)
    metadata = None
    detections = []
    for detections_file in args.detections:
        records = [json.loads(line) for line in detections_file.read_text().splitlines()]
        if not records or records[-1]["type"] != "summary" or not records[-1].get("completed", True):
            raise ValueError(f"Detections file is incomplete: {detections_file}")
        current_metadata = records[0]
        if metadata is None:
            metadata = current_metadata
        elif current_metadata["backend"] != metadata["backend"]:
            raise ValueError("All detections files must use the same backend")
        current_detections = [record for record in records if record["type"] == "frame"]
        if not current_detections or len(current_detections) != records[-1]["frames"]:
            raise ValueError(f"Detections count differs from the summary: {detections_file}")
        for record in current_detections:
            record.setdefault("scene", current_metadata["scene"])
        detections.extend(current_detections)
    if not detections:
        raise ValueError("No detections to render")
    is_video = args.output.suffix == ".mp4"
    if not is_video and (len(detections) != 1 or args.output.suffix != ".png"):
        raise ValueError("Choose PNG for one frame or MP4 for a video")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset = Dataset(config.dataset_directory)
    frames = {}
    for scene in dict.fromkeys(record["scene"] for record in detections):
        frames.update({frame["token"]: frame for frame in dataset.frames(scene)})
    mean_pipeline_ms = sum(record["timings_ms"]["pipeline"] for record in detections) / len(detections)
    renderer = Renderer(config.dataset_directory, 1000.0 / mean_pipeline_ms)
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
                    str(args.fps),
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
                frame, record["boxes"], prepared, index, len(detections)
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
