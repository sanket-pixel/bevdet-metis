# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run the six-camera Python ONNX baseline and save detections and a demo."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from common import CAMERAS, ROOT, load_config
from dataset import Dataset
from pipeline import Pipeline
from visualize import Renderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--backend", choices=["onnx", "split", "upstream", "aipu"], default="onnx")
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.frames < 1 or args.start < 0:
        parser.error("Choose positive frames and nonnegative start")
    config = load_config(args.config)
    dataset = Dataset(config.dataset_directory)
    frames = dataset.frames(config.scene, args.start, args.frames)
    if len(frames) != args.frames:
        parser.error(f"Selected scene has only {len(frames)} available frames at this start")
    pipeline = Pipeline(config, args.backend)
    renderer = Renderer(config.dataset_directory, args.backend)
    output = args.output or ROOT / "results" / args.backend / (
        "demo.png" if len(frames) == 1 else "demo.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    if len(frames) > 1 and not args.profile:
        command = [
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
            str(output),
        ]
        writer = subprocess.Popen(command, stdin=subprocess.PIPE)
    all_timings = []
    detections = output.with_suffix(".jsonl")
    try:
        with detections.open("w") as stream:
            stream.write(
                json.dumps(
                    {
                        "type": "metadata",
                        "backend": args.backend,
                        "scene": config.scene,
                        "camera_order": CAMERAS,
                        "coordinate_frame": "ego at CAM_FRONT_LEFT timestamp; x forward, y left, z up",
                        "box_format": "center xyz, dimensions length width height, yaw about z, velocity xy; metres and radians",
                        "timings": "Pipeline includes image reads; excludes model loading, rendering, JSON export and encoding",
                    }
                )
                + "\n"
            )
            for index, frame in enumerate(frames):
                boxes, prepared, timings = pipeline.Process(frame)
                all_timings.append(timings)
                stream.write(
                    json.dumps(
                        {
                            "type": "frame",
                            "index": args.start + index,
                            "token": frame["token"],
                            "timestamp": frame["timestamp"],
                            "boxes": boxes,
                            "timings_ms": timings,
                            "ground_truth": dataset.ground_truth(frame),
                        }
                    )
                    + "\n"
                )
                if not args.profile:
                    rendered = renderer.Process(frame, boxes, prepared, timings, index, len(frames))
                    if writer:
                        writer.stdin.write(rendered.tobytes())
                    else:
                        rendered.save(output)
                print(
                    f"{index + 1}/{len(frames)}: {len(boxes)} boxes, {timings['pipeline']:.1f} ms",
                    flush=True,
                )
            stream.write(json.dumps({"type": "summary", "frames": len(frames)}) + "\n")
        if writer:
            writer.stdin.close()
            if writer.wait():
                raise RuntimeError("FFmpeg encoding failed")
    finally:
        pipeline.Close()
        if writer and writer.poll() is None:
            writer.kill()
            writer.wait()
    means = {key: float(np.mean([t[key] for t in all_timings])) for key in all_timings[0]}
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "frames": len(frames),
                "backend": args.backend,
                "mean_timings_ms": means,
                "processing_fps": 1000 / means["pipeline"],
                "timing": "Includes six JPEG reads and projection indices; excludes model loading, visualization and encoding",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(means, indent=2), flush=True)
    print(f"Saved detections: {detections}", flush=True)
    if not args.profile:
        print(f"Saved demo: {output}", flush=True)


if __name__ == "__main__":
    main()
