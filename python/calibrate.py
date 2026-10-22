# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cache real mini_train six-camera images and reference BEV tensors for quantization."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from common import CAMERAS, ROOT, load_config, sha256
from dataset import Dataset
from inference import OnnxInference
from preprocessing import Preprocessing, view_transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--frames", type=int, default=32)
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = Dataset(config.dataset_directory)
    frames = dataset.calibration_frames(args.frames)
    folder = ROOT / "data/calibration"
    folder.mkdir(parents=True, exist_ok=True)
    model_hashes = {
        name: sha256(config.model_directory / name)
        for name in ("image.onnx", "bev.onnx", "fastbev.onnx")
    }
    manifest = folder / "manifest.json"
    if manifest.exists():
        cached = json.loads(manifest.read_text())
        if (
            cached["models"] == model_hashes
            and [f["token"] for f in cached["frames"]] == [f["token"] for f in frames]
            and all(
                (folder / f["file"]).exists() and sha256(folder / f["file"]) == f["sha256"]
                for f in cached["frames"]
            )
        ):
            print(
                f"Reusing {len(frames)} rig frames ({len(frames) * 6} camera inputs) and {len(frames)} BEV inputs"
            )
            return
    preprocessing = Preprocessing(config.dataset_directory)
    inference = OnnxInference(config, split=True)
    records = []
    start = perf_counter()
    for index, frame in enumerate(frames):
        prepared = preprocessing.Process(frame)
        features = inference.ImageFeatures(prepared["images"])
        bev = view_transform(features, prepared["coors_img"], prepared["coors_depth"])
        path = folder / f"{index:04d}.npz"
        np.savez_compressed(path, images=prepared["images"], bev=bev)
        records.append(
            {
                "token": frame["token"],
                "scene": frame["scene"],
                "file": path.name,
                "sha256": sha256(path),
                "images": [
                    {
                        "file": frame["cameras"][c]["file"],
                        "sha256": sha256(config.dataset_directory / frame["cameras"][c]["file"]),
                    }
                    for c in CAMERAS
                ],
            }
        )
        print(
            f"Calibration {index + 1}/{len(frames)}: {frame['scene']} ({perf_counter() - start:.1f}s)",
            flush=True,
        )
    preprocessing.Close()
    manifest.write_text(
        json.dumps(
            {
                "models": model_hashes,
                "partition": "nuScenes v1.0-mini official mini_train",
                "camera_inputs": len(frames) * 6,
                "bev_inputs": len(frames),
                "frames": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved calibration manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
