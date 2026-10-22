# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check PyTorch/ONNX partition parity and Metis output differences."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from common import HEADS, ROOT, load_config
from dataset import Dataset
from inference import OnnxInference, session
from model import lift_features, load_model, voxel_coordinates
from postprocessing import Postprocessing
from preprocessing import Preprocessing, view_transform


def errors(reference, actual):
    if actual.shape != reference.shape or not np.isfinite(actual).all():
        raise ValueError("Invalid inference output shape or nonfinite values")
    norm = max(float(np.linalg.norm(reference)), 1e-12)
    return {
        "max_abs": float(np.max(np.abs(reference - actual))),
        "relative_l2": float(np.linalg.norm(reference - actual) / norm),
    }


def check_torch_view(model, preprocessing, prepared, features):
    torch_voxels = voxel_coordinates().numpy()
    checkpoint_voxels = model.img_view_transformer.voxel_coords.detach().numpy()
    np.testing.assert_allclose(torch_voxels, checkpoint_voxels, rtol=0, atol=4e-5)
    np.testing.assert_allclose(preprocessing.voxels, torch_voxels, rtol=0, atol=4e-5)
    with torch.inference_mode():
        expected = lift_features(
            torch.from_numpy(features.copy()),
            torch.from_numpy(prepared["coors_img"]),
            torch.from_numpy(prepared["coors_depth"]),
        ).numpy()
    actual = view_transform(features, prepared["coors_img"], prepared["coors_depth"])
    np.testing.assert_allclose(expected, actual, rtol=1e-4, atol=1e-5)
    return {
        "voxel_coordinates_max_abs": float(np.max(np.abs(preprocessing.voxels - checkpoint_voxels))),
        "index_entries": len(prepared["coors_img"]),
        "bev_max_abs": float(np.max(np.abs(expected - actual))),
    }


def check_aipu(config, preprocessing, frames, full):
    from runtime import AipuInference

    hardware = AipuInference(config)
    decoding = Postprocessing(config)
    report = []
    try:
        for frame in frames:
            prepared = preprocessing.Process(frame)
            reference = full.Process(prepared)
            actual = hardware.Process(prepared)
            report.append(
                {
                    "token": frame["token"],
                    "scene": frame["scene"],
                    "heads": {name: errors(reference[name], actual[name]) for name in HEADS},
                    "fp32_detections": len(decoding.Process(reference)),
                    "int8_detections": len(decoding.Process(actual)),
                }
            )
            print(f"Compared Metis / FP32: {frame['scene']} / {frame['token']}", flush=True)
    finally:
        hardware.Close()
    path = ROOT / "results/aipu_validation.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "reference": "Standalone PyTorch-exported FP32 ONNX vs Metis CNN stages",
                "scope": "Real-frame numeric differences, not nuScenes mAP/NDS evaluation",
                "frames": report,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--backend", choices=("onnx", "aipu"), default="onnx")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = Dataset(config.dataset_directory)
    preprocessing = Preprocessing(config.dataset_directory)
    full = OnnxInference(config)
    frames = [
        dataset.frames(config.scene, count=1)[0],
        dataset.frames(config.scene, start=15, count=1)[0],
        dataset.frames("scene-0061", count=1)[0],
    ]
    try:
        if args.backend == "aipu":
            check_aipu(config, preprocessing, frames, full)
            return

        model = load_model(ROOT / "weights/torch/epoch_20_ema.pth")
        split = OnnxInference(config, split=True)
        parts = {
            name: session(config.model_directory / f"{name}.onnx", config.threads)
            for name in ("bev_backbone", "neck_resize", "bev_neck", "head_resize", "detection_head")
        }
        report = []
        for index, frame in enumerate(frames):
            prepared = preprocessing.Process(frame)
            reference = full.Process(prepared)
            features = split.ImageFeatures(prepared["images"])
            bev = view_transform(features, prepared["coors_img"], prepared["coors_depth"])
            packed = parts["bev_backbone"].run(None, {"bev": bev})[0]
            value = parts["neck_resize"].run(None, {"packed": packed})[0]
            value = parts["bev_neck"].run(None, {"neck_input": value})[0]
            value = parts["head_resize"].run(None, {"neck_features": value})[0]
            actual = dict(
                zip(HEADS, parts["detection_head"].run(None, {"head_input": value}))
            )
            head_errors = {}
            for name in HEADS:
                np.testing.assert_allclose(reference[name], actual[name], rtol=5e-5, atol=2e-5)
                head_errors[name] = errors(reference[name], actual[name])
            record = {"token": frame["token"], "scene": frame["scene"], "heads": head_errors}
            if index == 0:
                record["pytorch_view_transform"] = check_torch_view(
                    model, preprocessing, prepared, features
                )
            report.append(record)
            print(f"Validated {frame['scene']} / {frame['token']}", flush=True)
        output = ROOT / "results/onnx_validation.json"
        output.write_text(json.dumps({"frames": report}, indent=2) + "\n")
        print(f"Saved {output}", flush=True)
    finally:
        preprocessing.Close()


if __name__ == "__main__":
    main()
