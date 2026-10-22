# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare the unfused C++ deployment with the Python AxRuntime reference."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from common import HEADS, ROOT, load_config
from dataset import Dataset
from pipeline import Pipeline
from preprocessing import Preprocessing, view_transform
from runtime import AipuInference
from validate import tensor_errors


def cpp_run(binary, config, frames, start, output, dump=None):
    command = [
        str(binary),
        "--config",
        str(config),
        "--frames",
        str(frames),
        "--start",
        str(start),
        "--output",
        str(output),
        "--profile",
    ]
    if dump:
        command += ["--dump-directory", str(dump)]
    subprocess.run(command, check=True)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    return [record for record in records if record["type"] == "frame"]


def box_errors(reference, actual):
    result = {"python_count": len(reference), "cpp_count": len(actual)}
    result["class_order_equal"] = [b["class_id"] for b in reference] == [
        b["class_id"] for b in actual
    ]
    if len(reference) != len(actual) or not result["class_order_equal"]:
        result["passed"] = False
        return result
    tolerances = {"center": 1e-6, "dimensions": 1e-5, "yaw": 1e-6, "velocity": 1e-6, "score": 2e-7}
    result["max_abs"] = {}
    result["passed"] = True
    for name, tolerance in tolerances.items():
        a = np.asarray([box[name] for box in reference])
        b = np.asarray([box[name] for box in actual])
        result["max_abs"][name] = float(np.max(np.abs(a - b))) if a.size else 0.0
        if result["max_abs"][name] > tolerance:
            result["passed"] = False
    return result


def trace(prepared, inference):
    features = [
        inference.parts["image"].run(None, {"image": image[None]})[0]
        for image in prepared["images"]
    ]
    result = {f"image_{i}": image[None] for i, image in enumerate(prepared["images"])}
    result.update({f"features_{i}": feature for i, feature in enumerate(features)})
    result["bev"] = view_transform(
        np.concatenate(features), prepared["coors_img"], prepared["coors_depth"]
    )
    result["backbone"] = inference.parts["bev_backbone"].run(None, {"bev": result["bev"]})[0]
    result["neck_input"] = inference.neck_resize.run(None, {"packed": result["backbone"]})[0]
    result["neck_features"] = inference.parts["bev_neck"].run(
        None, {"neck_input": result["neck_input"]}
    )[0]
    result["head_input"] = inference.head_resize.run(
        None, {"neck_features": result["neck_features"]}
    )[0]
    result.update(
        zip(
            HEADS,
            inference.parts["detection_head"].run(
                list(HEADS), {"head_input": result["head_input"]}
            ),
        )
    )
    result["coors_img"] = prepared["coors_img"]
    result["coors_depth"] = prepared["coors_depth"]
    result["camera_to_keyego"] = np.asarray(prepared["camera_to_keyego"], dtype=np.float32)[None]
    return result


def compare_tensors(config, frames, directory):
    preprocessing = Preprocessing(config.dataset_directory)
    inference = AipuInference(config)
    report = []
    try:
        for frame in frames:
            prepared = preprocessing.Process(frame)
            reference = trace(prepared, inference)
            folder = directory / frame["token"]
            metadata = json.loads((folder / "tensors.json").read_text())
            tensors = {}
            for name, expected in reference.items():
                spec = metadata[name]
                actual = np.fromfile(folder / spec["file"], dtype=spec["dtype"]).reshape(
                    spec["shape"]
                )
                if actual.shape != expected.shape:
                    raise ValueError(
                        f"C++ shape mismatch for {name}: {actual.shape}, expected {expected.shape}"
                    )
                equal = np.array_equal(expected, actual)
                record = {
                    "exact": equal,
                    "different_elements": int(np.count_nonzero(expected != actual)),
                }
                if expected.dtype.kind == "f":
                    record.update(tensor_errors(expected, actual))
                tensors[name] = record
            required = [f"image_{i}" for i in range(6)] + ["coors_img", "coors_depth", *HEADS]
            passed = all(tensors[name]["exact"] for name in required)
            report.append({"token": frame["token"], "passed": passed, "tensors": tensors})
            print(
                f"Tensor comparison {frame['token']}: {'passed' if passed else 'DIFFERENCES'}",
                flush=True,
            )
    finally:
        inference.Close()
        preprocessing.Close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--binary", type=Path, default=ROOT / "build/bevdet_metis")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output-directory", type=Path, default=ROOT / "results/cpp")
    args = parser.parse_args()
    config = load_config(args.config)
    frames = Dataset(config.dataset_directory).frames(config.scene, args.start, args.frames)
    if len(frames) != args.frames or args.frames < 1 or args.start < 0:
        parser.error("Requested samples are outside the selected scene")
    folder = args.output_directory.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    cpp_records = cpp_run(
        args.binary.resolve(),
        args.config.resolve(),
        args.frames,
        args.start,
        folder / "profile.jsonl",
    )
    reference = Pipeline(config, "aipu")
    timings, boxes = [], []
    try:
        for index, (frame, cpp) in enumerate(zip(frames, cpp_records)):
            if frame["token"] != cpp["token"]:
                raise ValueError("C++ processed different samples")
            predicted, _, timing = reference.Process(frame)
            timings.append(timing)
            boxes.append({"token": frame["token"], **box_errors(predicted, cpp["boxes"])})
            print(
                f"Python reference {index + 1}/{len(frames)}: {timing['pipeline']:.1f} ms",
                flush=True,
            )
    finally:
        reference.Close()
    if len(cpp_records) != len(frames):
        raise ValueError("C++ did not process all requested samples")
    means = {name: float(np.mean([t[name] for t in timings])) for name in timings[0]}
    python_profile = {
        "backend": "aipu",
        "frames": len(frames),
        "mean_timings_ms": means,
        "processing_fps": 1000 / means["pipeline"],
    }
    (folder / "python_profile.json").write_text(json.dumps(python_profile, indent=2) + "\n")
    count = min(3, args.frames)
    cpp_run(
        args.binary.resolve(),
        args.config.resolve(),
        count,
        args.start,
        folder / "validation.jsonl",
        folder / "tensors",
    )
    tensor_report = compare_tensors(config, frames[:count], folder / "tensors")
    cpp_profile = json.loads((folder / "profile.json").read_text())
    passed = all(b["passed"] for b in boxes) and all(t["passed"] for t in tensor_report)
    report = {
        "passed": passed,
        "reference": "Python AxRuntime, same compiled models and YAML",
        "scope": "Port parity, not nuScenes mAP/NDS; includes first frame and JPEG reads, excludes model loading/export/rendering",
        "frames": len(frames),
        "python": python_profile,
        "cpp": cpp_profile,
        "speedup": means["pipeline"] / cpp_profile["mean_timings_ms"]["pipeline"],
        "detections": boxes,
        "tensor_comparison": tensor_report,
    }
    output = folder / "comparison.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Python {means['pipeline']:.1f} ms; C++ {cpp_profile['mean_timings_ms']['pipeline']:.1f} ms",
        flush=True,
    )
    print(f"Comparison {'passed' if passed else 'FAILED'}: {output}", flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
