# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check real-frame ONNX partition equivalence and the upstream view-transform math."""

import argparse
import ast
import json
import types
from pathlib import Path

import numpy as np
from common import CAMERAS, HEADS, ROOT, load_config
from dataset import Dataset
from inference import OnnxInference, session
from preprocessing import Preprocessing, view_transform


def tensor_errors(reference, actual):
    if actual.shape != reference.shape or not np.isfinite(actual).all():
        raise ValueError("Invalid inference output shape or nonfinite values")
    denominator = max(float(np.linalg.norm(reference)), 1e-12)
    return {
        "max_abs": float(np.max(np.abs(reference - actual))),
        "relative_l2": float(np.linalg.norm(reference - actual) / denominator),
        "cosine": float(
            np.dot(reference.ravel(), actual.ravel())
            / max(denominator * float(np.linalg.norm(actual)), 1e-12)
        ),
    }


def check_aipu(config, preprocessing, frames, full):
    from postprocessing import Postprocessing
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
                    "heads": {name: tensor_errors(reference[name], actual[name]) for name in HEADS},
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
                "reference": "Corrected FP32 ONNX vs four INT8 Metis CNN stages and exact host operators",
                "scope": "Real-frame execution and numeric differences, not nuScenes mAP/NDS evaluation",
                "frames": report,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {path}", flush=True)


def check_upstream_view(preprocessing, prepared, frame, features):
    try:
        import torch
        from torch import nn
    except ImportError:
        return {"status": "skipped; Torch is optional for this independent source comparison"}
    source = ROOT / "third_party/advanced_fastbev/mmdet3d/models/necks/fastray_transformer.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FastrayTransformer"
    )
    namespace = {"torch": torch, "nn": nn}
    for method in cls.body:
        if isinstance(method, ast.FunctionDef) and method.name in (
            "create_grid_infos",
            "create_voxel_coords",
            "get_fastray_input",
            "forward",
        ):
            exec(  # noqa: S102 - execute selected methods from the pinned local upstream source.
                compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace
            )
    ref = types.SimpleNamespace(
        stride=16,
        image_size=(256, 704),
        D=59,
        accelerate=False,
        grid_config={"depth": [1, 60, 1]},
        depth_act="sigmoid",
        use_depth=True,
        out_channels=64,
        is_transpose=True,
        fuse_type="sum",
        downsample=nn.Identity(),
    )
    namespace["create_grid_infos"](
        ref, x=[-51.2, 51.2, 0.8], y=[-51.2, 51.2, 0.8], z=[-2.5, 4.5, 1]
    )
    ref.voxel_coords = namespace["create_voxel_coords"](ref)
    ref.get_fastray_input = types.MethodType(namespace["get_fastray_input"], ref)
    ref.depth_net = lambda _: torch.from_numpy(features.copy())
    np.testing.assert_array_equal(preprocessing.voxels, ref.voxel_coords.detach().numpy())
    rot = torch.eye(3)[None, None].repeat(1, 6, 1, 1)
    rot[:, :, :2, :2] *= 0.44
    translation = torch.zeros((1, 6, 3))
    translation[:, :, 1] = -140
    inputs = [
        torch.zeros((1, 6, 256, 16, 44)),
        torch.tensor(np.array(prepared["camera_to_keyego"]))[None],
        torch.eye(4)[None, None].repeat(1, 6, 1, 1),
        torch.tensor(
            np.array([frame["cameras"][c]["calibration"]["camera_intrinsic"] for c in CAMERAS]),
            dtype=torch.float32,
        )[None],
        rot,
        translation,
        torch.eye(4)[None],
    ]
    voxels, pixels, depth = ref.get_fastray_input(inputs)
    order = torch.argsort(voxels[0])
    pixels = pixels[0][order]
    depth = depth[0][order]
    indices = (pixels[:, 0] * 16 * 44 + pixels[:, 1] * 44 + pixels[:, 2]).numpy()
    np.testing.assert_array_equal(indices, prepared["coors_img"])
    np.testing.assert_array_equal(indices * 59 + depth.numpy(), prepared["coors_depth"])
    with torch.no_grad():
        expected, _ = namespace["forward"](ref, inputs)
    actual = view_transform(features, prepared["coors_img"], prepared["coors_depth"])
    np.testing.assert_allclose(expected.numpy(), actual, rtol=1e-4, atol=1e-5)
    return {
        "status": "passed",
        "index_entries": len(indices),
        "image_index_differences": 0,
        "depth_index_differences": 0,
        "bev_max_abs": float(np.max(np.abs(expected.numpy() - actual))),
        "reference": "Unmodified upstream FastrayTransformer.forward with sigmoid; CNN outputs supplied by the published weights",
    }


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
    if args.backend == "aipu":
        check_aipu(config, preprocessing, frames, full)
        return
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
        actual = dict(zip(HEADS, parts["detection_head"].run(None, {"head_input": value})))
        errors = {}
        for name in HEADS:
            np.testing.assert_allclose(reference[name], actual[name], rtol=5e-5, atol=2e-5)
            errors[name] = {
                "max_abs": float(np.max(np.abs(reference[name] - actual[name]))),
                "relative_l2": float(
                    np.linalg.norm(reference[name] - actual[name])
                    / max(np.linalg.norm(reference[name]), 1e-12)
                ),
            }
        record = {"token": frame["token"], "scene": frame["scene"], "heads": errors}
        if index == 0:
            record["upstream_view_transform"] = check_upstream_view(
                preprocessing, prepared, frame, features
            )
        report.append(record)
        print(f"Validated {frame['scene']} / {frame['token']}", flush=True)
    folder = ROOT / "results"
    folder.mkdir(exist_ok=True)
    (folder / "onnx_validation.json").write_text(
        json.dumps(
            {
                "reference": "Corrected full graph vs four CNN partitions plus exact host operators",
                "rtol": 5e-5,
                "atol": 2e-5,
                "frames": report,
            },
            indent=2,
        )
        + "\n"
    )
    print("Saved results/onnx_validation.json", flush=True)


if __name__ == "__main__":
    main()
