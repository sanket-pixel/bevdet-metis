# SPDX-License-Identifier: AGPL-3.0-or-later
"""Quantize and compile the four CNN stages with real nuScenes calibration tensors."""

import argparse
import json
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import onnxruntime as ort
from common import ROOT, load_config, sha256

PARTS = ("image", "bev_backbone", "bev_neck", "detection_head")


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def validate_runtime_abi(directory, part):
    metadata = json.loads((directory / "manifest.json").read_text())
    native = json.loads((directory / "model.json").read_text())
    inputs, outputs = native["inputs"], native["outputs"]
    input_shapes, output_shapes = metadata["input_shapes"], metadata["output_shapes"]
    if len(outputs) != len(output_shapes) or len(inputs) < len(input_shapes):
        raise ValueError(f"{part}: compiled tensor counts differ from the source model")
    for tensor, shape in zip(inputs[: len(input_shapes)] + outputs, input_shapes + output_shapes):
        if (
            tensor["shape"] != shape
            or tensor["dtype"] != "int8"
            or tensor["size"] != int(np.prod(shape))
        ):
            raise ValueError(f"{part}: invalid compiled tensor layout: {tensor}")
    bindings = list(range(len(input_shapes)))
    if len(inputs) != len(input_shapes):
        # SDK 1.7 emits a second physical copy for the first residual block's input.
        if not (
            part == "bev_backbone"
            and len(input_shapes) == 1
            and len(inputs) == 2
            and inputs[1]["shape"] == [inputs[0]["size"]]
            and inputs[1]["size"] == inputs[0]["size"]
            and inputs[1]["dtype"] == inputs[0]["dtype"]
        ):
            raise ValueError(f"{part}: unrecognized extra runtime inputs")
        bindings.append(0)
    return bindings


def calibration(part, manifest, onnx):
    folder = ROOT / "data/calibration"
    backbone = (
        session(onnx / "bev_backbone.onnx") if part in ("bev_neck", "detection_head") else None
    )
    resize = session(onnx / "neck_resize.onnx") if backbone else None
    neck = session(onnx / "bev_neck.onnx") if part == "detection_head" else None
    head_resize = session(onnx / "head_resize.onnx") if neck else None
    for record in manifest["frames"]:
        path = folder / record["file"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Calibration checksum mismatch: {path}")
        with np.load(path) as sample:
            if part == "image":
                for image in sample["images"]:
                    yield np.ascontiguousarray(image[None])
                continue
            bev = sample["bev"]
        if part == "bev_backbone":
            yield bev
            continue
        packed = backbone.run(None, {"bev": bev})[0]
        value = resize.run(None, {"packed": packed})[0]
        if part == "detection_head":
            value = neck.run(None, {"neck_input": value})[0]
            value = head_resize.run(None, {"neck_features": value})[0]
        yield np.ascontiguousarray(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--part", choices=(*PARTS, "all"), default="all")
    parser.add_argument("--cores", type=int, default=1)
    args = parser.parse_args()
    if args.cores != 1:
        parser.error(
            "These stages are prepared for one core each; parallel scheduling follows baseline verification"
        )
    pipeline_config = load_config(args.config)
    manifest_path = ROOT / "data/calibration/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest["models"].items():
        if sha256(pipeline_config.model_directory / name) != digest:
            raise ValueError("Calibration model changed; run make calibration again")
    from axelera import compiler
    from axelera.compiler import CompilerConfig

    settings = {
        "aipu_cores": 1,
        "resources": 0.25,
        "ptq_scheme": "per_tensor_min_max",
        "remove_output_dir": False,
        "run_graph_cleaner": False,
        "double_buffer": False,
        "enable_buffer_promotion": True,
        "enable_swicr": True,
        "dpu_allocation_algorithm": "lazy",
        "save_error_artifact": True,
        "max_processes_used_during_compilation": 4,
        "max_memory_used_during_compilation": 6 * 1024**3,
    }
    root = pipeline_config.aipu_directory
    root.mkdir(parents=True, exist_ok=True)
    for part in PARTS if args.part == "all" else (args.part,):
        onnx = pipeline_config.model_directory / f"{part}.onnx"
        destination = root / part
        previous = destination / "provenance.json"
        if previous.exists() and (destination / "model.json").exists():
            saved = json.loads(previous.read_text())
            if (
                saved["onnx_sha256"] == sha256(onnx)
                and saved["calibration_manifest_sha256"] == sha256(manifest_path)
                and saved["settings"] == settings
                and saved["compiler_version"] == version("axelera-compiler")
            ):
                saved["input_bindings"] = validate_runtime_abi(destination, part)
                previous.write_text(json.dumps(saved, indent=2) + "\n")
                print(f"Reusing compiled {part}", flush=True)
                continue
        print(
            f"Quantizing {part}: {manifest['camera_inputs'] if part == 'image' else manifest['bev_inputs']} real samples",
            flush=True,
        )
        with tempfile.TemporaryDirectory(prefix=f".{part}-", dir=root) as temporary:
            output = Path(temporary) / "compiled"
            config = CompilerConfig(**settings, model_name=f"fastbev_{part}", output_dir=output)
            quantized = compiler.quantize(
                model=str(onnx),
                calibration_dataset=calibration(part, manifest, pipeline_config.model_directory),
                config=config,
            )
            print(f"Compiling {part} for one Metis core", flush=True)
            compiler.compile(model=quantized, config=config, output_dir=output)
            provenance = {
                "part": part,
                "onnx_sha256": sha256(onnx),
                "compiler_version": version("axelera-compiler"),
                "calibration_manifest_sha256": sha256(manifest_path),
                "calibration_partition": manifest["partition"],
                "calibration_frames": [f["token"] for f in manifest["frames"]],
                "settings": settings,
                "input_bindings": validate_runtime_abi(output, part),
            }
            (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
            path = output / "manifest.json"
            runtime = json.loads(path.read_text())
            if runtime.get("preprocess_graph") or runtime.get("postprocess_graph"):
                raise ValueError(f"{part} unexpectedly contains CPU pre/post graphs")
            runtime.update(
                quantized_model_file=None, model_params_file=None, model_lib_file="model.json"
            )
            path.write_text(json.dumps(runtime, indent=2) + "\n")
            destination = root / part
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(output), str(destination))
        print(f"Compiled stage: {destination}", flush=True)


if __name__ == "__main__":
    main()
