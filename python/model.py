# Geometry/model conventions adapted from FastBEV++ (Apache-2.0); see THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inspect the published ONNX and extract independent image/BEV CNNs for Metis."""

import json
from collections import Counter

import numpy as np
import onnx
from common import HEADS, ROOT, sha256
from download import MODEL_URL
from onnx import numpy_helper, utils, version_converter

UPSTREAM_COMMIT = "04c2b3610f7c33ebc323723fe90998121e22cfac"
CAMERA_FEATURES = "onnx::Shape_320"
BEV_FEATURES = "input.204"


def rename_tensor(model, old, new):
    for node in model.graph.node:
        for names in (node.input, node.output):
            for index, name in enumerate(names):
                if name == old:
                    names[index] = new
    for values in (
        model.graph.input,
        model.graph.output,
        model.graph.value_info,
        model.graph.initializer,
    ):
        for value in values:
            if value.name == old:
                value.name = new


def main():
    folder = ROOT / "weights/onnx"
    original = folder / "fastbev_fp16.onnx"
    model = onnx.load(original)
    onnx.checker.check_model(model, full_check=True)
    corrected = onnx.ModelProto()
    corrected.CopyFrom(model)
    for index, node in enumerate(corrected.graph.node):
        if node.name == "Gather_166":
            raw_depth = node.output[0]
            node.output[0] = "depth_logits_gathered"
            corrected.graph.node.insert(
                index + 1,
                onnx.helper.make_node(
                    "Sigmoid", ["depth_logits_gathered"], [raw_depth], name="RestoreDepthSigmoid"
                ),
            )
            break
    else:
        raise ValueError("Expected the published depth gather node")
    onnx.checker.check_model(corrected, full_check=True)
    onnx.save(corrected, folder / "fastbev.onnx")
    shape = lambda value: [d.dim_value for d in value.type.tensor_type.shape.dim]
    assert [v.name for v in model.graph.input] == ["img", "coors_img", "coors_depth"]
    assert shape(model.graph.input[0]) == [6, 3, 256, 704]
    assert all(v.type.tensor_type.elem_type == onnx.TensorProto.FLOAT for v in model.graph.output)
    inferred = onnx.shape_inference.infer_shapes(model)
    temporary = folder / ".inferred.onnx"
    onnx.save(inferred, temporary)
    try:
        utils.extract_model(str(temporary), str(folder / "image.onnx"), ["img"], [CAMERA_FEATURES])
        image = onnx.load(folder / "image.onnx")
        image.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
        image.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 1
        for index, value in enumerate(image.graph.initializer):
            if value.name == "onnx::Resize_314":
                dims = numpy_helper.to_array(value).copy()
                dims[0] = 1
                image.graph.initializer[index].CopyFrom(numpy_helper.from_array(dims, value.name))
        del image.graph.value_info[:]
        rename_tensor(image, "img", "image")
        rename_tensor(image, CAMERA_FEATURES, "features")
        image = version_converter.convert_version(image, 17)
        image = onnx.shape_inference.infer_shapes(image)
        onnx.checker.check_model(image, full_check=True)
        onnx.save(image, folder / "image.onnx")
        utils.extract_model(
            str(temporary),
            str(folder / "bev.onnx"),
            [BEV_FEATURES],
            [f"output_{i}" for i in range(6)],
        )
        bev = onnx.load(folder / "bev.onnx")
        rename_tensor(bev, BEV_FEATURES, "bev")
        for index, name in enumerate(HEADS):
            rename_tensor(bev, f"output_{index}", name)
        bev = version_converter.convert_version(bev, 17)
        onnx.checker.check_model(bev, full_check=True)
        onnx.save(bev, folder / "bev.onnx")
        partitions = {
            "bev_backbone": (["input.204"], ["input.220", "input.252"], ["bev"], ["skip", "deep"]),
            "neck_resize": (
                ["input.220", "input.252"],
                ["input.256"],
                ["skip", "deep"],
                ["neck_input"],
            ),
            "bev_neck": (["input.256"], ["onnx::Resize_407"], ["neck_input"], ["neck_features"]),
            "head_resize": (["onnx::Resize_407"], ["input.268"], ["neck_features"], ["head_input"]),
            "detection_head": (
                ["input.268"],
                [f"output_{i}" for i in range(6)],
                ["head_input"],
                list(HEADS),
            ),
        }
        for name, (inputs, outputs, input_names, output_names) in partitions.items():
            path = folder / f"{name}.onnx"
            utils.extract_model(str(temporary), str(path), inputs, outputs)
            part = onnx.load(path)
            for old, new in zip(inputs + outputs, input_names + output_names):
                rename_tensor(part, old, new)
            part = version_converter.convert_version(part, 17)
            if name == "bev_backbone":
                # Return one terminal tensor: SDK 1.7 misdescribes a tapped intermediate output.
                part.graph.initializer.extend(
                    [
                        numpy_helper.from_array(np.array([], dtype=np.float32), "pack_roi"),
                        numpy_helper.from_array(
                            np.array([1, 1, 4, 4], dtype=np.float32), "pack_scale"
                        ),
                    ]
                )
                part.graph.node.extend(
                    [
                        onnx.helper.make_node(
                            "Resize",
                            ["deep", "pack_roi", "pack_scale"],
                            ["deep_repeated"],
                            mode="nearest",
                            coordinate_transformation_mode="asymmetric",
                            nearest_mode="floor",
                            name="PackRepeatDeep",
                        ),
                        onnx.helper.make_node(
                            "Concat",
                            ["skip", "deep_repeated"],
                            ["packed"],
                            axis=1,
                            name="PackBackboneOutputs",
                        ),
                    ]
                )
                del part.graph.output[:]
                part.graph.output.append(
                    onnx.helper.make_tensor_value_info(
                        "packed", onnx.TensorProto.FLOAT, [1, 640, 64, 64]
                    )
                )
            elif name == "neck_resize":
                constants = {
                    "unpack_start_skip": [0],
                    "unpack_end_skip": [128],
                    "unpack_axis_skip": [1],
                    "unpack_step_skip": [1],
                    "unpack_start_deep": [128, 0, 0],
                    "unpack_end_deep": [640, 64, 64],
                    "unpack_axis_deep": [1, 2, 3],
                    "unpack_step_deep": [1, 4, 4],
                }
                part.graph.initializer.extend(
                    [
                        numpy_helper.from_array(np.array(value, dtype=np.int64), key)
                        for key, value in constants.items()
                    ]
                )
                nodes = [
                    onnx.helper.make_node(
                        "Slice",
                        [
                            "packed",
                            f"unpack_start_{kind}",
                            f"unpack_end_{kind}",
                            f"unpack_axis_{kind}",
                            f"unpack_step_{kind}",
                        ],
                        [kind],
                        name=f"Unpack_{kind}",
                    )
                    for kind in ("skip", "deep")
                ]
                for index, node in enumerate(nodes):
                    part.graph.node.insert(index, node)
                del part.graph.input[:]
                part.graph.input.append(
                    onnx.helper.make_tensor_value_info(
                        "packed", onnx.TensorProto.FLOAT, [1, 640, 64, 64]
                    )
                )
            onnx.checker.check_model(part, full_check=True)
            onnx.save(part, path)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {
        "upstream_repository": "https://github.com/ymlab/advanced-fastbev",
        "upstream_commit": UPSTREAM_COMMIT,
        "onnx_url": MODEL_URL,
        "checkpoint_url": "https://drive.google.com/file/d/1q49_7aR7EWWOCAUK8iE5jfj1JdtL9gJy/view",
        "note": "Published filename says fp16, but the actual inputs, weights and outputs are FP32. Original graph retained unchanged.",
        "files": {
            name: {"sha256": sha256(folder / name)}
            for name in (
                "fastbev_fp16.onnx",
                "fastbev.onnx",
                "image.onnx",
                "bev.onnx",
                "bev_backbone.onnx",
                "neck_resize.onnx",
                "bev_neck.onnx",
                "head_resize.onnx",
                "detection_head.onnx",
            )
        },
        "correction": "fastbev.onnx restores the depth Sigmoid used in FastrayTransformer.forward but absent from FastBEVTRT.forward and the published export. No weights changed; unmodified export remains fastbev_fp16.onnx.",
        "input": {v.name: shape(v) for v in model.graph.input},
        "output": {HEADS[i]: shape(v) for i, v in enumerate(model.graph.output)},
        "operators": dict(Counter(n.op_type for n in model.graph.node)),
        "metis_partitions": ["image", "bev_backbone", "bev_neck", "detection_head"],
        "backbone_packing": "One [1,640,64,64] terminal output: skip concatenated with nearest-repeat(deep,4). Host takes every fourth deep sample before the original align_corners resize; FP32 mapping is exact.",
        "host_resize_graphs": ["neck_resize.onnx", "head_resize.onnx"],
        "split": {
            "image": ["image: 1x3x256x704", "features: 1x123x16x44"],
            "host": "Six-camera index gather, sigmoid depth modulation, seven-height sum, XY transpose",
            "bev": ["bev: 1x64x128x128", "reg, height, dim, rot, vel, heatmap"],
        },
    }
    (folder / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        "Prepared image.onnx (one camera) and bev.onnx (encoder + six detection heads).", flush=True
    )


if __name__ == "__main__":
    main()
