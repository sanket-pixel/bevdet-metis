# SPDX-License-Identifier: AGPL-3.0-or-later
"""AxRuntime execution with metadata-driven padding, quantization and output identities."""

import json
from contextlib import ExitStack
from time import perf_counter

import numpy as np
import onnx
from common import HEADS, sha256
from inference import session
from preprocessing import view_transform


class AipuModel:
    def __init__(self, context, connection, stack, directory, source):
        provenance = json.loads((directory / "provenance.json").read_text())
        if provenance["onnx_sha256"] != sha256(source):
            raise ValueError(f"Compiled model differs from ONNX: {source.name}")
        manifest = json.loads((directory / "manifest.json").read_text())
        if (
            manifest["input_tensor_layout"] != "NHWC"
            or manifest.get("preprocess_graph")
            or manifest.get("postprocess_graph")
        ):
            raise ValueError("Expected an NHWC CNN without implicit host graphs")
        graph = onnx.load(source).graph
        self.input_names = [v.name for v in graph.input]
        self.output_names = [v.name for v in graph.output]
        self.model = stack.enter_context(context.load_model(directory / "model.json"))
        native_inputs = self.model.inputs()
        self.bindings = provenance["input_bindings"]
        if self.bindings[: len(self.input_names)] != list(range(len(self.input_names))):
            raise ValueError("Unexpected primary runtime input order")
        self.inputs = native_inputs[: len(self.input_names)]
        self.outputs = self.model.outputs()
        shapes = manifest["input_shapes"] + manifest["output_shapes"]
        logical = manifest["input_shapes_original"] + manifest["output_shapes_original"]
        paddings = manifest["n_padded_ch_inputs"] + manifest["n_padded_ch_outputs"]
        if len(manifest["quantize_params"]) != len(native_inputs):
            raise ValueError("Runtime input quantization count differs from metadata")
        parameters = manifest["quantize_params"][: len(self.inputs)] + manifest["dequantize_params"]
        source_shapes = [
            [d.dim_value for d in v.type.tensor_type.shape.dim]
            for v in list(graph.input) + list(graph.output)
        ]
        if logical != source_shapes:
            raise ValueError("Compiled ABI shapes differ from the source ONNX order")
        if len(self.inputs) != len(self.input_names) or len(self.outputs) != len(self.output_names):
            raise ValueError(
                f"{directory.name}: runtime has {len(self.inputs)} inputs/{len(self.outputs)} outputs; "
                f"ONNX declares {len(self.input_names)} inputs/{len(self.output_names)} outputs"
            )
        if len(self.bindings) != len(native_inputs):
            raise ValueError("Runtime input bindings differ from the compiled ABI")
        self.native_input_shapes = [info.shape for info in native_inputs]
        for index in range(len(self.inputs), len(native_inputs)):
            primary = self.inputs[self.bindings[index]]
            replica = native_inputs[index]
            if not (
                directory.name == "bev_backbone"
                and len(native_inputs) == 2
                and self.bindings == [0, 0]
                and replica.shape == (primary.size,)
                and replica.size == primary.size
                and replica.dtype == primary.dtype
                and replica.scale == primary.scale
                and replica.zero_point == primary.zero_point
                and manifest["quantize_params"][index] == manifest["quantize_params"][0]
            ):
                raise ValueError("Unrecognized runtime input replica")
        for info, physical, original, padding, params in zip(
            self.inputs + self.outputs, shapes, logical, paddings, parameters
        ):
            n, c, h, w = original
            if (
                list(info.shape) != physical
                or tuple(info.unpadded_shape) != (n, h, w, c)
                or np.dtype(info.dtype) != np.dtype("int8")
            ):
                raise ValueError(f"Unexpected runtime tensor layout: {info}")
            if (
                [v for pair in info.padding for v in pair] != padding
                or not np.isfinite(info.scale)
                or info.scale <= 0
            ):
                raise ValueError(f"Invalid runtime padding/scale: {info}")
            np.testing.assert_array_equal([info.scale, info.zero_point], params)
        description = json.loads((directory / "model.json").read_text())
        if [info.name for info in self.outputs] != [
            value["name"] for value in description["outputs"]
        ]:
            raise ValueError("Runtime ABI output order differs from model.json")
        self.input_buffers = [
            np.full(info.shape, int(np.clip(info.zero_point, -128, 127)), dtype=np.int8)
            for info in self.inputs
        ]
        self.quantization_work = [
            np.empty((n, min(c, 8), h, w), dtype=np.float32)
            for n, c, h, w in manifest["input_shapes_original"]
        ]
        self.output_buffers = [np.empty(info.shape, dtype=np.int8) for info in self.outputs]
        self.instance = stack.enter_context(
            connection.load_model_instance(
                self.model,
                num_sub_devices=1,
                aipu_cores=1,
                double_buffer=False,
                input_dmabuf=False,
                output_dmabuf=False,
            )
        )

    @staticmethod
    def crop(info):
        return tuple(
            slice(left, size - right) for size, (left, right) in zip(info.shape, info.padding)
        )

    def run(self, names, values):
        start = perf_counter()
        if set(values) != set(self.input_names):
            raise ValueError("Input identities differ from the compiled stage")
        for name, info, buffer, scratch in zip(
            self.input_names, self.inputs, self.input_buffers, self.quantization_work
        ):
            value = np.asarray(values[name], dtype=np.float32)
            n, h, w, channels = info.unpadded_shape
            if value.shape != (n, channels, h, w):
                raise ValueError(f"Unexpected input shape for {name}: {value.shape}")
            destination = buffer[self.crop(info)]
            # Small channel blocks avoid cache thrashing in the NCHW -> padded NHWC copy.
            for channel in range(0, channels, scratch.shape[1]):
                count = min(scratch.shape[1], channels - channel)
                work = scratch[:, :count]
                np.divide(value[:, channel : channel + count], np.float32(info.scale), out=work)
                np.add(work, np.float32(info.zero_point), out=work)
                np.rint(work, out=work)
                np.clip(work, -128, 127, out=work)
                destination[:, :, :, channel : channel + count] = work.astype(np.int8).transpose(
                    0, 2, 3, 1
                )
            if buffer.nbytes != info.size:
                raise ValueError("Runtime input buffer byte count mismatch")
        native_inputs = [
            self.input_buffers[binding].reshape(shape)
            for binding, shape in zip(self.bindings, self.native_input_shapes)
        ]
        quantized = perf_counter()
        self.instance.run(native_inputs, self.output_buffers)
        executed = perf_counter()
        outputs = {}
        for name, info, buffer in zip(self.output_names, self.outputs, self.output_buffers):
            raw = buffer[self.crop(info)]
            n, h, w, channels = raw.shape
            reordered = np.empty((n, channels, h, w), dtype=np.int8)
            for channel in range(0, channels, 8):
                reordered[:, channel : channel + 8] = raw[:, :, :, channel : channel + 8].transpose(
                    0, 3, 1, 2
                )
            logical = reordered.astype(np.float32)
            logical -= np.float32(info.zero_point)
            logical *= np.float32(info.scale)
            outputs[name] = logical
        end = perf_counter()
        self.timings = {
            "quantization": (quantized - start) * 1000,
            "runtime_execution": (executed - quantized) * 1000,
            "dequantization": (end - executed) * 1000,
        }
        return [outputs[name] for name in (names or self.output_names)]


class AipuInference:
    def __init__(self, config):
        from axelera.runtime import Context

        self.stack = ExitStack()
        try:
            context = self.stack.enter_context(Context())
            self.parts = {
                name: AipuModel(
                    context,
                    self.stack.enter_context(context.device_connect(num_sub_devices=1)),
                    self.stack,
                    config.aipu_directory / name,
                    config.model_directory / f"{name}.onnx",
                )
                for name in ("image", "bev_backbone", "bev_neck", "detection_head")
            }
            self.neck_resize = session(config.model_directory / "neck_resize.onnx", config.threads)
            self.head_resize = session(config.model_directory / "head_resize.onnx", config.threads)
        except BaseException:
            self.stack.close()
            raise

    def RunPart(self, name, values):
        start = perf_counter()
        part = self.parts[name]
        result = part.run(None, values)
        self.timings[name] = self.timings.get(name, 0) + (perf_counter() - start) * 1000
        for key, duration in part.timings.items():
            self.timings[key] = self.timings.get(key, 0) + duration
        return result

    def Process(self, prepared):
        self.timings = {}
        features = np.concatenate(
            [self.RunPart("image", {"image": image[None]})[0] for image in prepared["images"]]
        )
        start = perf_counter()
        bev = view_transform(features, prepared["coors_img"], prepared["coors_depth"])
        self.timings["view_transform"] = (perf_counter() - start) * 1000
        packed = self.RunPart("bev_backbone", {"bev": bev})[0]
        start = perf_counter()
        value = self.neck_resize.run(None, {"packed": packed})[0]
        self.timings["host_resize"] = (perf_counter() - start) * 1000
        value = self.RunPart("bev_neck", {"neck_input": value})[0]
        start = perf_counter()
        value = self.head_resize.run(None, {"neck_features": value})[0]
        self.timings["host_resize"] += (perf_counter() - start) * 1000
        heads = self.RunPart("detection_head", {"head_input": value})
        return dict(zip(HEADS, heads))

    def Close(self):
        self.stack.close()
