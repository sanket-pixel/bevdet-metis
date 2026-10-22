# SPDX-License-Identifier: AGPL-3.0-or-later
"""CPU ONNX Runtime sessions for the exported model and its CNN partitions."""

import json

import numpy as np
import onnxruntime as ort
from common import HEADS, sha256
from preprocessing import view_transform


def session(path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class OnnxInference:
    def __init__(self, config, split=False):
        self.split = split
        folder = config.model_directory
        metadata = json.loads((folder / "provenance.json").read_text())
        for name in ("image.onnx", "bev.onnx") if split else ("fastbev.onnx",):
            if sha256(folder / name) != metadata["files"][name]["sha256"]:
                raise ValueError(f"Model checksum mismatch: {name}")
        if split:
            self.image = session(folder / "image.onnx", config.threads)
            self.bev = session(folder / "bev.onnx", config.threads)
        else:
            self.model = session(folder / "fastbev.onnx", config.threads)

    def ImageFeatures(self, images):
        return np.concatenate([self.image.run(None, {"image": image[None]})[0] for image in images])

    def Process(self, prepared):
        if self.split:
            features = self.ImageFeatures(prepared["images"])
            bev = view_transform(features, prepared["coors_img"], prepared["coors_depth"])
            values = self.bev.run(list(HEADS), {"bev": bev})
        else:
            values = self.model.run(
                None,
                {
                    "img": prepared["images"],
                    "coors_img": prepared["coors_img"],
                    "coors_depth": prepared["coors_depth"],
                },
            )
        return dict(zip(HEADS, values))
