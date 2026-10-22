# SPDX-License-Identifier: AGPL-3.0-or-later
"""Export the standalone PyTorch model and its Metis CNN partitions."""

import json
from pathlib import Path

import onnx
import torch
from torch import nn
from torch.nn import functional as F

from common import HEADS, ROOT, sha256
from model import load_model


class Image(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model.image(image)


class Bev(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, bev):
        return self.model.forward_bev(bev)


class PackedBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.img_bev_encoder_backbone

    def forward(self, bev):
        skip, _, deep = self.backbone(bev)
        deep = F.interpolate(deep, scale_factor=4, mode="nearest")
        return torch.cat((skip, deep), dim=1)


class PrepareNeck(nn.Module):
    def forward(self, packed):
        skip = packed[:, :128]
        deep = F.interpolate(
            packed[:, 128:, ::4, ::4], scale_factor=4, mode="bilinear", align_corners=True
        )
        return torch.cat((skip, deep), dim=1)


class Neck(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.conv = model.img_bev_encoder_neck.conv

    def forward(self, value):
        return self.conv(value)


class ResizeHead(nn.Module):
    def forward(self, value):
        return F.interpolate(value, scale_factor=2, mode="bilinear", align_corners=True)


class DetectionHead(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.upsample_projection = nn.Sequential(*list(model.img_bev_encoder_neck.up2.children())[1:])
        self.head = model.pts_bbox_head

    def forward(self, value):
        return self.head(self.upsample_projection(value))


def save(model, inputs, path, input_names, output_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        inputs,
        str(path),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    graph = onnx.load(path)
    onnx.checker.check_model(graph, full_check=True)


def main():
    torch.set_num_threads(1)
    directory = ROOT / "weights/onnx"
    checkpoint = ROOT / "weights/torch/epoch_20_ema.pth"
    model = load_model(checkpoint)
    image = torch.zeros(1, 3, 256, 704)
    images = torch.zeros(6, 3, 256, 704)
    bev = torch.zeros(1, 64, 128, 128)
    packed = torch.zeros(1, 640, 64, 64)
    neck_features = torch.zeros(1, 512, 64, 64)
    head_input = torch.zeros(1, 512, 128, 128)
    image_indices = torch.zeros(114688, dtype=torch.int64)
    depth_indices = torch.zeros(114688, dtype=torch.int64)

    exports = (
        ("image.onnx", Image(model), (image,), ("image",), ("features",)),
        ("bev_backbone.onnx", PackedBackbone(model), (bev,), ("bev",), ("packed",)),
        ("neck_resize.onnx", PrepareNeck(), (packed,), ("packed",), ("neck_input",)),
        ("bev_neck.onnx", Neck(model), (torch.zeros(1, 640, 64, 64),), ("neck_input",), ("neck_features",)),
        ("head_resize.onnx", ResizeHead(), (neck_features,), ("neck_features",), ("head_input",)),
        ("detection_head.onnx", DetectionHead(model), (head_input,), ("head_input",), HEADS),
        ("bev.onnx", Bev(model), (bev,), ("bev",), HEADS),
        (
            "fastbev.onnx",
            model,
            (images, image_indices, depth_indices),
            ("img", "coors_img", "coors_depth"),
            HEADS,
        ),
    )
    for name, module, args, input_names, output_names in exports:
        print(f"Exporting {name}", flush=True)
        save(module, args, directory / name, input_names, output_names)

    files = {
        name: {"sha256": sha256(directory / name)}
        for name, *_ in exports
    }
    metadata = {
        "source": "Standalone PyTorch implementation and official epoch_20_ema.pth checkpoint",
        "checkpoint": "../torch/epoch_20_ema.pth",
        "checkpoint_sha256": sha256(checkpoint),
        "opset": 17,
        "input": {
            "img": [6, 3, 256, 704],
            "coors_img": [114688],
            "coors_depth": [114688],
        },
        "output": {
            "reg": [1, 2, 128, 128],
            "height": [1, 1, 128, 128],
            "dim": [1, 3, 128, 128],
            "rot": [1, 2, 128, 128],
            "vel": [1, 2, 128, 128],
            "heatmap": [1, 10, 128, 128],
        },
        "files": files,
        "metis_partitions": ["image", "bev_backbone", "bev_neck", "detection_head"],
        "host_resize_graphs": ["neck_resize", "head_resize"],
        "depth": "Sigmoid is applied to the 59 depth logits before camera lifting, matching the training model.",
    }
    (directory / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Exported {len(exports)} ONNX graphs to {directory}", flush=True)


if __name__ == "__main__":
    main()
