# Copyright (c) Phigent Robotics. All rights reserved.
# Adapted model components from FastBEV++; see THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: Apache-2.0
"""Standalone PyTorch definition of the deployed FastBEV++ network."""

import torch
from torch import nn
from torch.nn import functional as F

from common import HEADS


def convolution(in_channels, out_channels, kernel_size, stride=1, bias=False):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        bias=bias,
    )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else None
        )

    def forward(self, value):
        residual = value if self.downsample is None else self.downsample(value)
        value = self.relu(self.bn1(self.conv1(value)))
        value = self.relu(self.bn2(self.conv2(value)))
        value = self.bn3(self.conv3(value))
        return self.relu(value + residual)


class ImageBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.in_channels = 64
        self.layer1 = self._layer(64, 3, stride=1)
        self.layer2 = self._layer(128, 4, stride=2)
        self.layer3 = self._layer(256, 6, stride=2)
        self.layer4 = self._layer(512, 3, stride=2)

    def _layer(self, channels, count, stride):
        blocks = [Bottleneck(self.in_channels, channels, stride)]
        self.in_channels = channels * Bottleneck.expansion
        blocks.extend(Bottleneck(self.in_channels, channels) for _ in range(count - 1))
        return nn.Sequential(*blocks)

    def forward(self, image):
        image = self.maxpool(self.relu(self.bn1(self.conv1(image))))
        image = self.layer1(image)
        image = self.layer2(image)
        level_16 = self.layer3(image)
        level_32 = self.layer4(level_16)
        return level_16, level_32


class ImageNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [nn.ModuleDict({"conv": nn.Conv2d(c, 256, 1)}) for c in (1024, 2048)]
        )
        self.fpn_convs = nn.ModuleList(
            [nn.ModuleDict({"conv": convolution(256, 256, 3, bias=True)})]
        )

    def forward(self, features):
        shallow = self.lateral_convs[0]["conv"](features[0])
        deep = self.lateral_convs[1]["conv"](features[1])
        shallow = shallow + F.interpolate(deep, scale_factor=2, mode="nearest")
        return self.fpn_convs[0]["conv"](shallow)


def voxel_coordinates():
    x, y, z = torch.meshgrid(
        torch.arange(128), torch.arange(128), torch.arange(7), indexing="ij"
    )
    coords = torch.stack((x, y, z), dim=-1).float()
    lower = torch.tensor([-51.2, -51.2, -2.5])
    interval = torch.tensor([0.8, 0.8, 1.0])
    coords = coords * interval + (lower - interval / 2.0)
    return coords.reshape(-1, 3)


class ImageView(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_net = nn.Conv2d(256, 123, 1)
        self.voxel_coords = nn.Parameter(voxel_coordinates(), requires_grad=False)


def lift_features(image_features, coors_img, coors_depth):
    flat = image_features.permute(0, 2, 3, 1).reshape(-1, 123)
    flat = torch.cat((torch.zeros_like(flat[:1]), flat[1:]), dim=0)
    depth = flat[:, :59].reshape(-1).sigmoid()
    features = flat[:, 59:]
    lifted = features[coors_img] * depth[coors_depth].unsqueeze(1)
    return lifted.reshape(1, 128, 128, 7, 64).sum(dim=3).permute(0, 3, 2, 1)


class BevBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = convolution(in_channels, out_channels, 3, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = convolution(out_channels, out_channels, 3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = (
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
            if stride != 1 or in_channels != out_channels
            else None
        )

    def forward(self, value):
        residual = value if self.downsample is None else self.downsample(value)
        value = self.relu(self.bn1(self.conv1(value)))
        value = self.bn2(self.conv2(value))
        return self.relu(value + residual)


class BevBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (128, 256, 512)
        current = 64
        layers = []
        for output in channels:
            layers.append(
                nn.Sequential(
                    BevBasicBlock(current, output, stride=2),
                    BevBasicBlock(output, output, stride=1),
                )
            )
            current = output
        self.layers = nn.Sequential(*layers)

    def forward(self, bev):
        features = []
        for layer in self.layers:
            bev = layer(bev)
            features.append(bev)
        return features


class BevNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            convolution(640, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            convolution(512, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            convolution(512, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1),
        )


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = convolution(in_channels, out_channels, 3)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, value):
        return self.relu(self.bn(self.conv(value)))


class DetectionTask(nn.Module):
    def __init__(self):
        super().__init__()
        for name, channels in (
            ("reg", 2),
            ("height", 1),
            ("dim", 3),
            ("rot", 2),
            ("vel", 2),
            ("heatmap", 10),
        ):
            setattr(
                self,
                name,
                nn.Sequential(ConvNormAct(64, 64), nn.Conv2d(64, channels, 3, padding=1)),
            )

    def forward(self, value):
        return {name: getattr(self, name)(value) for name in HEADS}


class DetectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv = ConvNormAct(256, 64)
        self.task_heads = nn.ModuleList([DetectionTask()])

    def forward(self, value):
        value = self.shared_conv(value)
        return tuple(self.task_heads[0](value)[name] for name in HEADS)


class FastBEVModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_backbone = ImageBackbone()
        self.img_neck = ImageNeck()
        self.img_view_transformer = ImageView()
        self.img_bev_encoder_backbone = BevBackbone()
        self.img_bev_encoder_neck = BevNeck()
        self.pts_bbox_head = DetectionHead()

    def image(self, images):
        return self.img_view_transformer.depth_net(self.img_neck(self.img_backbone(images)))

    def bev_features(self, bev):
        skip, _, deep = self.img_bev_encoder_backbone(bev)
        deep = F.interpolate(deep, scale_factor=4, mode="bilinear", align_corners=True)
        return self.img_bev_encoder_neck.conv(torch.cat((skip, deep), dim=1))

    def forward_bev(self, bev):
        features = self.bev_features(bev)
        features = self.img_bev_encoder_neck.up2(features)
        return self.pts_bbox_head(features)

    def forward(self, images, coors_img, coors_depth):
        image_features = self.image(images)
        bev = lift_features(image_features, coors_img, coors_depth)
        return self.forward_bev(bev)


def load_model(checkpoint):
    model = FastBEVModel()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = saved.get("state_dict", saved)
    model.load_state_dict(state, strict=True)
    return model.eval()
