# Geometry/model conventions adapted from FastBEV++ (Apache-2.0); see THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Image preparation and deterministic six-camera view indices, following upstream."""

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import numpy as np
from common import CAMERAS
from dataset import transform
from PIL import Image


class Preprocessing:
    def __init__(self, dataset_directory):
        self.directory = dataset_directory
        xyz = np.stack(
            np.meshgrid(np.arange(128), np.arange(128), np.arange(7), indexing="ij"), axis=-1
        )
        # Preserve upstream create_voxel_coords(), including its negative half-cell offset.
        interval = np.array([0.8, 0.8, 1.0], dtype=np.float32)
        lower = np.array([-51.2, -51.2, -2.5], dtype=np.float32)
        self.voxels = (xyz.astype(np.float32) * interval + (lower - interval / 2)).reshape(-1, 3)
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        self.executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="camera")

    def camera_transforms(self, frame):
        globals_, sensors = [], []
        for name in CAMERAS:
            record = frame["cameras"][name]
            c, p = record["calibration"], record["pose"]
            globals_.append(
                transform(p["rotation"], p["translation"]).astype(np.float32).astype(np.float64)
            )
            sensors.append(
                transform(c["rotation"], c["translation"]).astype(np.float32).astype(np.float64)
            )
        key_from_world = np.linalg.inv(globals_[0])
        return [(key_from_world @ g @ s).astype(np.float32) for g, s in zip(globals_, sensors)]

    def ReadCamera(self, record):
        with Image.open(self.directory / record["file"]) as image:
            image = image.convert("RGB")
            resize = 704 / image.width
            new_size = (int(image.width * resize), int(image.height * resize))
            crop_y = new_size[1] - 256
            crop_x = max(0, new_size[0] - 704) // 2
            image = image.resize(new_size, Image.Resampling.BICUBIC).crop(
                (crop_x, crop_y, crop_x + 704, crop_y + 256)
            )
            # Upstream PIL -> mmcv.imnormalize(to_rgb=True) reverses the RGB array.
            array = np.asarray(image, dtype=np.float32)[:, :, ::-1]
            normalized = ((array - self.mean) / self.std).transpose(2, 0, 1)
        return (
            normalized,
            (np.float32(resize), np.float32(crop_x), np.float32(crop_y)),
            np.asarray(record["calibration"]["camera_intrinsic"], dtype=np.float32),
        )

    def Process(self, frame):
        start = perf_counter()
        records = [frame["cameras"][name] for name in CAMERAS]
        images, post, intrinsics = zip(*self.executor.map(self.ReadCamera, records))
        images_ready = perf_counter()
        transforms = self.camera_transforms(frame)
        image_indices = np.zeros(len(self.voxels), dtype=np.int64)
        depth_indices = np.zeros(len(self.voxels), dtype=np.int64)
        assigned = np.zeros(len(self.voxels), dtype=bool)
        for camera, (matrix, intrinsic, (resize, crop_x, crop_y)) in enumerate(
            zip(transforms, intrinsics, post)
        ):
            inv = np.linalg.inv(matrix).astype(np.float32)
            projection = intrinsic @ inv[:3, :]
            points = (projection[:, :3] @ self.voxels.T + projection[:, 3:4]).T
            depth = points[:, 2]
            candidates = np.flatnonzero((depth >= 1) & (depth < 60) & ~assigned)
            pixels = points[candidates, :2] / depth[candidates, None]
            pixels = (pixels * resize - [crop_x, crop_y]).astype(np.float32)
            indices = (pixels / 16).astype(np.int64)  # Torch .long() truncates towards zero.
            valid = (
                (indices[:, 0] >= 0)
                & (indices[:, 0] < 44)
                & (indices[:, 1] >= 0)
                & (indices[:, 1] < 16)
            )
            locations = camera * 16 * 44 + indices[valid, 1] * 44 + indices[valid, 0]
            visible = candidates[valid]
            image_indices[visible] = locations
            depth_indices[visible] = locations * 59 + depth[visible].astype(np.int64) - 1
            assigned[visible] = True
        result = {
            "images": np.ascontiguousarray(np.stack(images)),
            "coors_img": image_indices,
            "coors_depth": depth_indices,
            "camera_to_keyego": transforms,
            "visible_voxels": int(assigned.sum()),
        }
        self.timings = {
            "camera_preparation": (images_ready - start) * 1000,
            "projection": (perf_counter() - images_ready) * 1000,
        }
        return result

    def Close(self):
        self.executor.shutdown(wait=True)


def view_transform(features, image_indices, depth_indices):
    flat = features.transpose(0, 2, 3, 1).reshape(-1, 123).copy()
    flat[0] = 0
    # Match FastrayTransformer.forward; the unmodified TRT export omitted this sigmoid.
    logits = flat[:, :59].reshape(-1)
    depth = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    pixels = image_indices.reshape(-1, 7)
    depths = depth_indices.reshape(-1, 7)
    total = np.zeros((128 * 128, 64), dtype=np.float32)
    # Accumulate in height order without materializing the 29 MB voxel feature volume.
    for height in range(7):
        total += flat[:, 59:][pixels[:, height]] * depth[depths[:, height], None]
    return np.ascontiguousarray(total.reshape(1, 128, 128, 64).transpose(0, 3, 2, 1))
