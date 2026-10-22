# Geometry/model conventions adapted from FastBEV++ (Apache-2.0); see THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read nuScenes camera keyframes and calibration without the training framework."""

import json

import numpy as np
from common import CAMERAS, CLASSES


def transform(rotation, translation):
    w, x, y, z = np.asarray(rotation, dtype=np.float64)
    norm = w * w + x * x + y * y + z * z
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(
        [
            [
                1 - 2 * (y * y + z * z) / norm,
                2 * (x * y - z * w) / norm,
                2 * (x * z + y * w) / norm,
            ],
            [
                2 * (x * y + z * w) / norm,
                1 - 2 * (x * x + z * z) / norm,
                2 * (y * z - x * w) / norm,
            ],
            [
                2 * (x * z - y * w) / norm,
                2 * (y * z + x * w) / norm,
                1 - 2 * (x * x + y * y) / norm,
            ],
        ]
    )
    matrix[:3, 3] = translation
    return matrix


class Dataset:
    def __init__(self, directory):
        self.directory = directory
        self.tables = {}
        for name in (
            "scene",
            "sample",
            "sample_data",
            "calibrated_sensor",
            "sensor",
            "ego_pose",
            "sample_annotation",
            "instance",
            "category",
        ):
            rows = json.loads((directory / "v1.0-mini" / f"{name}.json").read_text())
            self.tables[name] = {row["token"]: row for row in rows}
        self.scene_names = {row["name"]: row for row in self.tables["scene"].values()}
        self.annotations = {}
        for row in self.tables["sample_annotation"].values():
            self.annotations.setdefault(row["sample_token"], []).append(row["token"])
        self.cameras = {}
        for row in self.tables["sample_data"].values():
            calibration = self.tables["calibrated_sensor"][row["calibrated_sensor_token"]]
            sensor = self.tables["sensor"][calibration["sensor_token"]]
            if row["is_key_frame"] and sensor["channel"] in CAMERAS:
                self.cameras.setdefault(row["sample_token"], {})[sensor["channel"]] = {
                    "file": row["filename"],
                    "timestamp": row["timestamp"],
                    "calibration": calibration,
                    "pose": self.tables["ego_pose"][row["ego_pose_token"]],
                }

    def frames(self, scene, start=0, count=None):
        if scene not in self.scene_names:
            raise ValueError(
                f"Unknown scene {scene}; available: {', '.join(sorted(self.scene_names))}"
            )
        result = []
        token = self.scene_names[scene]["first_sample_token"]
        while token:
            sample = self.tables["sample"][token]
            cameras = self.cameras[token]
            if set(cameras) != set(CAMERAS):
                raise ValueError(f"Sample {token} does not have all six cameras")
            result.append(
                {
                    "token": token,
                    "scene": scene,
                    "timestamp": sample["timestamp"],
                    "cameras": cameras,
                    "annotations": self.annotations.get(token, []),
                }
            )
            token = sample["next"]
        return result[start:] if count is None else result[start : start + count]

    def calibration_frames(self, count):
        # Official mini_train scenes; demo scene-0103 is held out in mini_val.
        names = (
            "scene-0061",
            "scene-0553",
            "scene-0655",
            "scene-0757",
            "scene-0796",
            "scene-1077",
            "scene-1094",
            "scene-1100",
        )
        available = [frame for name in names for frame in self.frames(name)]
        if not 1 <= count <= len(available):
            raise ValueError(f"Choose 1..{len(available)} distinct rig frames for calibration")
        return [available[i] for i in np.linspace(0, len(available) - 1, count, dtype=int)]

    def ground_truth(self, frame):
        pose = frame["cameras"][CAMERAS[0]]["pose"]
        key_from_world = np.linalg.inv(transform(pose["rotation"], pose["translation"]))
        boxes = []
        for token in frame["annotations"]:
            annotation = self.tables["sample_annotation"][token]
            instance = self.tables["instance"][annotation["instance_token"]]
            category = self.tables["category"][instance["category_token"]]["name"]
            name = category.split(".")[-1]
            if category.startswith("human.pedestrian."):
                name = "pedestrian"
            elif category.startswith("vehicle.bus."):
                name = "bus"
            elif name == "trafficcone":
                name = "traffic_cone"
            elif name == "construction":
                name = "construction_vehicle"
            if name not in CLASSES:
                continue
            box_pose = key_from_world @ transform(annotation["rotation"], annotation["translation"])
            w, length, h = annotation["size"]
            boxes.append(
                {
                    "class": name,
                    "center": box_pose[:3, 3].tolist(),
                    "dimensions": [length, w, h],
                    "yaw": float(np.arctan2(box_pose[1, 0], box_pose[0, 0])),
                }
            )
        return boxes
