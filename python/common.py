# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared configuration and the fixed published FastBEV++ model contract."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)
CLASSES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)
HEADS = ("reg", "height", "dim", "rot", "vel", "heatmap")
@dataclass(frozen=True)
class Config:
    model_directory: Path
    aipu_directory: Path
    dataset_directory: Path
    scene: str
    threads: int
    score_threshold: float
    nms_threshold: float
    max_detections: int


def load_config(path):
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    expected = {
        "inference": {"model_directory", "compiled_model_directory", "threads"},
        "data": {"directory", "scene"},
        "postprocessing": {"score_threshold", "nms_threshold", "max_detections"},
    }
    if not isinstance(data, dict) or set(data) != set(expected):
        raise ValueError("Expected inference, data and postprocessing config sections")
    for section, keys in expected.items():
        if not isinstance(data[section], dict) or set(data[section]) != keys:
            raise ValueError(f"Unexpected or missing settings in {section}")
    inference, dataset, post = data["inference"], data["data"], data["postprocessing"]
    config = Config(
        (path.parent / inference["model_directory"]).resolve(),
        (path.parent / inference["compiled_model_directory"]).resolve(),
        (path.parent / dataset["directory"]).resolve(),
        dataset["scene"],
        int(inference["threads"]),
        float(post["score_threshold"]),
        float(post["nms_threshold"]),
        int(post["max_detections"]),
    )
    if not (
        1 <= config.threads <= 32
        and 0 < config.score_threshold < 1
        and 0 < config.nms_threshold < 1
        and 1 <= config.max_detections <= 500
    ):
        raise ValueError("Invalid thread count or detection settings")
    return config


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
