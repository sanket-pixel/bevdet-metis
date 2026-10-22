# Geometry/model conventions adapted from FastBEV++ (Apache-2.0); see THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CenterPoint decoding and class-scaled rotated BEV NMS, following upstream."""

import cv2
import numpy as np
from common import CLASSES, NMS_SCALE


def corners_bev(box):
    x, y, _, length, width, _, yaw = box[:7]
    corners = np.array([[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5]], dtype=np.float32) * [
        length,
        width,
    ]
    c, s = np.cos(yaw), np.sin(yaw)
    return np.asarray(corners @ np.array([[c, s], [-s, c]]) + [x, y], dtype=np.float32)


def rotated_nms(boxes, scores, labels, threshold, maximum):
    scaled = boxes.copy()
    scaled[:, 3:5] *= np.asarray(NMS_SCALE)[labels, None]
    polygons = [corners_bev(box) for box in scaled]
    lower = np.array([p.min(axis=0) for p in polygons])
    upper = np.array([p.max(axis=0) for p in polygons])
    areas = scaled[:, 3] * scaled[:, 4]
    order = np.argsort(-scores, kind="stable")
    keep = []
    while len(order) and len(keep) < maximum:
        current, remaining = order[0], order[1:]
        keep.append(current)
        intersects = np.all(upper[remaining] > lower[current], axis=1) & np.all(
            lower[remaining] < upper[current], axis=1
        )
        suppressed = np.zeros(len(remaining), dtype=bool)
        for position in np.flatnonzero(intersects):
            other = remaining[position]
            intersection = max(
                0.0, cv2.intersectConvexConvex(polygons[current], polygons[other])[0]
            )
            iou = intersection / max(1e-12, areas[current] + areas[other] - intersection)
            suppressed[position] = iou > threshold
        order = remaining[~suppressed]
    return keep


class Postprocessing:
    def __init__(self, config):
        self.config = config

    def Process(self, heads):
        logits = heads["heatmap"][0]
        if logits.shape != (10, 128, 128):
            raise ValueError(f"Unexpected detection heatmap: {logits.shape}")
        heat = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
        flat = heat.reshape(-1)
        indices = np.argpartition(flat, -500)[-500:]
        indices = indices[np.argsort(-flat[indices], kind="stable")]
        scores = flat[indices]
        labels, positions = indices // (128 * 128), indices % (128 * 128)
        y, x = positions // 128, positions % 128
        values = {name: value[0, :, y, x] for name, value in heads.items() if name != "heatmap"}
        with np.errstate(over="ignore", invalid="ignore"):
            boxes = np.column_stack(
                (
                    (x + values["reg"][:, 0]) * 0.8 - 51.2,
                    (y + values["reg"][:, 1]) * 0.8 - 51.2,
                    values["height"][:, 0],
                    np.exp(values["dim"]),
                    np.arctan2(values["rot"][:, 0], values["rot"][:, 1]),
                    values["vel"],
                )
            )
        valid = (scores > self.config.score_threshold) & np.isfinite(boxes).all(axis=1)
        valid &= (boxes[:, :3] >= [-61.2, -61.2, -10]).all(axis=1) & (
            boxes[:, :3] <= [61.2, 61.2, 10]
        ).all(axis=1)
        boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
        if not len(boxes):
            return []
        keep = rotated_nms(
            boxes, scores, labels, self.config.nms_threshold, self.config.max_detections
        )
        return [
            {
                "class": CLASSES[int(labels[i])],
                "class_id": int(labels[i]),
                "score": float(scores[i]),
                "center": boxes[i, :3].tolist(),
                "dimensions": boxes[i, 3:6].tolist(),
                "yaw": float(boxes[i, 6]),
                "velocity": boxes[i, 7:9].tolist(),
            }
            for i in keep
        ]
