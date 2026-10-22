# SPDX-License-Identifier: AGPL-3.0-or-later
"""Six camera projections and an ego-frame bird's-eye view of predicted boxes."""

from pathlib import Path

import numpy as np
from common import CAMERAS, CLASSES
from PIL import Image, ImageDraw, ImageFont
from postprocessing import corners_bev

COLORS = (
    (99, 232, 218),
    (255, 193, 111),
    (183, 164, 235),
    (118, 176, 245),
    (198, 171, 122),
    (154, 177, 199),
    (238, 140, 179),
    (157, 215, 132),
    (245, 155, 118),
    (224, 219, 125),
)
BG, FG, MUTED = (12, 18, 25), (236, 243, 246), (133, 154, 168)


def font(size, bold=False):
    name = "Ubuntu-M.ttf" if bold else "Ubuntu-R.ttf"
    path = Path("/usr/share/fonts/truetype/ubuntu") / name
    return ImageFont.truetype(
        str(path) if path.exists() else ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size
    )


def corners_3d(box):
    length, width, height = box["dimensions"]
    offsets = np.array(
        [
            [x, y, z]
            for z in (-0.5, 0.5)
            for x, y in ((0.5, 0.5), (0.5, -0.5), (-0.5, -0.5), (-0.5, 0.5))
        ]
    ) * [length, width, height]
    c, s = np.cos(box["yaw"]), np.sin(box["yaw"])
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return offsets @ rotation.T + box["center"]


class Renderer:
    def __init__(self, directory, backend):
        self.directory, self.backend = directory, backend
        self.fonts = {size: font(size) for size in (12, 16, 18, 20, 22, 28, 40, 60)}

    def Process(self, frame, boxes, prepared, timings, index=0, total=1):
        canvas = Image.new("RGB", (1920, 1080), BG)
        draw = ImageDraw.Draw(canvas)
        draw.text((48, 28), "bevdet / metis", font=font(40, True), fill=FG)
        draw.text((48, 82), "SIX-CAMERA 3D DETECTION / FASTBEV++", font=self.fonts[18], fill=MUTED)
        badge = "ONNX CPU REFERENCE" if self.backend != "aipu" else "INT8 / AXELERA METIS"
        draw.text((1500, 53), badge, font=self.fonts[22], fill=COLORS[0])
        draw.line((48, 123, 1872, 123), fill=(45, 59, 70))
        draw.text((48, 145), "01 / CAMERA PROJECTIONS", font=self.fonts[20], fill=MUTED)
        draw.text((1360, 145), "02 / BIRD'S-EYE VIEW", font=self.fonts[20], fill=MUTED)
        edge_pairs = (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )
        for camera, name in enumerate(CAMERAS):
            record = frame["cameras"][name]
            with Image.open(self.directory / record["file"]) as source:
                original = source.size
                panel = source.convert("RGB").resize((416, 234), Image.Resampling.LANCZOS)
            panel_draw = ImageDraw.Draw(panel)
            camera_from_key = np.linalg.inv(prepared["camera_to_keyego"][camera])
            intrinsic = np.array(record["calibration"]["camera_intrinsic"])
            for box in boxes:
                color = COLORS[box["class_id"]]
                points = corners_3d(box) @ camera_from_key[:3, :3].T + camera_from_key[:3, 3]
                projected = points @ intrinsic.T
                depth = projected[:, 2]
                xy = projected[:, :2] / np.maximum(depth[:, None], 0.01)
                xy *= [416 / original[0], 234 / original[1]]
                for a, b in edge_pairs:
                    if depth[a] > 0.1 and depth[b] > 0.1:
                        panel_draw.line([tuple(xy[a]), tuple(xy[b])], fill=color, width=2)
            left, top = 48 + (camera % 3) * 432, 211 + (camera // 3) * 286
            draw.text(
                (left, top - 27),
                name.replace("CAM_", "").replace("_", " "),
                font=self.fonts[16],
                fill=MUTED,
            )
            canvas.paste(panel, (left, top))
        bev_panel = Image.new("RGB", (512, 512), (17, 27, 36))
        bev_draw = ImageDraw.Draw(bev_panel)
        scale = 5.0

        def project(point):
            x, y = point[:2]
            return 256 - y * scale, 256 - x * scale

        for metres in range(-50, 51, 10):
            bev_draw.line((project((metres, -51.2)), project((metres, 51.2))), fill=(30, 42, 53))
            bev_draw.line((project((-51.2, metres)), project((51.2, metres))), fill=(30, 42, 53))
        for box in boxes:
            color = COLORS[box["class_id"]]
            polygon = corners_bev([*box["center"], *box["dimensions"], box["yaw"]])
            xy = [project(p) for p in polygon]
            bev_draw.line(xy + [xy[0]], fill=color, width=2)
            nose = (
                np.array(box["center"][:2])
                + np.array([np.cos(box["yaw"]), np.sin(box["yaw"])]) * box["dimensions"][0] / 2
            )
            bev_draw.line((project(box["center"]), project(nose)), fill=color, width=2)
        bev_draw.polygon(
            (project((2.2, 1)), project((2.2, -1)), project((-2.2, -1)), project((-2.2, 1))),
            fill=FG,
        )
        bev_draw.text((13, 484), "x forward / y left / 10 m grid", font=self.fonts[16], fill=MUTED)
        canvas.paste(bev_panel, (1360, 190))
        draw.text((48, 817), "CLASSES", font=self.fonts[18], fill=MUTED)
        for i, name in enumerate(CLASSES):
            x, y = 48 + (i % 5) * 260, 858 + (i // 5) * 40
            draw.rectangle((x, y + 3, x + 9, y + 12), fill=COLORS[i])
            label = "construction" if name == "construction_vehicle" else name.replace("_", " ")
            draw.text((x + 20, y), label, font=self.fonts[18], fill=FG)
        draw.text((1360, 749), str(len(boxes)), font=self.fonts[60], fill=FG)
        draw.text((1480, 785), "3D detections", font=self.fonts[22], fill=MUTED)
        draw.text(
            (1360, 857),
            f"{timings['pipeline']:.0f} ms / frame",
            font=self.fonts[28],
            fill=COLORS[0],
        )
        draw.text(
            (1360, 906),
            "CPU reference / rendering excluded"
            if self.backend != "aipu"
            else "Host + AIPU / rendering excluded",
            font=self.fonts[18],
            fill=MUTED,
        )
        draw.text(
            (48, 975),
            f"nuScenes mini / {frame['scene']} / frame {index + 1:03d} of {total:03d}",
            font=self.fonts[20],
            fill=MUTED,
        )
        draw.line((48, 1019, 1872, 1019), fill=(39, 54, 65), width=2)
        draw.line((48, 1019, 48 + 1824 * (index + 1) / total, 1019), fill=COLORS[0], width=2)
        draw.text(
            (48, 1034),
            "nuScenes / Motional / CC BY-NC-SA 4.0 / camera-only input",
            font=self.fonts[18],
            fill=MUTED,
        )
        draw.text((1450, 1034), "FastBEV++ ONNX baseline", font=self.fonts[18], fill=MUTED)
        return canvas
