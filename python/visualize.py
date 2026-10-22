# SPDX-License-Identifier: AGPL-3.0-or-later
"""Six camera projections and an ego-frame bird's-eye view of predicted boxes."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from common import CLASSES
from postprocessing import corners_bev

BACKGROUND = (248, 248, 245)
FOREGROUND = (37, 41, 43)
SECONDARY = (111, 117, 118)
RULE = (210, 213, 210)
GRID = (228, 230, 227)
AXIS = (184, 190, 186)
OUTLINE = (30, 35, 38)
CLASS_COLORS = dict(zip(CLASSES, (
    (0, 114, 178),      # car — blue
    (230, 159, 0),      # truck — amber
    (0, 143, 102),      # construction vehicle — green
    (213, 73, 54),      # bus — vermilion
    (190, 91, 151),     # trailer — rose
    (0, 139, 190),      # barrier — cyan blue
    (117, 82, 160),     # motorcycle — violet
    (116, 139, 0),      # bicycle — olive
    (194, 45, 112),     # pedestrian — magenta
    (192, 126, 0),      # traffic cone — ochre
)))
FONT = Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf")
BOLD = Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf")


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
    def __init__(self, directory, processing_fps):
        self.directory = directory
        self.processing_fps = processing_fps
        self.fonts = {
            size: ImageFont.truetype(str(FONT), size) for size in (14, 15, 17, 20, 26, 32)
        }
        self.bold = ImageFont.truetype(str(BOLD), 26)

    def Process(self, frame, boxes, prepared, index=0, total=1):
        canvas = Image.new("RGB", (1920, 1080), BACKGROUND)
        draw = ImageDraw.Draw(canvas)

        draw.text((64, 26), "Multi-camera 3D detection", font=self.bold, fill=FOREGROUND)
        draw.text(
            (64, 60), "nuScenes mini · FastBEV++ · Axelera Metis",
            font=self.fonts[17], fill=SECONDARY,
        )
        draw.text((1410, 27), f"{self.processing_fps:.1f} FPS", font=self.bold, fill=FOREGROUND)
        draw.text((1412, 60), "mean processing rate", font=self.fonts[17], fill=SECONDARY)
        draw.text((1640, 27), f"{len(boxes)} detections", font=self.bold, fill=FOREGROUND)
        for index, class_name in enumerate(CLASSES):
            column, row = index % 5, index // 5
            x, y = 500 + column * 178, 17 + row * 34
            color = CLASS_COLORS[class_name]
            draw.line((x, y + 9, x + 14, y + 9), fill=color, width=4)
            label = class_name.replace("_", " ")
            draw.text((x + 21, y), label, font=self.fonts[14], fill=FOREGROUND)
        draw.line((64, 94, 1856, 94), fill=RULE, width=1)
        draw.line((64, 94, 108, 94), fill=(0, 114, 178), width=3)

        edge_pairs = (
            (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
            (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7),
        )
        camera_names = (
            "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
        )
        image_width, image_height = 448, 252
        for camera, name in enumerate(camera_names):
            record = frame["cameras"][name]
            with Image.open(self.directory / record["file"]) as source:
                original = source.size
                panel = source.convert("RGB").resize(
                    (image_width, image_height), Image.Resampling.LANCZOS
                )
            panel_draw = ImageDraw.Draw(panel)
            camera_from_key = np.linalg.inv(prepared["camera_to_keyego"][camera])
            intrinsic = np.asarray(record["calibration"]["camera_intrinsic"])
            for box in boxes:
                points = corners_3d(box) @ camera_from_key[:3, :3].T + camera_from_key[:3, 3]
                projected = points @ intrinsic.T
                depth = projected[:, 2]
                xy = projected[:, :2] / np.maximum(depth[:, None], 0.01)
                xy *= [image_width / original[0], image_height / original[1]]
                for first, second in edge_pairs:
                    if depth[first] > 0.1 and depth[second] > 0.1:
                        edge = [tuple(xy[first]), tuple(xy[second])]
                        color = CLASS_COLORS[box["class"]]
                        panel_draw.line(edge, fill=OUTLINE, width=5)
                        panel_draw.line(edge, fill=color, width=2)

            column, row = camera % 2, camera // 2
            left, top = 64 + column * 464, 112 + row * 290
            label = name.removeprefix("CAM_").replace("_", " ").lower()
            draw.text((left, top), label, font=self.fonts[15], fill=SECONDARY)
            canvas.paste(panel, (left, top + 22))
            draw.rectangle(
                (left, top + 22, left + image_width - 1, top + 21 + image_height),
                outline=RULE,
                width=1,
            )

        map_left, map_top, map_size = 1008, 112, 848
        draw.text((map_left, map_top), "Bird's-eye view · ego frame", font=self.fonts[15], fill=SECONDARY)
        map_top += 22
        bev_panel = Image.new("RGB", (map_size, map_size), (252, 252, 250))
        bev_draw = ImageDraw.Draw(bev_panel)
        scale = map_size / 102.4

        def project(point):
            x, y = point[:2]
            return map_size / 2 - y * scale, map_size / 2 - x * scale

        for metres in range(-50, 51, 10):
            color = AXIS if metres == 0 else GRID
            bev_draw.line(
                (project((metres, -51.2)), project((metres, 51.2))), fill=color, width=1
            )
            bev_draw.line(
                (project((-51.2, metres)), project((51.2, metres))), fill=color, width=1
            )
        for box in boxes:
            polygon = corners_bev([*box["center"], *box["dimensions"], box["yaw"]])
            points = [project(point) for point in polygon]
            color = CLASS_COLORS[box["class"]]
            bev_draw.line(points + [points[0]], fill=color, width=3)
            nose = np.asarray(box["center"][:2]) + np.asarray(
                [np.cos(box["yaw"]), np.sin(box["yaw"])]
            ) * box["dimensions"][0] / 2
            bev_draw.line((project(box["center"]), project(nose)), fill=color, width=2)
        bev_draw.polygon(
            (
                project((2.2, 1)), project((2.2, -1)),
                project((-2.2, -1)), project((-2.2, 1)),
            ),
            fill=FOREGROUND,
        )
        canvas.paste(bev_panel, (map_left, map_top))

        footer_y = 1020
        draw.line((64, footer_y, 1856, footer_y), fill=RULE, width=1)
        draw.text(
            (64, footer_y + 14),
            f"{frame['scene']}   ·   frame {index + 1:03d} / {total:03d}",
            font=self.fonts[17], fill=SECONDARY,
        )
        draw.text(
            (1480, footer_y + 14), "forward ↑   ·   left ←   ·   10 m grid",
            font=self.fonts[17], fill=SECONDARY,
        )
        return canvas
