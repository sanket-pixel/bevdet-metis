# SPDX-License-Identifier: AGPL-3.0-or-later
"""Six camera projections and an ego-frame bird's-eye view of predicted boxes."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from common import CLASSES
from postprocessing import corners_bev

BACKGROUND = (17, 20, 19)
PANEL = (23, 27, 26)
FOREGROUND = (235, 232, 220)
SECONDARY = (142, 148, 140)
RULE = (53, 60, 56)
GRID = (39, 47, 43)
AXIS = (78, 88, 79)
CLASS_COLORS = dict(zip(CLASSES, (
    (239, 182, 82),     # car — honey
    (232, 117, 83),     # truck — vermilion
    (161, 190, 91),     # construction vehicle — leaf
    (121, 163, 199),    # bus — slate blue
    (194, 133, 157),    # trailer — rose
    (104, 183, 155),    # barrier — sea glass
    (179, 153, 199),    # motorcycle — heather
    (203, 204, 107),    # bicycle — citron
    (229, 143, 119),    # pedestrian — coral
    (221, 197, 127),    # traffic cone — straw
)))
FONT = Path("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf")
BOLD = Path("/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf")
MONO = Path("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf")


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
        self.fonts = {size: ImageFont.truetype(str(FONT), size) for size in (13, 14, 16, 18, 24, 30, 38)}
        self.bold = ImageFont.truetype(str(BOLD), 24)
        self.metric = ImageFont.truetype(str(MONO), 37)

    def Process(self, frame, boxes, prepared, index=0, total=1):
        canvas = Image.new("RGB", (1920, 1080), BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, 1919, 100), fill=PANEL)
        draw.text((64, 18), "FASTBEV++", font=self.bold, fill=FOREGROUND)
        draw.text((64, 53), "SIX-CAMERA  /  3D PERCEPTION", font=self.fonts[13], fill=SECONDARY)
        draw.text((335, 18), "nuScenes mini   ·   Axelera Metis", font=self.fonts[16], fill=SECONDARY)
        # A compact key keeps the class colors legible without competing with the views.
        for class_index, class_name in enumerate(CLASSES):
            column, row = class_index % 5, class_index // 5
            x, y = 635 + column * 174, 14 + row * 32
            color = CLASS_COLORS[class_name]
            draw.rounded_rectangle((x, y + 4, x + 14, y + 18), radius=4, fill=color)
            draw.text((x + 21, y + 2), class_name.replace("_", " "), font=self.fonts[13], fill=FOREGROUND)
        draw.text((1512, 21), f"{self.processing_fps:.1f} FPS", font=self.metric, fill=FOREGROUND)
        draw.text((1515, 61), f"{len(boxes):03d} objects", font=self.fonts[16], fill=SECONDARY)
        draw.line((64, 100, 1856, 100), fill=RULE, width=1)
        draw.line((64, 100, 145, 100), fill=CLASS_COLORS["car"], width=2)

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
                        panel_draw.line(edge, fill=color, width=2)

            column, row = camera % 2, camera // 2
            left, top = 64 + column * 464, 119 + row * 290
            label = name.removeprefix("CAM_").replace("_", " ")
            draw.text((left, top), f"{camera + 1:02d}", font=ImageFont.truetype(str(MONO), 14), fill=CLASS_COLORS["car"])
            draw.text((left + 34, top), label, font=self.fonts[14], fill=FOREGROUND)
            image_top = top + 24
            draw.rounded_rectangle(
                (left - 2, image_top - 2, left + image_width + 1, image_top + image_height + 1),
                radius=3,
                fill=PANEL,
                outline=RULE,
                width=1,
            )
            canvas.paste(panel, (left, image_top))

        map_left, map_top, map_size = 1008, 119, 848
        draw.text((map_left, map_top), "EGO-FRAME MAP", font=self.fonts[14], fill=FOREGROUND)
        draw.text((map_left + 164, map_top), "TOP DOWN   ·   80 m EXTENT", font=self.fonts[13], fill=SECONDARY)
        map_top += 24
        bev_panel = Image.new("RGB", (map_size, map_size), PANEL)
        bev_draw = ImageDraw.Draw(bev_panel)
        scale = map_size / 88.0

        def project(point):
            x, y = point[:2]
            return map_size / 2 - y * scale, map_size / 2 - x * scale

        for metres in range(-40, 41, 10):
            color = AXIS if metres == 0 else GRID
            bev_draw.line(
                (project((metres, -44)), project((metres, 44))), fill=color, width=1
            )
            bev_draw.line(
                (project((-44, metres)), project((44, metres))), fill=color, width=1
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
            fill=(216, 211, 193),
        )
        # Registration marks and scale bars give the map a quiet technical frame.
        for x, y, dx, dy in ((14, 14, 1, 1), (map_size - 15, 14, -1, 1),
                             (14, map_size - 15, 1, -1), (map_size - 15, map_size - 15, -1, -1)):
            bev_draw.line((x, y, x + dx * 18, y), fill=AXIS, width=1)
            bev_draw.line((x, y, x, y + dy * 18), fill=AXIS, width=1)
        bar_y = map_size - 28
        bar_right = map_size - 42
        bar_left = bar_right - round(10 * scale)
        bev_draw.text((bar_left + 22, bar_y - 22), "10 m", font=self.fonts[13], fill=SECONDARY)
        bev_draw.line((bar_left, bar_y, bar_right, bar_y), fill=FOREGROUND, width=2)
        bev_draw.line((bar_left, bar_y - 4, bar_left, bar_y + 4), fill=FOREGROUND, width=1)
        bev_draw.line((bar_right, bar_y - 4, bar_right, bar_y + 4), fill=FOREGROUND, width=1)
        canvas.paste(bev_panel, (map_left, map_top))

        footer_y = 1023
        draw.line((64, footer_y, 1856, footer_y), fill=RULE, width=1)
        draw.text(
            (64, footer_y + 14),
            f"{frame['scene']}    /    {index + 1:03d} : {total:03d}",
            font=ImageFont.truetype(str(MONO), 14), fill=SECONDARY,
        )
        draw.text(
            (1665, footer_y + 13), "FORWARD    /    10 m",
            font=self.fonts[14], fill=SECONDARY,
        )
        return canvas
