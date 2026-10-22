# SPDX-License-Identifier: AGPL-3.0-or-later
"""One six-camera frame through preprocessing, ONNX and box decoding."""

from time import perf_counter

from inference import OnnxInference
from postprocessing import Postprocessing
from preprocessing import Preprocessing


class Pipeline:
    def __init__(self, config, backend="onnx"):
        self.preprocessing = Preprocessing(config.dataset_directory)
        if backend == "aipu":
            from runtime import AipuInference

            self.inference = AipuInference(config)
        else:
            self.inference = OnnxInference(config, split=backend == "split")
        self.postprocessing = Postprocessing(config)

    def Process(self, frame):
        start = perf_counter()
        prepared = self.preprocessing.Process(frame)
        pre_end = perf_counter()
        heads = self.inference.Process(prepared)
        inf_end = perf_counter()
        boxes = self.postprocessing.Process(heads)
        end = perf_counter()
        timings = {
            "preprocessing": (pre_end - start) * 1000,
            "inference": (inf_end - pre_end) * 1000,
            "postprocessing": (end - inf_end) * 1000,
            "pipeline": (end - start) * 1000,
        }
        if hasattr(self.inference, "timings"):
            timings.update(
                {f"inference_{key}": value for key, value in self.inference.timings.items()}
            )
        timings.update(
            {f"preprocessing_{key}": value for key, value in self.preprocessing.timings.items()}
        )
        return boxes, prepared, timings

    def Close(self):
        try:
            if hasattr(self.inference, "Close"):
                self.inference.Close()
        finally:
            self.preprocessing.Close()
