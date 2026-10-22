# Third-party notices

## FastBEV++ / OpenMMLab

Source: https://github.com/ymlab/advanced-fastbev

Pinned revision: `04c2b3610f7c33ebc323723fe90998121e22cfac`.

The upstream repository is licensed under Apache License 2.0 and includes OpenMMLab and Phigent Robotics copyright notices. Its license is preserved in [licenses/fastbev_apache_2.0.txt](licenses/fastbev_apache_2.0.txt). The upstream source is fetched separately into the ignored `third_party/advanced_fastbev` directory; original notices in those files are retained.

The deployment's image preparation, camera projection/view transform, CenterPoint decoding and class-scaled rotated NMS follow the upstream implementation and configuration. These operations have been rewritten for standalone Python and C++ execution; source-derived modules carry an attribution notice. Geometry is independently checked against unmodified upstream method bodies when PyTorch is installed.

The published ONNX and optional PyTorch checkpoint are downloaded separately from the upstream model zoo. They are not included in Git. No additional license is granted here for model assets or their training data; users must follow the upstream distribution terms. The original ONNX is preserved. A separately generated graph restores the sigmoid present in the normal upstream model's depth path; provenance records the change and model hashes.

## nuScenes data and demonstration imagery

Source: https://www.nuscenes.org/nuscenes

Dataset: nuScenes v1.0-mini, including six camera keyframes, camera calibration, ego poses and annotations. nuScenes is made available by Motional under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/). The same data terms apply to the derived demonstration image in `assets/demo.jpg` and rendered videos. Dataset files are downloaded separately and are not included in Git.

Citation: Holger Caesar et al., “nuScenes: A multimodal dataset for autonomous driving,” CVPR 2020.

Rendered images modify the original camera photographs by adding predicted 3D boxes, labels and a BEV visualization. They include visible nuScenes/Motional attribution.

## Axelera Voyager / AxRuntime

The compiler and runtime are supplied by the separately installed Axelera Voyager SDK and remain subject to Axelera's terms. This repository does not redistribute the SDK, firmware or SDK binaries. Compilation requires a compatible installed SDK; AIPU inference requires a compatible Metis device and runtime.

## Project license

The C++ bicubic resampling in `cpp/src/preprocessing.cpp` follows Pillow 12.2.0's `src/libImaging/Resample.c`. Pillow/PIL copyright notices and the MIT-CMU license are preserved in [licenses/pillow_mit_cmu.txt](licenses/pillow_mit_cmu.txt).

The bilinear coordinate/weight conventions in `cpp/src/feature_resize.cpp` follow ONNX Runtime's CPU Upsample kernel. Copyright (c) Microsoft Corporation. Its MIT license is preserved in [licenses/onnxruntime_mit.txt](licenses/onnxruntime_mit.txt). The C++ application does not link ONNX Runtime; these resizes execute in its native C++ module.

The finite-score Top-K selection in `cpp/src/postprocessing.cpp` adapts NumPy 1.26.4's `numpy/core/src/npysort/selection.cpp`, retaining its tie ordering for comparison with the SDK Python reference. The NumPy BSD license is preserved in [licenses/numpy_bsd.txt](licenses/numpy_bsd.txt). There is no NumPy or Python dependency in the C++ binary.

Original deployment code in this repository is licensed under AGPL-3.0-or-later. The complete license text is in [LICENSE](LICENSE). This license does not replace the separate terms above.
