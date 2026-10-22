# bevdet-metis

Six-camera, camera-only 3D object detection with [FastBEV++](https://github.com/ymlab/advanced-fastbev), exported from its published PyTorch checkpoint and deployed on Axelera Metis.

This is a modular C++ AxRuntime deployment with a Python ONNX/AxRuntime reference, separate from the upstream training framework. It reads nuScenes camera samples and calibration, predicts oriented 3D boxes, and renders the six camera views alongside a bird's-eye view. Boxes include class, confidence, center, dimensions, yaw and planar velocity.

The C++ implementation fuses normalization, quantization, layout and padding, and keeps features in native INT8 runtime buffers between CNN stages. Image preparation, geometry and feature resizing use parallel CPU work and SIMD. Inference runs entirely in C++ using AxRuntime; Python provides optional presentation and reference comparisons.

The validated unfused implementation is frozen on `main` and tagged [`baseline-cpp-v1`](https://github.com/sanket-pixel/bevdet-metis/tree/baseline-cpp-v1). This branch, `optimize-host`, contains the host optimizations. Models, calibration, thresholds and core allocation are unchanged.

![Six-camera C++ Metis demonstration](assets/demo.jpg)

## Quick start

On this workspace, the existing uv environment and Voyager SDK are reused:

```bash
cd /home/sanket-shah/Desktop/Projects/axelera/bevdet-metis
make setup
make models
make data
make build
make compile                  # Quantize and compile the exported CNN stages
make demo
make video
make compare
```

The default interpreter is `../deeplsd-metis/.venv/bin/python`. Only Python inference dependencies are added there; the training framework is not installed. On another machine, point `PYTHON` to an existing environment, or create one:

```bash
uv venv --python 3.11 .venv
make setup PYTHON=.venv/bin/python
make models data PYTHON=.venv/bin/python
make demo BACKEND=onnx PYTHON=.venv/bin/python
```

The C++ build requires a C++17 compiler, CMake, pkg-config, OpenCV, yaml-cpp, nlohmann-json, OpenSSL, OpenBLAS and an installed AxRuntime. On Ubuntu, the development packages are:

```bash
sudo apt install build-essential cmake pkg-config libopencv-dev libyaml-cpp-dev \
  nlohmann-json3-dev libssl-dev libopenblas-dev
```

Git, uv and FFmpeg with libx264 are needed for setup and presentation. `make setup` installs CPU PyTorch, ONNX Runtime and the small Python dependencies. The model and export code are local; setup does not clone the training repository or install MMCV, MMDetection or CUDA. `BACKEND=onnx` runs the Python CPU reference without building C++ or requiring Metis hardware.

## Model and data

`make models` downloads the [official `epoch_20_ema.pth` checkpoint](https://drive.google.com/file/d/1q49_7aR7EWWOCAUK8iE5jfj1JdtL9gJy/view), checks its SHA-256, and exports the full ONNX model and the CPU/Metis partitions from the standalone PyTorch architecture in `python/model.py`. The shared ResNet-50 image encoder processes six `3×256×704` camera images; the BEV grid is `128×128` over approximately `102.4×102.4 m`.

To download the checkpoint without exporting models:

```bash
make checkpoint
```

The model implementation uses plain PyTorch modules and loads the checkpoint strictly. Export details, source checksum and generated ONNX checksums are recorded in `weights/onnx/provenance.json`.

`make data` downloads the smallest official packaged release, [nuScenes v1.0-mini](https://www.nuscenes.org/nuscenes), verifies the archive, and retains only camera keyframes and metadata:

| Item | Size / count |
| --- | --- |
| Official archive downloaded | 4.17 GB |
| Retained RGB images and JSON | About 449 MB |
| Scenes / six-camera samples / images | 10 / 404 / 2,424 |
| Default demo | `scene-0103`, official `mini_val` |

The archive is removed after successful extraction. LiDAR, radar, maps and sweeps are not retained or required. Calibration and camera poses come from nuScenes metadata; six arbitrary JPEGs without calibration are not a substitute for a rig sample. nuScenes data and derived demo imagery are under CC BY-NC-SA 4.0; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

### PyTorch export

The model applies sigmoid to its 59 depth logits before lifting image features into BEV. This operation is present in `python/model.py` and all exported graphs. On a real frame, the generated full ONNX outputs match the prior reference export within `3e-5` absolute error.

This project does not claim a reproduced nuScenes mAP/NDS result. The INT8 deployment still needs a full accuracy evaluation.

## Commands

| Command | Result |
| --- | --- |
| `make build` | Build `build/bevdet_metis` and the reusable `bevdet_cpp` library |
| `make demo` | C++ detections + one rendered sample, `results/cpp/demo.png` |
| `make video` | C++ detections + 80 samples at 2 FPS, `results/cpp/video.mp4` |
| `make showcase` | 300 optimized detections across eight nuScenes mini scenes; clean 30-second video at 10 FPS |
| `make profile` | C++ timings without rendering or encoding |
| `make compare` | Compare C++ with Python AxRuntime on 20 samples and dump/check three tensor traces |
| `make verify` | Same C++ comparison by default |
| `make verify BACKEND=onnx` | Full ONNX vs partitions and standalone PyTorch view transform |
| `make verify BACKEND=aipu` | Run Metis on the same samples and report INT8/FP32 differences |
| `make calibration` | Cache 32 `mini_train` rig samples / 192 camera images |
| `make compile` | Quantize and compile the CNN stages using Voyager |

```bash
make video FRAMES=30 START=5 FPS=10
make showcase
make compare FRAMES=20
make demo BACKEND=split
make profile BACKEND=onnx FRAMES=20
make calibration CALIB_FRAMES=64
make compile PART=image
make demo BACKEND=aipu
make video BACKEND=aipu
```

`BACKEND=cpp` is the default. `BACKEND=aipu` selects Python AxRuntime; `onnx` runs the full exported model and `split` runs the exported partitions. Use `CONFIG=path/to/config.yaml` to select a config, `OUTPUT_DIR=results/my_run` for another output directory, and `SDK=/path/to/voyager-sdk` for another SDK installation. `BUILD_DIR` selects a CMake build directory and `AXRUNTIME_ROOT` selects the runtime installation, defaulting to `/opt/axelera`. Builds enable `-march=native` by default, including AVX2 kernels on supported x86 hosts. Use `make build NATIVE=OFF` for a portable build with scalar fallbacks; rebuild for the target CPU before copying a native binary to another machine. Floating-point contraction remains disabled and fast-math is not enabled.

Paths inside YAML are relative to the YAML file. The same `configs/default.yaml` is loaded into typed C++ configuration structs and used by Python. It sets model/data directories, scene, CPU threads, score threshold, rotated NMS threshold and maximum detections. Model geometry, normalization and tensor quantization are validated contracts rather than user tuning controls.

Each run saves JSONL frame tokens, predicted boxes and timings, plus a JSON timing summary. Python inference runs additionally include ground-truth annotations. Profile runs use `profile.json` / `profile.jsonl` so they do not overwrite video results. Box centers are in the ego frame at the first camera's timestamp: **x forward, y left, z up**, dimensions **length, width, height**, metres and radians.

To adjust the presentation without repeating inference:

```bash
../deeplsd-metis/.venv/bin/python python/render.py --config configs/default.yaml \
  --detections results/cpp/video.jsonl --output results/cpp/video.mp4
```

`make video` renders at 2 FPS by default, matching nuScenes' annotated keyframe interval. Set `FPS=10` for smoother playback; this changes playback speed, not the sampled frames or processing throughput. `make showcase` processes a fixed 300-frame selection across eight scenes and writes a scene-labelled, 30-second video at 10 FPS to `results/optimized/final_30s/bevdet_metis_30s_10fps.mp4`. The frames come from separate nuScenes mini scenes; the scene label identifies each cut. Set `SHOWCASE_DIR=path` to choose another output folder. Profiling includes six JPEG reads, image preprocessing, projection-index construction, model execution, geometry and box decoding; it excludes model loading, visualization, JSON export and video encoding. Averages include the first processed frame.

## Metis execution

The camera backbone is shared, not six separate trained networks. The current implementation calls one image model six times, then lifts and combines its features. Model calls execute synchronously; batching and concurrent frame scheduling remain future work. Within a frame, camera preparation runs in parallel while the calling thread constructs projection matrices. Projection-index selection and fused feature resizing also use OpenCV's worker pool.

```text
6 camera images
  → image CNN ×6                         [Metis]
  → depth sigmoid, indexed lift, sum z   [host]
  → BEV backbone                        [Metis]
  → exact bilinear resize + concat      [C++ host]
  → BEV neck                            [Metis]
  → exact bilinear resize               [C++ host]
  → detection head                      [Metis]
  → CenterPoint decode + rotated NMS    [host]
```

Although the learned stages are CNNs, the complete exported graph also contains camera-geometry gathers and two `align_corners` bilinear resizes. SDK 1.7 does not support that resize convention on the AIPU, so these operations stay on the host rather than being replaced with an approximation.

Each compiled CNN requests one Metis core. A shared AxRuntime context owns four model instances, each with its own device connection reserving one core. Input/output shape, padding, scales and zero points are validated against the compiled runtime metadata; quantization parameters are not maintained in YAML. The backbone packs its two feature scales into one terminal output and the host reverses that packing before the exact resize. SDK 1.7 exposes a second physical copy of the backbone input; its ABI binding is explicit and checked before use.

Calibration defaults to 32 uniformly selected, distinct samples from the official eight-scene `mini_train` partition. The default demo scene is held out. Image calibration sees 192 real images; downstream stages see 32 reference feature tensors. Caches and compiled artifacts retain source/model/calibration checksums and compiler settings. This is a bring-up calibration set, not a substitute for a representative deployment dataset.

## C++ modules

The layout follows `lines-metis`: flat headers/sources with `common` and `inference` subfolders. Classes own their state directly; there are no Python bindings or PImpl classes.

| Module | Responsibility |
| --- | --- |
| `Preprocessing` | Parallel JPEG reads, exact bicubic resize/crop, fused normalization + INT8 quantization + physical padding, camera alignment and projection indices |
| `AipuContext` | Own the shared AxRuntime context |
| `AipuInference` | Own one model/connection and fixed-size native INT8 buffers; execute directly, decode outputs when needed |
| `ViewTransform` | INT8 lookup tables, indexed lift + ordered height reduction + next-model quantization/padding, without materializing the voxel volume |
| `FeatureResize` | Fused INT8 dequantization, exact `align_corners` resize, requantization and next-model padding; AVX2 channel kernels |
| `Postprocessing` | CenterPoint box decoding, class-scaled rotated NMS |
| `Pipeline` | Own and call the stages in order, returning ego-frame detections |

`AipuTensorInfo` describes native NHWC storage and quantization. Each backend owns its fixed-size input and output buffers; native outputs are valid until that backend's next `Process()` call. Production inference avoids intermediate float NCHW tensors. Only the six small detection heads are decoded for postprocessing. Lookup tables use the validated model metadata, including independent input/output scales.

The float `Tensor` path remains available for validation. Construct `Pipeline(config, true)` to retain traces: each fused physical input is checked byte-for-byte against unfused quantization/layout/padding before model execution. The command-line `--dump-directory` enables this mode. Validation materializes reference intermediates and is intentionally excluded from performance measurements.

```cpp
const auto config = bev::LoadConfig("configs/default.yaml");
const bev::Dataset dataset(config.data.directory);
bev::Pipeline pipeline(config);
const auto frame = dataset.frames(config.data.scene, 0, 1).front();
bev::Timings timings;
const auto boxes = pipeline.Process(frame, &timings);
```

Preprocessing preserves the upstream BGR channel convention and Pillow's antialiased bicubic coefficients and integer rounding; OpenCV's cubic filter is not substituted. Projection uses OpenBLAS, matching the reference provider: tiny float32 differences near integer pixel boundaries can otherwise change a voxel's selected feature. Top-K selection preserves the SDK NumPy 1.26 reference's equal-score ordering, which matters for greedy NMS with INT8 score ties. These compatibility routines execute entirely in C++.

## Validation and performance

Validated on Ubuntu 24.04, Ryzen 7 3700X, Voyager compiler 1.7.0 and AxRuntime 1.7.0, using connected Metis hardware. The two Python environments are deliberate: ONNX/data preparation uses the existing uv environment; compilation and AxRuntime use the SDK's own environment without installing packages into it.

- Three real rig samples pass full corrected ONNX versus partitioned ONNX comparison (`rtol=5e-5`, `atol=2e-5`); the largest absolute head difference is `1.15e-5`.
- The checked frame uses 114,688 image/depth projection entries. Its NumPy host view transform and the standalone PyTorch implementation differ by at most `1.91e-6` in the resulting BEV tensor.
- All four compiled CNN stages run in either a standalone C++ process or the Python reference on four reserved Metis cores.
- On 20 `scene-0103` samples, all **775 decoded detections** match Python in class, score, center, yaw and velocity. The largest dimension difference is `4.77e-7 m`.
- Three full tensor traces have bit-identical preprocessed images, projection indices and all six raw heads. Intermediate BEV floats differ by at most `1.44e-6`, without changing the quantized inference outputs.

The repeated C++ comparison used the same YAML (`inference.threads=4`), compiled models and 20 rig samples, with three sequential baseline/optimized pairs. Every trial includes the first frame and JPEG reads; model loading, JSON/tensor export and rendering are excluded. On this host:

| Stage | Frozen unfused C++ | Optimized C++ |
| --- | ---: | ---: |
| Preprocessing, including JPEG reads | 77.0 ms | 32.2 ms |
| Inference, including host geometry/resize | 138.8 ms | 65.1 ms |
| Box decoding and NMS | 1.0 ms | 1.7 ms |
| Complete pipeline | 216.8 ms | 99.0 ms |
| Time inside AxRuntime calls | 57.9 ms | 58.5 ms |
| Total host time (pipeline minus runtime calls) | 158.8 ms | 40.5 ms |
| Processing throughput | 4.61 rig frames/s | 10.10 rig frames/s |

That is a **2.19× end-to-end speedup** and about **74% less host time**. Host feature resizing fell from approximately 23 ms to 3.1 ms and the BEV view transform from approximately 19 ms to 1.8 ms. The optimized path avoids the 29 MB voxel volume, repeated float NCHW/NHWC conversions and the 33.6 MB float head-input tensor. Exact bicubic resampling processes only source rows contributing to the retained crop. Normalization/quantization, depth sigmoid and heatmap sigmoid use finite lookup tables, preserving the original float operation order.

All 775 detections are **exactly equal to the frozen C++ outputs** in each paired trial. A separate paired Python run measured 213.3 ms versus 100.0 ms for C++; detection centers, scores, yaw and velocity match exactly, with dimension differences below `1e-6 m`. Three complete tensor traces in each of two scenes have bit-identical raw heads, images and projection indices. Every fused model input also passes physical INT8 byte equality against the unfused operations. A portable `NATIVE=OFF` build passes the same input checks and matches the native build's images, indices and heads on one sample.

The compact [measurement report](assets/host_optimization.json) records all three trials, validation scope and build settings. Absolute timings depend on host load. CPU parallelism is configurable; the default OpenCV worker count remains four, with projection construction overlapping image preparation on the calling thread. These timings use synchronous inference, without batching or cross-frame pipelining.

`make compare` writes `results/cpp/comparison.json`, `python_profile.json`, the C++ profile and three binary tensor traces. It fails if fused input bytes differ, checked images/indices/heads differ, or detection fields exceed tolerances in `python/compare_cpp.py`. This checks deployment parity rather than nuScenes mAP/NDS. The second-scene check uses three `scene-0061` samples, two CPU threads, a 0.5 threshold and paths resolved from another YAML location.

`make profile` prints main stage timings and saves the detailed breakdown in `results/cpp/profile.json`; `BACKEND=aipu` profiles Python. Inference details include the image CNN across six cameras, each BEV CNN, fused host lifting/resizing and runtime execution. Quantization now belongs mainly to preprocessing and the fused lifting/resize stages; the standalone quantization counter mostly measures the required backbone-input replica. Detail counters overlap and should not all be added together. AxRuntime execution includes transfers and waiting, rather than measuring only AIPU arithmetic.

Core allocation and scheduling are distinct: four cores are reserved, with one model stage on each, but a single frame still executes those stages in order. Its six camera CNN calls share the image core. Independent frames are not yet pipelined across the stages, so these measurements do not represent full four-core throughput.

INT8 changes detections: the first demo sample gives 33 FP32 boxes and 24 INT8 boxes at the same 0.4 threshold. Three-frame numeric comparisons are saved in `results/aipu_validation.json`; rotation and velocity outputs show appreciable quantization differences. Full detection accuracy, improved calibration/PTQ, concurrent scheduling and further host optimization remain follow-up work.

## Layout

```text
configs/              Runtime configuration
cpp/include/          Public module interfaces; common/ and inference/ subfolders
cpp/src/              Module implementations and fused host kernels
cpp/app/main.cpp      Dataset application, JSONL export and profiling
build/                CMake build and bevdet_metis executable
python/               Preprocessing, ONNX/AxRuntime, geometry, decoding, rendering
weights/torch/        Published model checkpoint
weights/onnx/         ONNX exports, CNN partitions and provenance
weights/aipu/         Compiled per-stage runtime bundles
data/nuscenes/        Camera keyframes and calibration/annotation metadata
data/calibration/     Reproducible real-image / BEV calibration cache
results/              Videos, images, JSONL detections, profiles and validation reports
```

Downloaded data, model binaries, compiler artifacts, build output and results are ignored by Git.

## References and licensing

- [FastBEV++ source and model zoo](https://github.com/ymlab/advanced-fastbev), pinned at `04c2b3610f7c33ebc323723fe90998121e22cfac`.
- [FastBEV++ paper](https://arxiv.org/abs/2512.08237).
- [nuScenes](https://www.nuscenes.org/nuscenes), Caesar et al., CVPR 2020.

Project code is AGPL-3.0-or-later, consistent with the existing deployment projects. Upstream Apache-2.0 notices are retained in [licenses/fastbev_apache_2.0.txt](licenses/fastbev_apache_2.0.txt); data/model/SDK terms are described separately in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
