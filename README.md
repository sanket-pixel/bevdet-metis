# bevdet-metis

Six-camera, camera-only 3D object detection with [FastBEV++](https://github.com/ymlab/advanced-fastbev), starting from its published ONNX model and targeting Axelera Metis.

This is a small Python deployment project, separate from the upstream training framework. It reads synchronized nuScenes camera samples and calibration, predicts oriented 3D boxes, and renders the six camera views alongside a bird's-eye view. Boxes include class, confidence, center, dimensions, yaw and planar velocity.

The first baseline uses ONNX Runtime on the CPU. The Metis path separates the shared image CNN, BEV encoder and detection head from the geometry operations. There is no tracking, temporal fusion or C++ port yet.

![Six-camera ONNX demonstration](assets/demo.jpg)

## Quick start

On this workspace, the existing uv environment and Voyager SDK are reused:

```bash
cd /home/sanket-shah/Desktop/Projects/axelera/bevdet-metis
make setup
make models
make data
make demo
make video
```

The default interpreter is `../deeplsd-metis/.venv/bin/python`. Only Python inference dependencies are added there; the training framework is not installed. On another machine, point `PYTHON` to an existing environment, or create one:

```bash
uv venv --python 3.11 .venv
make setup PYTHON=.venv/bin/python
make models data demo PYTHON=.venv/bin/python
```

System dependencies are Git, uv and FFmpeg with libx264. `make setup` also fetches the pinned upstream source for the independent geometry verification. PyTorch is optional for that comparison and is already available in the shared environment; ordinary ONNX inference needs no PyTorch, MMCV or CUDA.

## Model and data

`make models` downloads the [official ONNX export](https://drive.google.com/file/d/1qefFlah6PkKPtz0Zgh9aqn0rqITjU2C8/view), checks its SHA-256, and prepares the CPU and Metis partitions. Despite its `fastbev_fp16.onnx` filename, this particular export has FP32 weights, inputs and outputs. Its shared ResNet-50 image encoder processes six `3×256×704` camera images; the BEV grid is `128×128` over approximately `102.4×102.4 m`.

The [original `epoch_20_ema.pth` checkpoint](https://drive.google.com/file/d/1q49_7aR7EWWOCAUK8iE5jfj1JdtL9gJy/view) is available too:

```bash
make checkpoint
```

It is optional; inference and compilation start directly from ONNX. Asset URLs, upstream revision and generated model checksums are recorded in [weights/onnx/provenance.json](weights/onnx/provenance.json).

`make data` downloads the smallest official packaged release, [nuScenes v1.0-mini](https://www.nuscenes.org/nuscenes), verifies the archive, and retains only camera keyframes and metadata:

| Item | Size / count |
| --- | --- |
| Official archive downloaded | 4.17 GB |
| Retained RGB images and JSON | About 449 MB |
| Scenes / six-camera samples / images | 10 / 404 / 2,424 |
| Default demo | `scene-0103`, official `mini_val` |

The archive is removed after successful extraction. LiDAR, radar, maps and sweeps are not retained or required. Calibration and camera poses come from nuScenes metadata; six arbitrary JPEGs without calibration are not a substitute for a rig sample. nuScenes data and derived demo imagery are under CC BY-NC-SA 4.0; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

### Export correction

The pinned upstream `FastrayTransformer.forward()` applies a sigmoid to depth logits before lifting image features into BEV. `FastBEVTRT.forward()` and the published ONNX omit that sigmoid. The original export is preserved byte-for-byte as `fastbev_fp16.onnx`; the default `fastbev.onnx` restores this one operation without changing weights. The host view transform uses the same sigmoid.

`BACKEND=upstream` runs the unmodified export for comparison. This project does not claim a reproduced nuScenes mAP/NDS result. The correction and the INT8 deployment still need a full accuracy evaluation.

## Commands

| Command | Result |
| --- | --- |
| `make build` | Alias for downloading and preparing the ONNX models |
| `make demo` | One rendered sample, `results/onnx/demo.png` |
| `make video` | 20 samples / 10 seconds, `results/onnx/video.mp4` |
| `make profile` | Timings without rendering or encoding |
| `make verify` | Full corrected ONNX vs partitions on three real samples |
| `make verify BACKEND=aipu` | Run Metis on the same samples and report INT8/FP32 differences |
| `make calibration` | Cache 32 `mini_train` rig samples / 192 camera images |
| `make compile` | Quantize and compile the CNN stages using Voyager |

```bash
make video FRAMES=30 START=5
make demo BACKEND=split
make profile BACKEND=onnx FRAMES=20
make calibration CALIB_FRAMES=64
make compile PART=image
make demo BACKEND=aipu
make video BACKEND=aipu
```

Use `CONFIG=path/to/config.yaml` to select a config, `OUTPUT_DIR=results/my_run` for another output directory, and `SDK=/path/to/voyager-sdk` for another SDK installation. Paths inside YAML are relative to the YAML file. `configs/default.yaml` sets model/data directories, scene, CPU threads, score threshold, rotated NMS threshold and maximum detections. Model geometry, normalization and tensor quantization are model contracts, rather than user tuning controls.

Each run saves a JSONL file containing frame tokens, predicted boxes, ground truth and timings, plus a JSON timing summary. Profile runs use `profile.json` / `profile.jsonl` so they do not overwrite video results. Box centers are in the ego frame at the first camera's timestamp: **x forward, y left, z up**, dimensions **length, width, height**, metres and radians.

`make video` plays at nuScenes' native **2 Hz annotated keyframe rate**. Playback rate is independent of processing throughput. Profiling includes six JPEG reads, image preprocessing, projection-index construction, model execution, geometry and box decoding; it excludes model loading, visualization, JSON export and video encoding. Averages include the first processed frame.

## Metis execution

The camera backbone is shared, not six separate trained networks. The current implementation calls one image model six times, then lifts and combines its features. It executes synchronously to establish a correct baseline before introducing concurrent camera scheduling or frame pipelining.

```text
6 camera images
  → image CNN ×6                         [Metis]
  → depth sigmoid, indexed lift, sum z   [host]
  → BEV backbone                        [Metis]
  → exact bilinear resize + concat      [host ONNX Runtime]
  → BEV neck                            [Metis]
  → exact bilinear resize               [host ONNX Runtime]
  → detection head                      [Metis]
  → CenterPoint decode + rotated NMS    [host]
```

Although the learned stages are CNNs, the complete exported graph also contains camera-geometry gathers and two `align_corners` bilinear resizes. SDK 1.7 does not support that resize convention on the AIPU, so these operations stay on the host rather than being replaced with an approximation.

Each compiled CNN requests one Metis core. A shared AxRuntime context owns four model instances, each with its own device connection reserving one core. Input/output shape, padding, scales and zero points are validated against the compiled runtime metadata; quantization parameters are not maintained in YAML. The backbone packs its two feature scales into one terminal output and the host reverses that packing before the exact resize. SDK 1.7 exposes a second physical copy of the backbone input; its ABI binding is explicit and checked before use.

Calibration defaults to 32 uniformly selected, distinct samples from the official eight-scene `mini_train` partition. The default demo scene is held out. Image calibration sees 192 real images; downstream stages see 32 reference feature tensors. Caches and compiled artifacts retain source/model/calibration checksums and compiler settings. This is a bring-up calibration set, not a substitute for a representative deployment dataset.

### Verified locally

Validated on Ubuntu 24.04, Ryzen 7 3700X, Voyager compiler 1.7.0 and AxRuntime 1.7.0, using connected Metis hardware. The two Python environments are deliberate: ONNX/data preparation uses the existing uv environment; compilation and AxRuntime use the SDK's own environment without installing packages into it.

- Three real rig samples pass full corrected ONNX versus partitioned ONNX comparison (`rtol=5e-5`, `atol=2e-5`); the largest absolute head difference is `1.15e-5`.
- All 114,688 image/depth projection indices match the unmodified upstream PyTorch implementation on the checked sample. The BEV maximum absolute difference is `1.91e-6`.
- All four compiled CNN stages run in one Python process on four reserved Metis cores. Both CPU and AIPU paths have generated 20-frame, 1080p demo videos and detection JSONL files.

The initial 20-frame profiles, including the first frame, are:

| Stage | FP32 ONNX CPU | INT8 Metis + host |
| --- | ---: | ---: |
| Preprocessing, including JPEG reads | 150.2 ms | 156.0 ms |
| Inference, including host geometry/resize | 734.1 ms | 275.9 ms |
| Box decoding and NMS | 3.6 ms | 4.3 ms |
| Complete pipeline | **887.9 ms** | **436.2 ms** |
| Processing throughput | 1.13 rig frames/s | 2.29 rig frames/s |

These are synchronous Python bring-up measurements, not optimized deployment results. INT8 changes detections: the first demo sample gives 33 FP32 boxes and 24 INT8 boxes at the same 0.4 threshold. Three-frame numeric comparisons are saved in `results/aipu_validation.json`; rotation and velocity outputs show appreciable quantization differences. Full detection accuracy, improved calibration/PTQ, concurrent camera scheduling and removal of host overhead remain follow-up work.

## Layout

```text
configs/              Runtime configuration
python/               Preprocessing, ONNX/AxRuntime, geometry, decoding, rendering
weights/onnx/         Published model, corrected model, CNN partitions, provenance
weights/aipu/         Compiled per-stage runtime bundles
data/nuscenes/        Camera keyframes and calibration/annotation metadata
data/calibration/     Reproducible real-image / BEV calibration cache
results/              Videos, images, JSONL detections, profiles and validation reports
third_party/          Pinned upstream source, fetched by setup
```

Downloaded data, model binaries, compiler artifacts and results are ignored by Git. The C++ deployment can follow once the Python ONNX and INT8 baselines have been evaluated.

## References and licensing

- [FastBEV++ source and model zoo](https://github.com/ymlab/advanced-fastbev), pinned at `04c2b3610f7c33ebc323723fe90998121e22cfac`.
- [FastBEV++ paper](https://arxiv.org/abs/2512.08237).
- [nuScenes](https://www.nuscenes.org/nuscenes), Caesar et al., CVPR 2020.

Project code is AGPL-3.0-or-later, consistent with the existing deployment projects. Upstream Apache-2.0 notices are retained in [licenses/fastbev_apache_2.0.txt](licenses/fastbev_apache_2.0.txt); data/model/SDK terms are described separately in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
