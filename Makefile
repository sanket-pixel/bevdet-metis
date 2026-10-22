SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --no-print-directory
.DEFAULT_GOAL := demo
PYTHON ?= ../deeplsd-metis/.venv/bin/python
SDK ?= ../voyager-sdk
SDK_PYTHON ?= $(SDK)/venv/bin/python
CONFIG ?= configs/default.yaml
FRAMES ?= 80
FPS ?= 2
START ?= 0
CALIB_FRAMES ?= 32
PART ?= all
OUTPUT_DIR ?= results/cpp
SHOWCASE_DIR ?= results/optimized/final_30s
BUILD_DIR ?= build
JOBS ?= 4
NATIVE ?= ON
AXRUNTIME_ROOT ?= /opt/axelera

.PHONY: help setup data models checkpoint build demo video showcase profile calibration compile

help:
	@printf '%s\n' \
	  'make setup        Add dependencies to the existing uv environment' \
	  'make build        Build the modular C++ AxRuntime application' \
	  'make models       Download checkpoint + export the ONNX models' \
	  'make checkpoint   Download the official PyTorch weights' \
	  'make data         Download nuScenes mini; retain camera keyframes + metadata' \
	  'make demo         Save a six-camera + BEV demo image' \
	  'make video        Save an 80-frame demo video (FPS=2; override with FPS=10)' \
	  'make showcase    Run 300 optimized samples; render a 30-second 10 FPS montage' \
	  'make profile      Profile the C++ application without rendering' \
	  'make calibration  Cache 32 mini_train rig frames (192 camera images)' \
	  'make compile      Compile all four CNN stages; PART=image selects one' \
	  'Options: FRAMES=80 FPS=2 START=0 CONFIG=configs/default.yaml OUTPUT_DIR=results/cpp'

setup:
	PYTHON="$(PYTHON)" ./setup.sh

models:
	$(MAKE) checkpoint
	"$(PYTHON)" python/export.py

build:
	cmake -S cpp -B "$(BUILD_DIR)" -DCMAKE_BUILD_TYPE=Release -DAXRUNTIME_ROOT="$(AXRUNTIME_ROOT)" -DBEV_NATIVE="$(NATIVE)"
	cmake --build "$(BUILD_DIR)" -j "$(JOBS)"

showcase: build
	@mkdir -p "$(SHOWCASE_DIR)"
	@set -e; for sample in scene-0061:39 scene-0103:40 scene-0553:41 scene-0655:41 scene-0757:41 scene-0796:40 scene-0916:41 scene-1077:17; do \
	  scene="$${sample%:*}"; count="$${sample#*:}"; \
	  "$(BUILD_DIR)/bevdet_metis" --config "$(CONFIG)" --scene "$$scene" --frames "$$count" \
	    --output "$(SHOWCASE_DIR)/$$scene.jsonl"; \
	done
	"$(PYTHON)" python/render.py --config "$(CONFIG)" --fps 10 --detections \
	  $(foreach scene,scene-0061 scene-0103 scene-0553 scene-0655 scene-0757 scene-0796 scene-0916 scene-1077,"$(SHOWCASE_DIR)/$(scene).jsonl") \
	  --output "$(SHOWCASE_DIR)/bevdet_metis_30s_10fps.mp4"

checkpoint:
	"$(PYTHON)" python/download.py checkpoint

data:
	"$(PYTHON)" python/download.py data

demo video profile: build
	@mkdir -p "$(OUTPUT_DIR)"
	"$(BUILD_DIR)/bevdet_metis" --config "$(CONFIG)" \
	  --frames $(if $(filter demo,$@),1,$(FRAMES)) --start "$(START)" \
	  --output "$(OUTPUT_DIR)/$(if $(filter demo,$@),demo,$(if $(filter profile,$@),profile,video)).jsonl" $(if $(filter profile,$@),--profile)
	@if [ "$@" != "profile" ]; then \
	  "$(PYTHON)" python/render.py --config "$(CONFIG)" \
	  --detections "$(OUTPUT_DIR)/$(if $(filter demo,$@),demo,video).jsonl" \
	  --output "$(OUTPUT_DIR)/$(if $(filter demo,$@),demo.png,video.mp4)" $(if $(filter video,$@),--fps "$(FPS)"); \
	fi

calibration:
	"$(PYTHON)" python/calibrate.py --config "$(CONFIG)" --frames "$(CALIB_FRAMES)"

compile: calibration
	source "$(SDK)/venv/bin/activate" && "$(SDK_PYTHON)" python/compile.py --config "$(CONFIG)" --part "$(PART)"
