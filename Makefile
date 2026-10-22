SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --no-print-directory
.DEFAULT_GOAL := demo
PYTHON ?= ../deeplsd-metis/.venv/bin/python
SDK ?= ../voyager-sdk
SDK_PYTHON ?= $(SDK)/venv/bin/python
CONFIG ?= configs/default.yaml
BACKEND ?= cpp
FRAMES ?= 20
START ?= 0
CALIB_FRAMES ?= 32
PART ?= all
OUTPUT_DIR ?= results/$(BACKEND)
BUILD_DIR ?= build
JOBS ?= 4
AXRUNTIME_ROOT ?= /opt/axelera

.PHONY: help setup data models checkpoint build demo video profile calibration compile verify compare

help:
	@printf '%s\n' \
	  'make setup        Add dependencies to the existing uv environment' \
	  'make build        Build the modular C++ AxRuntime application' \
	  'make models       Download ONNX + prepare image/BEV CNN partitions' \
	  'make checkpoint   Download the optional original PyTorch checkpoint' \
	  'make data         Download nuScenes mini; retain camera keyframes + metadata' \
	  'make demo         Save a six-camera + BEV demo image' \
	  'make video        Save a 20-frame demo video at native 2 Hz' \
	  'make profile      Profile the selected backend without rendering' \
	  'make verify       Validate the selected backend (default: C++ vs Python)' \
	  'make compare      Compare C++ tensors, detections and timing to Python' \
	  'make calibration  Cache 32 mini_train rig frames (192 camera images)' \
	  'make compile      Compile all four CNN stages; PART=image selects one' \
	  'Options: BACKEND=cpp|onnx|split|upstream|aipu FRAMES=20 START=0 CONFIG=configs/default.yaml'

setup:
	PYTHON="$(PYTHON)" ./setup.sh

models:
	"$(PYTHON)" python/download.py model
	"$(PYTHON)" python/model.py

build:
	cmake -S cpp -B "$(BUILD_DIR)" -DCMAKE_BUILD_TYPE=Release -DAXRUNTIME_ROOT="$(AXRUNTIME_ROOT)"
	cmake --build "$(BUILD_DIR)" -j "$(JOBS)"

checkpoint:
	"$(PYTHON)" python/download.py checkpoint

data:
	"$(PYTHON)" python/download.py data

demo video profile: $(if $(filter cpp,$(BACKEND)),build)
	@mkdir -p "$(OUTPUT_DIR)"
ifeq ($(BACKEND),cpp)
	"$(BUILD_DIR)/bevdet_metis" --config "$(CONFIG)" \
	  --frames $(if $(filter demo,$@),1,$(FRAMES)) --start "$(START)" \
	  --output "$(OUTPUT_DIR)/$(if $(filter demo,$@),demo,$(if $(filter profile,$@),profile,video)).jsonl" $(if $(filter profile,$@),--profile)
	@if [ "$@" != "profile" ]; then \
	  "$(PYTHON)" python/render.py --config "$(CONFIG)" \
	  --detections "$(OUTPUT_DIR)/$(if $(filter demo,$@),demo,video).jsonl" \
	  --output "$(OUTPUT_DIR)/$(if $(filter demo,$@),demo.png,video.mp4)"; \
	fi
else
ifeq ($(BACKEND),aipu)
	source "$(SDK)/venv/bin/activate" && "$(SDK_PYTHON)" python/app.py --config "$(CONFIG)" --backend aipu \
	  --frames $(if $(filter demo,$@),1,$(FRAMES)) --start "$(START)" \
	  --output "$(OUTPUT_DIR)/$(if $(filter demo,$@),demo.png,$(if $(filter profile,$@),profile.json,video.mp4))" $(if $(filter profile,$@),--profile)
else
	"$(PYTHON)" python/app.py --config "$(CONFIG)" --backend "$(BACKEND)" \
	  --frames $(if $(filter demo,$@),1,$(FRAMES)) --start "$(START)" \
	  --output "$(OUTPUT_DIR)/$(if $(filter demo,$@),demo.png,$(if $(filter profile,$@),profile.json,video.mp4))" $(if $(filter profile,$@),--profile)
endif
endif

calibration:
	"$(PYTHON)" python/calibrate.py --config "$(CONFIG)" --frames "$(CALIB_FRAMES)"

compile: calibration
	source "$(SDK)/venv/bin/activate" && "$(SDK_PYTHON)" python/compile.py --config "$(CONFIG)" --part "$(PART)"

compare: build
	source "$(SDK)/venv/bin/activate" && "$(SDK_PYTHON)" python/compare_cpp.py \
	  --config "$(CONFIG)" --binary "$(BUILD_DIR)/bevdet_metis" --frames "$(FRAMES)" --start "$(START)" --output-directory "$(OUTPUT_DIR)"

verify: $(if $(filter cpp,$(BACKEND)),build)
ifeq ($(BACKEND),cpp)
	source "$(SDK)/venv/bin/activate" && "$(SDK_PYTHON)" python/compare_cpp.py \
	  --config "$(CONFIG)" --binary "$(BUILD_DIR)/bevdet_metis" --frames "$(FRAMES)" --start "$(START)" --output-directory "$(OUTPUT_DIR)"
else
ifeq ($(BACKEND),aipu)
	source "$(SDK)/venv/bin/activate" && "$(SDK_PYTHON)" python/validate.py --config "$(CONFIG)" --backend aipu
else
	"$(PYTHON)" python/validate.py --config "$(CONFIG)"
endif
endif
