SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --no-print-directory
.DEFAULT_GOAL := demo
PYTHON ?= ../deeplsd-metis/.venv/bin/python
SDK ?= ../voyager-sdk
SDK_PYTHON ?= $(SDK)/venv/bin/python
CONFIG ?= configs/default.yaml
BACKEND ?= onnx
FRAMES ?= 20
START ?= 0
CALIB_FRAMES ?= 32
PART ?= all
OUTPUT_DIR ?= results/$(BACKEND)

.PHONY: help setup data models checkpoint build demo video profile calibration compile verify

help:
	@printf '%s\n' \
	  'make setup        Add dependencies to the existing uv environment' \
	  'make build        Download and prepare the Python ONNX models' \
	  'make models       Download ONNX + prepare image/BEV CNN partitions' \
	  'make checkpoint   Download the optional original PyTorch checkpoint' \
	  'make data         Download nuScenes mini; retain camera keyframes + metadata' \
	  'make demo         Save a six-camera + BEV demo image' \
	  'make video        Save a 20-frame demo video at native 2 Hz' \
	  'make profile      Profile the selected backend without rendering' \
	  'make verify       Check ONNX partition equivalence on real camera frames' \
	  'make calibration  Cache 32 mini_train rig frames (192 camera images)' \
	  'make compile      Compile all four CNN stages; PART=image selects one' \
	  'Options: BACKEND=onnx|split|upstream|aipu FRAMES=20 START=0 CONFIG=configs/default.yaml'

setup:
	PYTHON="$(PYTHON)" ./setup.sh

models:
	"$(PYTHON)" python/download.py model
	"$(PYTHON)" python/model.py

build: models

checkpoint:
	"$(PYTHON)" python/download.py checkpoint

data:
	"$(PYTHON)" python/download.py data

demo video profile:
	@mkdir -p "$(OUTPUT_DIR)"
ifeq ($(BACKEND),aipu)
	source "$(SDK)/venv/bin/activate" && "$(SDK_PYTHON)" python/app.py --config "$(CONFIG)" --backend aipu \
	  --frames $(if $(filter demo,$@),1,$(FRAMES)) --start "$(START)" \
	  --output "$(OUTPUT_DIR)/$(if $(filter demo,$@),demo.png,$(if $(filter profile,$@),profile.json,video.mp4))" $(if $(filter profile,$@),--profile)
else
	"$(PYTHON)" python/app.py --config "$(CONFIG)" --backend "$(BACKEND)" \
	  --frames $(if $(filter demo,$@),1,$(FRAMES)) --start "$(START)" \
	  --output "$(OUTPUT_DIR)/$(if $(filter demo,$@),demo.png,$(if $(filter profile,$@),profile.json,video.mp4))" $(if $(filter profile,$@),--profile)
endif

calibration:
	"$(PYTHON)" python/calibrate.py --config "$(CONFIG)" --frames "$(CALIB_FRAMES)"

compile: calibration
	source "$(SDK)/venv/bin/activate" && "$(SDK_PYTHON)" python/compile.py --config "$(CONFIG)" --part "$(PART)"

verify:
ifeq ($(BACKEND),aipu)
	source "$(SDK)/venv/bin/activate" && "$(SDK_PYTHON)" python/validate.py --config "$(CONFIG)" --backend aipu
else
	"$(PYTHON)" python/validate.py --config "$(CONFIG)"
endif
