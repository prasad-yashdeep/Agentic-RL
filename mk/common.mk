# Shared make settings for Agentic-RL subprojects: `include ../mk/common.mk` (swe-bench) or `include mk/common.mk` (root).
# GNU make >= 3.82 (RECIPEPREFIX avoids tab-vs-space accidents).
.RECIPEPREFIX = >
ARL  := $(realpath $(dir $(lastword $(MAKEFILE_LIST)))..)
PY   := $(ARL)/.venv/bin/python
# Compute never runs on the headnode: suites and runs go over ssh to NODE. node03 is the CPU default; node08 is
# the GPU node (and the only compute node up on 2026-09-24, so pass NODE=node08 until node03 returns).
NODE ?= node03
ENV  := PYTHONDONTWRITEBYTECODE=1 HF_HOME=$(HOME)/hf-cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MPLBACKEND=Agg
