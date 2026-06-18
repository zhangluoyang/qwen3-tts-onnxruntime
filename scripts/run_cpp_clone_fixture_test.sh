#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

FIXTURE_DIR="outputs/cpp_fixture"
ONNX_DIR="onnx_fp16"

python scripts/dump_cpp_clone_fixture.py \
  --onnx-dir "${ONNX_DIR}" \
  --out-dir "${FIXTURE_DIR}" \
  --max-new-tokens 32

cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j

export LD_LIBRARY_PATH="/home/zhang/miniconda3/lib/python3.12/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/home/zhang/miniconda3/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="/home/zhang/miniconda3/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="/home/zhang/miniconda3/lib/python3.12/site-packages/nvidia/cufft/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="/home/zhang/miniconda3/lib/python3.12/site-packages/nvidia/curand/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="/home/zhang/github/Qwen3-Audio-bar/cpp/third_party/onnxruntime-local/onnxruntime-linux-x64-gpu-1.26.0/lib:${LD_LIBRARY_PATH}"

./cpp/build/qwen3tts_clone_fixture \
  --onnx-dir "${ONNX_DIR}" \
  --fixture-dir "${FIXTURE_DIR}" \
  --out-dir "${FIXTURE_DIR}"

python scripts/compare_cpp_clone_fixture.py --fixture-dir "${FIXTURE_DIR}"
