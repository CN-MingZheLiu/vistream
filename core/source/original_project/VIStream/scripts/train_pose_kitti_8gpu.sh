#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/pose_kitti_8gpu_20260810}"
PYTHON_BIN="${VISTREAM_PYTHON:-/home/mingzheliu/.conda/envs/lingbot-map/bin/python}"
CONFIG="${ROOT_DIR}/configs/vistream_pose_kitti.yaml"

export VISTREAM_PYTHON="${PYTHON_BIN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "${OUTPUT_ROOT}"
cd "${ROOT_DIR}"

run_stage() {
    local stage="$1"
    shift
    echo "[$(date --iso-8601=seconds)] starting ${stage}" | tee -a "${OUTPUT_ROOT}/status.log"
    ./run_vistream.sh -m torch.distributed.run --standalone --nproc_per_node=8 \
        scripts/train.py --config "${CONFIG}" --stage "${stage}" "$@" \
        2>&1 | tee "${OUTPUT_ROOT}/${stage}.log"
    test -f "${OUTPUT_ROOT}/${stage}/adapter_best.pt"
    echo "[$(date --iso-8601=seconds)] completed ${stage}" | tee -a "${OUTPUT_ROOT}/status.log"
}

run_stage rotation \
    --window-frames 32 --stride-frames 16 \
    --learning-rate 5e-5 --epochs 10 --bf16 \
    --output-dir "${OUTPUT_ROOT}/rotation"

run_stage translation \
    --window-frames 32 --stride-frames 16 \
    --learning-rate 2e-5 --epochs 10 --bf16 \
    --resume "${OUTPUT_ROOT}/rotation/adapter_best.pt" \
    --output-dir "${OUTPUT_ROOT}/translation"

run_stage joint \
    --window-frames 64 --stride-frames 32 \
    --learning-rate 1e-5 --epochs 5 --bf16 \
    --resume "${OUTPUT_ROOT}/translation/adapter_best.pt" \
    --output-dir "${OUTPUT_ROOT}/joint"

echo "[$(date --iso-8601=seconds)] curriculum completed" | tee -a "${OUTPUT_ROOT}/status.log"
