#!/usr/bin/env bash
set -euo pipefail

project_root=/home/mingzheliu/VIStream
python_bin=/home/mingzheliu/.conda/envs/lingbot-map/bin/python
lingbot_root=/home/mingzheliu/MZL_workplace/lingbot-map/outputs/oxford_all_current_long_3840
dataset_root=/data2/huangshuaixin/datasets/Oxford_Spires
orb_root="$project_root/third_party/ORB_SLAM3"
binary="$orb_root/Examples/Monocular-Inertial/mono_inertial_tum_vi"
vocabulary="$orb_root/Vocabulary/ORBvoc.txt"
input_root="$project_root/outputs/orbslam3_oxford_suite_inputs_v3_exploratory"
run_root="$project_root/outputs/orbslam3_oxford_suite_runs_v3_exploratory"
maximum_parallel_jobs="${MAXIMUM_PARALLEL_JOBS:-3}"

mkdir -p "$input_root" "$run_root"

prepare_one() {
    local cache="$1"
    local slug full_sequence output
    slug="$(basename "$(dirname "$cache")")"
    full_sequence="$($python_bin - "$cache" <<'PY'
from pathlib import Path
import sys
import numpy as np

with np.load(sys.argv[1], allow_pickle=False) as archive:
    first = Path(str(archive["image_paths"][0]))
matches = [part for part in first.parts if part.startswith("2024-")]
if len(matches) != 1:
    raise RuntimeError(f"cannot derive Oxford sequence from {first}")
print(matches[0])
PY
)"
    output="$input_root/$slug"
    if [[ -f "$output/manifest_before_gt.json" ]]; then
        return
    fi
    if [[ -e "$output" ]]; then
        mv "$output" "$output.failed.$(date +%Y%m%d%H%M%S)"
    fi
    "$python_bin" "$project_root/scripts/prepare_orbslam3_oxford.py" \
        --cache "$cache" \
        --imu "$dataset_root/sequences/$full_sequence/raw/imu.csv" \
        --camera "$dataset_root/calibration/cam0.yaml" \
        --imu-calibration "$dataset_root/calibration/imu.yaml" \
        --extrinsics "$dataset_root/calibration/cam-lidar-imu.yaml" \
        --output-dir "$output" \
        >"$output.prepare.log" 2>&1
}

run_one() {
    local cache="$1"
    local slug input output name
    slug="$(basename "$(dirname "$cache")")"
    input="$input_root/$slug"
    output="$run_root/$slug"
    name="${slug//-/_}_v3"
    if compgen -G "$output/f_*.txt" >/dev/null; then
        return
    fi
    mkdir -p "$output"
    (
        cd "$output"
        timeout 2400 "$binary" \
            "$vocabulary" \
            "$input/Oxford-Spires.yaml" \
            "$input/images" \
            "$input/timestamps.txt" \
            "$input/imu_orbslam3.csv" \
            "$name" \
            >run.log 2>&1
    )
}

wait_for_slot() {
    while (( $(jobs -pr | wc -l) >= maximum_parallel_jobs )); do
        wait -n
    done
}

mapfile -t caches < <(find "$lingbot_root" -mindepth 2 -maxdepth 2 -name trajectory.npz -print | sort)
if [[ "${#caches[@]}" -ne 10 ]]; then
    echo "expected 10 Oxford caches, found ${#caches[@]}" >&2
    exit 1
fi

for cache in "${caches[@]}"; do
    wait_for_slot
    prepare_one "$cache" &
done
wait

for cache in "${caches[@]}"; do
    wait_for_slot
    run_one "$cache" &
done
wait
