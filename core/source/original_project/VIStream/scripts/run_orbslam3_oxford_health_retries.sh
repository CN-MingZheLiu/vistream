#!/usr/bin/env bash
set -euo pipefail

project_root=/home/mingzheliu/VIStream
orb_root="$project_root/third_party/ORB_SLAM3"
binary="$orb_root/Examples/Monocular-Inertial/mono_inertial_tum_vi"
vocabulary="$orb_root/Vocabulary/ORBvoc.txt"
input_root="$project_root/outputs/orbslam3_oxford_suite_inputs_v3_exploratory"
run_root="$project_root/outputs/orbslam3_oxford_suite_runs_v3_exploratory"
retry_root="$project_root/outputs/orbslam3_oxford_suite_retries_v3_exploratory"
minimum_completeness="${MINIMUM_COMPLETENESS:-0.95}"
maximum_parallel_jobs="${MAXIMUM_PARALLEL_JOBS:-3}"

mkdir -p "$retry_root"

run_retry() {
    local slug="$1"
    local input="$input_root/$slug"
    local output="$retry_root/$slug/replica_2"
    local name="${slug//-/_}_v3_r2"
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

for manifest in "$input_root"/*/manifest_before_gt.json; do
    [[ -f "$manifest" ]] || continue
    input="$(dirname "$manifest")"
    slug="$(basename "$input")"
    trajectory="$(find "$run_root/$slug" -maxdepth 1 -name 'f_*.txt' -print -quit)"
    if [[ -z "$trajectory" ]]; then
        echo "base run not ready: $slug" >&2
        continue
    fi
    frames="$(wc -l < "$trajectory")"
    total="$(wc -l < "$input/timestamps.txt")"
    retry="$(( $(awk -v frames="$frames" -v total="$total" -v threshold="$minimum_completeness" 'BEGIN { print frames / total < threshold }') ))"
    if (( retry )); then
        echo "retry $slug: $frames/$total" >&2
        wait_for_slot
        run_retry "$slug" &
    fi
done
wait
