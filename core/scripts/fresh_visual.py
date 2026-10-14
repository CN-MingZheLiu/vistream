#!/usr/bin/env python3
"""Generate KITTI visual outputs using packaged model functions, without GT."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

BUNDLE = Path('/bundle')
SOURCE = BUNDLE / 'source/original_project/lingbot-map'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sequence', required=True)
    args = ap.parse_args()
    output = Path('/work/visual') / args.sequence
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(SOURCE))
    spec = importlib.util.spec_from_file_location('packaged_visual', SOURCE / 'evaluate_kitti_odometry.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    assert not Path('/sensors/poses').exists()
    assert not (BUNDLE / 'evidence').exists()
    paths, timestamps, indices = evaluator.sampled_inputs(Path('/sensors'), args.sequence, 6)
    kfi = math.ceil(len(paths) / 320)
    model_args = SimpleNamespace(inference_mode='streaming', image_size=518, patch_size=14,
        max_frame_num=1024, kv_cache_sliding_window=64, num_scale_frames=8,
        use_sdpa=True, camera_num_iterations=4,
        model_path=BUNDLE / 'runtime/weights/lingbot-map-long.pt')
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.time()
    # Same exact preprocessing/model/inference functions as the release.
    images = evaluator.load_and_preprocess_images([str(p) for p in paths], mode='crop', image_size=518, patch_size=14)
    model = evaluator.build_model(model_args, device, torch.bfloat16)
    prediction, seconds, peak = evaluator.infer(model, images, model_args, device, torch.bfloat16, kfi)
    legacy, intrinsics = evaluator.predictions_to_poses(prediction['pose_enc'], tuple(images.shape[-2:]), 'benchmark_c2w')
    corrected, _ = evaluator.predictions_to_poses(prediction['pose_enc'], tuple(images.shape[-2:]), 'opencv_w2c')
    assert len(legacy) == len(indices) and np.isfinite(legacy).all()
    # Explicit historical interpretation for reproducing the historical table;
    # preserve the physically corrected convention too, without substituting it.
    target = output / 'trajectory.npz'
    np.savez_compressed(target, predicted_c2w=legacy, opencv_c2w=corrected,
                        intrinsics=intrinsics, timestamps=timestamps, frame_indices=indices)
    inputs = [{'frame_index': int(i), 'image': str(p), 'sha256': sha(p)} for i, p in zip(indices, paths)]
    (output / 'input_images.json').write_text(json.dumps(inputs, indent=2))
    report = {'status': 'PASS', 'sequence': args.sequence, 'new_inference': True,
              'GT_opened': False, 'archived_trajectories_accessible': False,
              'frames': len(indices), 'first_frame': int(indices[0]), 'last_frame': int(indices[-1]),
              'keyframe_interval': kfi, 'frame_stride': 6, 'image_size': 518,
              'dtype': 'bfloat16', 'use_sdpa': True, 'checkpoint_sha256': sha(model_args.model_path),
              'evaluator_sha256': sha(SOURCE / 'evaluate_kitti_odometry.py'),
              'output_sha256': sha(target), 'inference_seconds': seconds,
              'elapsed_seconds': time.time() - started, 'peak_gpu_gib': peak,
              'historical_pose_interpretation': 'benchmark_c2w (legacy direct-decoder interpretation)',
              'corrected_pose_key': 'opencv_c2w'}
    (output / 'generation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
