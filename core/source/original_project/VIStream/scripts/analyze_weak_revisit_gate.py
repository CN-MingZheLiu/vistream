#!/usr/bin/env python3
"""Summarize weak frozen-descriptor revisit evidence without ground truth."""

from __future__ import annotations

from pathlib import Path

import numpy as np


visual_root = Path(
    "/home/mingzheliu/MZL_workplace/lingbot-map/outputs/"
    "kitti_00_10_stride6_frozen"
)
descriptor_root = Path(
    "/home/mingzheliu/VIStream/outputs/"
    "training_free_loop_environment_gate_stride6_20260813"
)


def poses_4x4(values: np.ndarray) -> np.ndarray:
    if values.shape[1:] == (4, 4):
        return values
    output = np.repeat(np.eye(4)[None], len(values), axis=0)
    output[:, :3, :4] = values
    return output


for index in range(11):
    sequence = f"{index:02d}"
    with np.load(visual_root / sequence / "trajectory.npz") as archive:
        visual = poses_4x4(archive["predicted_c2w"])
    descriptors = np.load(
        descriptor_root / sequence / "resnet50_imagenet.npy"
    )
    similarity = descriptors @ descriptors.T
    src, dst = np.triu_indices(len(descriptors), k=min(40, len(descriptors) - 1))
    order = np.argsort(similarity[src, dst])[::-1]
    chosen = []
    for item in order:
        a, b = int(src[item]), int(dst[item])
        if any(abs(a - x) < 8 and abs(b - y) < 8 for x, y in chosen):
            continue
        chosen.append((a, b))
        if len(chosen) == 30:
            break
    steps = np.linalg.norm(np.diff(visual[:, :3, 3], axis=0), axis=1)
    step_scale = max(float(np.median(steps[steps > 1e-9])), 1e-9)
    values = []
    for a, b in chosen:
        displacement = float(
            np.linalg.norm(visual[b, :3, 3] - visual[a, :3, 3])
        )
        values.append((
            float(similarity[a, b]),
            b - a,
            displacement / (step_scale * np.sqrt(b - a)),
            displacement / max(float(np.sum(steps[a:b])), 1e-9),
        ))
    array = np.asarray(values)
    high = array[:, 0] >= max(float(np.quantile(array[:, 0], 0.75)), 0.85)
    selected = array[high]
    if not len(selected):
        selected = array[np.argsort(array[:, 0])[-min(4, len(array)):]]
    print(
        sequence,
        "score_q", np.round(np.quantile(array[:, 0], [0.5, 0.75, 1.0]), 4).tolist(),
        "rw_closure_q", np.round(np.quantile(selected[:, 2], [0.1, 0.5, 0.9]), 4).tolist(),
        "path_closure_q", np.round(np.quantile(selected[:, 3], [0.1, 0.5, 0.9]), 4).tolist(),
        "pairs", len(selected),
    )
