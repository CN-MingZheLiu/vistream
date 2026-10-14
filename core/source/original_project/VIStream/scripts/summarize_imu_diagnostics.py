#!/usr/bin/env python3
"""Print compact GT-free diagnostics for the raw-IMU KITTI run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


root = Path("/home/mingzheliu/VIStream/outputs/training_free_kitti_raw_imu_stride6_20260813")
for sequence_dir in sorted(root.glob("[0-9][0-9]")):
    path = sequence_dir / "candidate_before_gt.diagnostics.json"
    diagnostics = json.loads(path.read_text())
    if not diagnostics.get("accepted"):
        print(sequence_dir.name, diagnostics.get("fallback_reason"))
        continue
    innovation = np.asarray(diagnostics["innovation_rad"])
    unreliability = np.asarray(diagnostics["visual_temporal_unreliability"])
    bias = np.asarray(diagnostics["estimated_gyro_bias_rate"])
    print(
        sequence_dir.name,
        "gate_mean", round(diagnostics["gate_mean"], 6),
        "active", round(diagnostics["gate_active_fraction"], 6),
        "innovation_q", np.round(np.quantile(innovation, [0.5, 0.9, 0.99]), 6).tolist(),
        "unreliability_q", np.round(np.quantile(unreliability, [0.5, 0.9, 0.99]), 6).tolist(),
        "bias_norm_q", np.round(
            np.quantile(np.linalg.norm(bias, axis=1), [0.5, 0.9, 0.99]), 6
        ).tolist(),
        "step_change", round(diagnostics["relative_step_change"], 6),
        "path_change", round(diagnostics["relative_path_change"], 6),
    )
