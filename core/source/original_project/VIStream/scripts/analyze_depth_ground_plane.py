#!/usr/bin/env python3
"""Fit per-frame ground planes in frozen Lingbot depth artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def fit_ground(points: np.ndarray, rng: np.random.Generator) -> tuple[float, float, np.ndarray] | None:
    if len(points) < 100:
        return None
    best_count = 0
    best = None
    threshold = max(0.02, 0.015 * float(np.median(points[:, 2])))
    for _ in range(256):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal /= norm
        if normal[1] < 0:
            normal = -normal
        # In camera coordinates y points down, so road normals are y-dominant.
        if normal[1] < 0.75:
            continue
        offset = -float(normal @ sample[0])
        residual = np.abs(points @ normal + offset)
        inliers = residual < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best = normal, offset, inliers
    if best is None or best_count < 80:
        return None
    normal, offset, inliers = best
    selected = points[inliers]
    center = selected.mean(axis=0)
    _, _, vt = np.linalg.svd(selected - center, full_matrices=False)
    normal = vt[-1]
    if normal[1] < 0:
        normal = -normal
    offset = -float(normal @ center)
    distance = abs(offset)
    residual = np.abs(selected @ normal + offset)
    return distance, float(np.median(residual)), normal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    with np.load(args.artifact, allow_pickle=False) as archive:
        depth = archive["depth"].astype(np.float64)
        confidence = archive["depth_confidence"].astype(np.float64)
        intrinsics = archive["intrinsics"].astype(np.float64)
        pixel_stride = int(archive["depth_pixel_stride"])
        frame_indices = archive["frame_indices"]
    rng = np.random.default_rng(20260813)
    rows = []
    height, width = depth.shape[1:]
    vv, uu = np.meshgrid(
        np.arange(height) * pixel_stride,
        np.arange(width) * pixel_stride,
        indexing="ij",
    )
    for index, (values, conf, calibration) in enumerate(
        zip(depth, confidence, intrinsics)
    ):
        camera = np.stack(
            (
                (uu - calibration[0, 2]) * values / calibration[0, 0],
                (vv - calibration[1, 2]) * values / calibration[1, 1],
                values,
            ),
            axis=-1,
        )
        mask = (
            np.isfinite(camera).all(axis=-1)
            & np.isfinite(conf)
            & (conf >= 1.5)
            & (values > 0.1)
            & (values < 80.0)
            & (vv >= 0.55 * height * pixel_stride)
            & (np.abs(camera[..., 0]) < 0.8 * camera[..., 2])
        )
        result = fit_ground(camera[mask], rng)
        if result is None:
            rows.append({"frame": int(frame_indices[index]), "valid": False})
            continue
        distance, residual, normal = result
        rows.append({
            "frame": int(frame_indices[index]),
            "valid": True,
            "distance": distance,
            "residual": residual,
            "normal": normal.tolist(),
            "points": int(mask.sum()),
        })
    distances = np.asarray([row["distance"] for row in rows if row["valid"]])
    summary = {
        "frames": len(rows),
        "valid_frames": len(distances),
        "valid_fraction": len(distances) / max(len(rows), 1),
        "distance_quantiles": (
            np.quantile(distances, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist()
            if len(distances) else []
        ),
        "relative_mad": (
            float(1.4826 * np.median(np.abs(distances - np.median(distances))) / np.median(distances))
            if len(distances) else None
        ),
        "rows": rows,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
