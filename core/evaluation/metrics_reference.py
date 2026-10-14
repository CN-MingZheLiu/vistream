"""SE(3)/Sim(3) trajectory alignment and metric point-cloud scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Alignment:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.scale * (points @ self.rotation.T) + self.translation


def umeyama_alignment(
    predicted: np.ndarray,
    target: np.ndarray,
    with_scale: bool,
) -> Alignment:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 3:
        raise ValueError("predicted and target trajectories must both be [N,3]")
    if len(predicted) < 3:
        raise ValueError("trajectory alignment needs at least three positions")
    pred_mean = predicted.mean(axis=0)
    target_mean = target.mean(axis=0)
    pred_centered = predicted - pred_mean
    target_centered = target - target_mean
    covariance = (target_centered.T @ pred_centered) / len(predicted)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    if with_scale:
        variance = np.mean(np.sum(pred_centered ** 2, axis=1))
        scale = float(np.sum(singular * np.diag(correction)) / max(variance, 1e-12))
    else:
        scale = 1.0
    translation = target_mean - scale * (rotation @ pred_mean)
    return Alignment(scale, rotation, translation)


def ate_rmse(
    predicted: np.ndarray,
    target: np.ndarray,
    alignment: str = "se3",
) -> dict[str, float | np.ndarray]:
    if alignment not in {"se3", "sim3"}:
        raise ValueError("alignment must be 'se3' or 'sim3'")
    transform = umeyama_alignment(predicted, target, with_scale=alignment == "sim3")
    aligned = transform.apply(np.asarray(predicted))
    errors = np.linalg.norm(aligned - np.asarray(target), axis=-1)
    return {
        "ate_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "ate_mean": float(errors.mean()),
        "ate_median": float(np.median(errors)),
        "ate_max": float(errors.max()),
        "scale": transform.scale,
        "aligned": aligned,
    }


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return points.reshape(0, 3)
    cells = np.floor(points / voxel_size).astype(np.int64)
    _, unique = np.unique(cells, axis=0, return_index=True)
    return points[np.sort(unique)]


def point_cloud_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    thresholds: Iterable[float] = (0.05, 0.10),
    voxel_size: float = 0.05,
) -> dict[str, float]:
    predicted = voxel_downsample(predicted, voxel_size)
    target = voxel_downsample(target, voxel_size)
    if len(predicted) == 0 or len(target) == 0:
        raise ValueError("point-cloud metrics require two non-empty clouds")
    target_tree = cKDTree(target)
    predicted_tree = cKDTree(predicted)
    pred_to_target = target_tree.query(predicted, workers=-1)[0]
    target_to_pred = predicted_tree.query(target, workers=-1)[0]
    result: dict[str, float] = {
        "cd_l1_m": float(pred_to_target.mean() + target_to_pred.mean()),
    }
    for threshold in thresholds:
        precision = float(np.mean(pred_to_target <= threshold))
        recall = float(np.mean(target_to_pred <= threshold))
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        suffix = f"{int(round(threshold * 100)):02d}cm"
        result[f"precision_{suffix}"] = precision
        result[f"recall_{suffix}"] = recall
        result[f"f1_{suffix}"] = f1
    return result
