"""Small NumPy SO(3) helpers for the training-free backend."""

from __future__ import annotations

import numpy as np


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def so3_exp(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    matrix = skew(vector)
    if angle < 1e-8:
        return np.eye(3) + matrix + 0.5 * matrix @ matrix
    return (
        np.eye(3)
        + np.sin(angle) / angle * matrix
        + (1.0 - np.cos(angle)) / angle**2 * (matrix @ matrix)
    )


def so3_log(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    vee = np.array(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        )
    )
    if angle < 1e-7:
        return 0.5 * vee
    if np.pi - angle < 1e-5:
        diagonal = np.diag(rotation)
        axis_index = int(np.argmax(diagonal))
        axis = rotation[:, axis_index] + np.eye(3)[:, axis_index]
        norm = float(np.linalg.norm(axis))
        axis = np.array((1.0, 0.0, 0.0)) if norm < 1e-8 else axis / norm
        return angle * axis
    return angle / (2.0 * np.sin(angle)) * vee


def project_rotation(rotation: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    result = left @ right
    if np.linalg.det(result) < 0:
        left[:, -1] *= -1.0
        result = left @ right
    return result


def rotation_angle(rotation: np.ndarray) -> float:
    return float(np.linalg.norm(so3_log(rotation)))
