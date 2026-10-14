"""Small SO(3)/SE(3) helpers used by the non-learning VIO backend."""

from __future__ import annotations

import numpy as np


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(vector: np.ndarray) -> np.ndarray:
    """Exponential map from a rotation vector to a rotation matrix."""
    vector = np.asarray(vector, dtype=np.float64)
    theta = float(np.linalg.norm(vector))
    k = skew(vector)
    if theta < 1e-8:
        return np.eye(3) + k + 0.5 * k @ k
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * k + b * (k @ k)


def so3_log(rotation: np.ndarray) -> np.ndarray:
    """Logarithm map with a stable small-angle and near-pi branch."""
    rotation = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cosine))
    vee = np.array([rotation[2, 1] - rotation[1, 2],
                    rotation[0, 2] - rotation[2, 0],
                    rotation[1, 0] - rotation[0, 1]])
    if theta < 1e-7:
        return 0.5 * vee
    if np.pi - theta < 1e-5:
        # The skew part loses precision at pi. Recover the axis from R+I.
        diagonal = np.diag(rotation)
        axis_index = int(np.argmax(diagonal))
        axis = rotation[:, axis_index] + np.eye(3)[:, axis_index]
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = axis / norm
        return theta * axis
    return theta / (2.0 * np.sin(theta)) * vee


def project_rotation(rotation: np.ndarray) -> np.ndarray:
    """Project a numerically perturbed matrix onto SO(3)."""
    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = project_rotation(rotation)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ transform[:3, 3]
    return result
