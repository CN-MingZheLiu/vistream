"""Differentiable SO(3)/SE(3) operations without an external Lie library.

Twists use ``[rho_x, rho_y, rho_z, phi_x, phi_y, phi_z]``.  Transforms map
points from a local frame into its parent frame.
"""

from __future__ import annotations

import torch


def skew(vector: torch.Tensor) -> torch.Tensor:
    if vector.shape[-1] != 3:
        raise ValueError(f"expected [...,3], got {tuple(vector.shape)}")
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(vector.shape[:-1] + (3, 3))


def _series_coefficients(theta2: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Stable Rodrigues and left-Jacobian coefficients."""
    # Keep the unselected exact branch differentiable at theta=0.  A raw
    # sqrt(0) has an infinite derivative, and torch.where would otherwise
    # turn the inactive branch into 0 * inf = NaN during backward.
    theta = torch.sqrt(torch.clamp(theta2, min=1e-12))
    small = theta2 < 1e-8
    safe_theta = torch.where(small, torch.ones_like(theta), theta)
    safe_theta2 = safe_theta.square()
    a_exact = torch.sin(safe_theta) / safe_theta
    b_exact = (1.0 - torch.cos(safe_theta)) / safe_theta2
    c_exact = (safe_theta - torch.sin(safe_theta)) / (safe_theta2 * safe_theta)
    a_series = 1.0 - theta2 / 6.0 + theta2.square() / 120.0
    b_series = 0.5 - theta2 / 24.0 + theta2.square() / 720.0
    c_series = 1.0 / 6.0 - theta2 / 120.0 + theta2.square() / 5040.0
    return (
        torch.where(small, a_series, a_exact),
        torch.where(small, b_series, b_exact),
        torch.where(small, c_series, c_exact),
    )


def so3_exp(phi: torch.Tensor) -> torch.Tensor:
    theta2 = phi.square().sum(dim=-1, keepdim=True)
    a, b, _ = _series_coefficients(theta2)
    omega = skew(phi)
    eye = torch.eye(3, device=phi.device, dtype=phi.dtype).expand(omega.shape)
    return eye + a.unsqueeze(-1) * omega + b.unsqueeze(-1) * (omega @ omega)


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"expected [...,3,3], got {tuple(rotation.shape)}")
    vee = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    # The former acos(cos(theta)) implementation has an infinite derivative
    # at an exact identity rotation.  That is common for the anchored first
    # frame and turns a zero loss into NaN gradients.  Use the skew magnitude
    # as sin(theta) and a Taylor branch with a detached positive floor.
    trace = torch.diagonal(rotation, dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    # ``vector_norm(0)`` itself has an undefined backward in PyTorch.  The
    # clamped squared norm gives the same value away from zero and a finite
    # zero derivative at the identity.
    sin_half = 0.5 * torch.sqrt(vee.square().sum(dim=-1).clamp_min(1e-12))
    theta = torch.atan2(sin_half, cos_theta)
    theta2 = sin_half.square()
    safe_sin = sin_half.clamp_min(1e-4)
    exact = theta / (2.0 * safe_sin)
    series = 0.5 + theta2 / 12.0 + theta2.square() / 120.0
    scale = torch.where(theta2 < 1e-8, series, exact)
    return vee * scale.unsqueeze(-1)


def so3_left_jacobian(phi: torch.Tensor) -> torch.Tensor:
    theta2 = phi.square().sum(dim=-1, keepdim=True)
    _, b, c = _series_coefficients(theta2)
    omega = skew(phi)
    eye = torch.eye(3, device=phi.device, dtype=phi.dtype).expand(omega.shape)
    return eye + b.unsqueeze(-1) * omega + c.unsqueeze(-1) * (omega @ omega)


def so3_left_jacobian_inverse(phi: torch.Tensor) -> torch.Tensor:
    theta2 = phi.square().sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1e-12))
    small = theta2 < 1e-8
    safe_theta = torch.where(small, torch.ones_like(theta), theta)
    half = 0.5 * safe_theta
    coefficient = (1.0 - half / torch.tan(half)) / safe_theta.square()
    coefficient = torch.where(
        small,
        1.0 / 12.0 + theta2 / 720.0 + theta2.square() / 30240.0,
        coefficient,
    )
    omega = skew(phi)
    eye = torch.eye(3, device=phi.device, dtype=phi.dtype).expand(omega.shape)
    return eye - 0.5 * omega + coefficient.unsqueeze(-1) * (omega @ omega)


def se3_exp(twist: torch.Tensor) -> torch.Tensor:
    if twist.shape[-1] != 6:
        raise ValueError(f"expected [...,6], got {tuple(twist.shape)}")
    rho, phi = twist.split(3, dim=-1)
    rotation = so3_exp(phi)
    translation = (so3_left_jacobian(phi) @ rho.unsqueeze(-1)).squeeze(-1)
    transform = torch.zeros(
        twist.shape[:-1] + (4, 4), device=twist.device, dtype=twist.dtype
    )
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = translation
    transform[..., 3, 3] = 1.0
    return transform


def se3_log(transform: torch.Tensor) -> torch.Tensor:
    if transform.shape[-2:] != (4, 4):
        raise ValueError(f"expected [...,4,4], got {tuple(transform.shape)}")
    phi = so3_log(transform[..., :3, :3])
    rho = (
        so3_left_jacobian_inverse(phi) @ transform[..., :3, 3].unsqueeze(-1)
    ).squeeze(-1)
    return torch.cat((rho, phi), dim=-1)


def compose_se3(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left @ right


def invert_se3(transform: torch.Tensor) -> torch.Tensor:
    rotation_t = transform[..., :3, :3].transpose(-1, -2)
    translation = -(rotation_t @ transform[..., :3, 3].unsqueeze(-1)).squeeze(-1)
    result = torch.zeros_like(transform)
    result[..., :3, :3] = rotation_t
    result[..., :3, 3] = translation
    result[..., 3, 3] = 1.0
    return result


def quaternion_xyzw_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def matrix_to_quaternion_xyzw(rotation: torch.Tensor) -> torch.Tensor:
    """Convert a rotation matrix using a branch-free, normalized formula."""
    diagonal = torch.diagonal(rotation, dim1=-2, dim2=-1)
    qx = torch.sqrt(torch.clamp(1 + diagonal[..., 0] - diagonal[..., 1] - diagonal[..., 2], min=0)) / 2
    qy = torch.sqrt(torch.clamp(1 - diagonal[..., 0] + diagonal[..., 1] - diagonal[..., 2], min=0)) / 2
    qz = torch.sqrt(torch.clamp(1 - diagonal[..., 0] - diagonal[..., 1] + diagonal[..., 2], min=0)) / 2
    qw = torch.sqrt(torch.clamp(1 + diagonal.sum(dim=-1), min=0)) / 2
    qx = torch.copysign(qx, rotation[..., 2, 1] - rotation[..., 1, 2])
    qy = torch.copysign(qy, rotation[..., 0, 2] - rotation[..., 2, 0])
    qz = torch.copysign(qz, rotation[..., 1, 0] - rotation[..., 0, 1])
    quaternion = torch.stack((qx, qy, qz, qw), dim=-1)
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
