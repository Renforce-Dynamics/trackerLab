import torch
import numpy as np
from typing import Optional

def _interpolate(
    a: torch.Tensor,
    *,
    b: Optional[torch.Tensor] = None,
    blend: Optional[torch.Tensor] = None,
    start: Optional[np.ndarray] = None,
    end: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Linear interpolation between consecutive values.

    Args:
        a: The first value. Shape is (N, X) or (N, M, X).
        b: The second value. Shape is (N, X) or (N, M, X).
        blend: Interpolation coefficient between 0 (a) and 1 (b).
        start: Indexes to fetch the first value. If both, ``start`` and ``end` are specified,
            the first and second values will be fetches from the argument ``a`` (dimension 0).
        end: Indexes to fetch the second value. If both, ``start`` and ``end` are specified,
            the first and second values will be fetches from the argument ``a`` (dimension 0).

    Returns:
        Interpolated values. Shape is (N, X) or (N, M, X).
    """
    if start is not None and end is not None:
        return _interpolate(a=a[start], b=a[end], blend=blend)
    if a.ndim >= 2:
        blend = blend.unsqueeze(-1)
    if a.ndim >= 3:
        blend = blend.unsqueeze(-1)
    return (1.0 - blend) * a + blend * b

def _slerp(
    q0: torch.Tensor,
    *,
    q1: Optional[torch.Tensor] = None,
    blend: Optional[torch.Tensor] = None,
    start: Optional[np.ndarray] = None,
    end: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Interpolation between consecutive rotations (Spherical Linear Interpolation).

    Args:
        q0: The first quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
        q1: The second quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
        blend: Interpolation coefficient between 0 (q0) and 1 (q1).
        start: Indexes to fetch the first quaternion. If both, ``start`` and ``end` are specified,
            the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).
        end: Indexes to fetch the second quaternion. If both, ``start`` and ``end` are specified,
            the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).

    Returns:
        Interpolated quaternions. Shape is (N, 4) or (N, M, 4).
    """
    if start is not None and end is not None:
        return _slerp(q0=q0[start], q1=q0[end], blend=blend)
    if q0.ndim >= 2:
        blend = blend.unsqueeze(-1)
    if q0.ndim >= 3:
        blend = blend.unsqueeze(-1)

    qw, qx, qy, qz = 0, 1, 2, 3  # wxyz
    cos_half_theta = (
        q0[..., qw] * q1[..., qw]
        + q0[..., qx] * q1[..., qx]
        + q0[..., qy] * q1[..., qy]
        + q0[..., qz] * q1[..., qz]
    )

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
    ratio_b = torch.sin(blend * half_theta) / sin_half_theta

    new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
    new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
    new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
    new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

    new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=len(new_q_w.shape) - 1)
    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
    return new_q
