"""Normalization helpers shared by WaveDiff modules."""

import torch.nn as nn


def resolve_norm_type(use_batchnorm=True, norm_type=None):
    """Map legacy batchnorm flags to an explicit normalization mode."""
    if norm_type is None:
        return "batch" if use_batchnorm else "none"
    norm_type = norm_type.lower()
    if norm_type not in {"batch", "group", "none"}:
        raise ValueError(
            f"norm_type must be 'batch', 'group', or 'none', got {norm_type!r}"
        )
    return norm_type


def compatible_group_count(channels, requested_groups=8):
    """Return the largest requested-or-smaller group count dividing channels."""
    groups = min(max(int(requested_groups), 1), channels)
    while channels % groups != 0:
        groups -= 1
    return groups


def make_norm(channels, use_batchnorm=True, norm_type=None, norm_groups=8):
    """Construct BatchNorm, GroupNorm, or identity with legacy compatibility."""
    resolved = resolve_norm_type(use_batchnorm, norm_type)
    if resolved == "batch":
        return nn.BatchNorm2d(channels)
    if resolved == "group":
        return nn.GroupNorm(
            compatible_group_count(channels, norm_groups),
            channels,
        )
    return nn.Identity()
