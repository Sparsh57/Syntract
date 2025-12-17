"""
Backward-compatible wrapper for ANTs transform utilities.

This module simply re-exports the updated implementations so imports that
reference ``synthesis.ants_transform`` continue to work.
"""

try:
    from .ants_transform_updated import (
        apply_ants_transform_to_mri,
        apply_ants_transform_to_streamlines,
        process_with_ants,
    )
except ImportError as exc:
    raise ImportError(
        "ants_transform_updated is required but could not be imported; "
        "ensure the module exists and dependencies are installed."
    ) from exc

__all__ = [
    "apply_ants_transform_to_mri",
    "apply_ants_transform_to_streamlines",
    "process_with_ants",
]
