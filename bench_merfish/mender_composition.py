"""MENDER-style multi-range aggregation for soft decoder compositions."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


def valid_composition(values: np.ndarray) -> np.ndarray:
    """Map unconstrained decoder predictions to soft state memberships."""
    out = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    total = out.sum(1, keepdims=True)
    np.divide(out, total, out=out, where=total > 0)
    return out


def _ring_counts(
    coords: np.ndarray,
    membership: np.ndarray,
    scale: int,
    radius: float,
) -> np.ndarray:
    """Concatenate raw soft state counts in non-overlapping distance rings."""
    tree = cKDTree(np.asarray(coords, dtype=np.float64))
    distances = tree.sparse_distance_matrix(
        tree, max_distance=scale * radius, output_type="coo_matrix"
    )
    result = []
    for ring in range(scale):
        left, right = ring * radius, (ring + 1) * radius
        take = (distances.data >= left) & (distances.data < right)
        adjacency = sp.csr_matrix(
            (
                np.ones(int(take.sum()), dtype=np.float32),
                (distances.row[take], distances.col[take]),
            ),
            shape=(len(coords), len(coords)),
        )
        result.append(np.asarray(adjacency @ membership, dtype=np.float32))
    return np.concatenate(result, axis=1)


def _adaptive_ring_counts(
    coords: np.ndarray,
    membership: np.ndarray,
    scale: int,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate physical rings using each cell's kth-neighbor radius."""
    if len(coords) < 2:
        out = np.tile(membership, (1, scale))
        return out, np.ones(len(coords)), np.zeros(len(coords), dtype=np.int32)
    k = min(int(neighbors), len(coords) - 1)
    if k < 1:
        raise ValueError("neighbors must be positive")
    coords = np.asarray(coords, dtype=np.float64)
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=k + 1)
    radii = np.asarray(distances[:, -1], dtype=np.float64)
    positive = radii[np.isfinite(radii) & (radii > 0)]
    fallback = float(np.median(positive)) if len(positive) else 1.0
    radii[~np.isfinite(radii) | (radii <= 0)] = fallback
    radii *= 1.0 + 1e-6
    first_ring = tree.query_ball_point(coords, radii, return_length=True) - 1
    neighborhoods = tree.query_ball_point(coords, scale * radii)
    result = np.zeros(
        (len(coords), scale, membership.shape[1]), dtype=np.float32
    )
    for row, columns in enumerate(neighborhoods):
        columns = np.asarray(columns, dtype=np.int64)
        distance = np.linalg.norm(coords[columns] - coords[row], axis=1)
        ring = np.floor(distance / radii[row]).astype(np.int64)
        valid = ring < scale
        columns, ring = columns[valid], ring[valid]
        for level in np.unique(ring):
            result[row, level] = membership[columns[ring == level]].sum(0)
    return (
        result.reshape(len(coords), scale * membership.shape[1]),
        radii,
        np.asarray(first_ring, dtype=np.int32),
    )


def _normalize_log_counts(out: np.ndarray) -> np.ndarray:
    """Apply MENDER's normalize-total and log1p transform to one slot."""
    out = np.asarray(out, dtype=np.float32).copy()
    total = out.sum(1, keepdims=True)
    np.divide(out * 1e4, total, out=out, where=total > 0)
    np.log1p(out, out=out)
    return out


def aggregate_mender_compositions(
    coords: np.ndarray,
    samples: np.ndarray,
    background_composition: np.ndarray,
    salient_composition: np.ndarray,
    *,
    scale: int = 6,
    radius: float = 15.0,
    radius_neighbors: int | None = None,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[dict]]:
    """Create background/salient MENDER blocks on independent sample graphs."""
    bg = valid_composition(background_composition)
    sal = valid_composition(salient_composition)
    n_bg, n_sal = bg.shape[1], sal.shape[1]
    bg_out = np.zeros((len(coords), scale * n_bg), dtype=np.float32)
    sal_out = np.zeros((len(coords), scale * n_sal), dtype=np.float32)
    diagnostics: list[dict] = []
    samples = np.asarray(samples).astype(str)
    coords = np.asarray(coords, dtype=np.float32)
    for value in np.unique(samples):
        rows = np.flatnonzero(samples == value)
        local_radii = np.repeat(float(radius), len(rows))
        if radius_neighbors is not None:
            joint_flat, local_radii, first_ring = _adaptive_ring_counts(
                coords[rows],
                np.concatenate([bg[rows], sal[rows]], axis=1),
                scale,
                radius_neighbors,
            )
            joint = joint_flat.reshape(len(rows), scale, n_bg + n_sal)
        else:
            tree = cKDTree(np.asarray(coords[rows], dtype=np.float64))
            first_ring = tree.query_ball_point(
                coords[rows], radius, return_length=True
            ) - 1
            joint = _ring_counts(
                coords[rows],
                np.concatenate([bg[rows], sal[rows]], axis=1),
                scale,
                radius,
            ).reshape(len(rows), scale, n_bg + n_sal)
        audit = {
            "sample": str(value),
            "n_cells": int(len(rows)),
            "radius_mode": (
                "cell_local_kth_neighbor"
                if radius_neighbors is not None
                else "fixed"
            ),
            "radius_median": float(np.median(local_radii)),
            "radius_q10": float(np.quantile(local_radii, 0.1)),
            "radius_q90": float(np.quantile(local_radii, 0.9)),
            "target_neighbors": (
                None if radius_neighbors is None else int(radius_neighbors)
            ),
            "first_ring_neighbors_median": float(np.median(first_ring)),
            "first_ring_neighbors_q10": float(np.quantile(first_ring, 0.1)),
            "first_ring_neighbors_q90": float(np.quantile(first_ring, 0.9)),
        }
        diagnostics.append(audit)
        print(
            f"[MENDER] {value}: {len(rows)} cells, "
            f"radius median={audit['radius_median']:.4g}, "
            f"first-ring median={audit['first_ring_neighbors_median']:.1f}",
            flush=True,
        )
        bg_counts = joint[:, :, :n_bg].reshape(len(rows), scale * n_bg)
        sal_counts = joint[:, :, n_bg:].reshape(len(rows), scale * n_sal)
        bg_out[rows] = _normalize_log_counts(bg_counts)
        sal_out[rows] = _normalize_log_counts(sal_counts)
    if return_diagnostics:
        return bg_out, sal_out, diagnostics
    return bg_out, sal_out
