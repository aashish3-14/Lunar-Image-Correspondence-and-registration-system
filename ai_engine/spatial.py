from __future__ import annotations

import numpy as np


def analyze_spatial_distribution(
    points: np.ndarray,
    image_width: int,
    image_height: int,
    grid_rows: int = 4,
    grid_cols: int = 4,
) -> dict:
    """
    Analyze the spatial distribution of correspondence points.

    The image is divided into a grid. Coverage measures the fraction
    of occupied cells, while entropy-based uniformity measures how
    evenly the points are distributed among occupied cells.
    """

    points = np.asarray(
        points,
        dtype=np.float32,
    )

    empty_result = {
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "occupied_cells": 0,
        "total_cells": grid_rows * grid_cols,
        "coverage": 0.0,
        "uniformity": 0.0,
        "distribution": "EMPTY",
    }

    if len(points) == 0:
        return empty_result

    valid = (
        np.isfinite(points[:, 0])
        & np.isfinite(points[:, 1])
        & (points[:, 0] >= 0)
        & (points[:, 0] < image_width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < image_height)
    )

    points = points[valid]

    if len(points) == 0:
        return empty_result

    cell_width = image_width / grid_cols
    cell_height = image_height / grid_rows

    cols = np.floor(
        points[:, 0] / cell_width
    ).astype(int)

    rows = np.floor(
        points[:, 1] / cell_height
    ).astype(int)

    cols = np.clip(
        cols,
        0,
        grid_cols - 1,
    )

    rows = np.clip(
        rows,
        0,
        grid_rows - 1,
    )

    cell_indices = (
        rows * grid_cols + cols
    )

    total_cells = (
        grid_rows * grid_cols
    )

    counts = np.bincount(
        cell_indices,
        minlength=total_cells,
    )

    occupied_cells = int(
        np.count_nonzero(counts)
    )

    coverage = (
        occupied_cells /
        total_cells
    )

    probabilities = (
        counts[counts > 0]
        .astype(np.float64)
    )

    probabilities /= probabilities.sum()

    entropy = -np.sum(
        probabilities *
        np.log(probabilities)
    )

    max_entropy = np.log(
        total_cells
    )

    uniformity = (
        float(entropy / max_entropy)
        if max_entropy > 0
        else 0.0
    )

    uniformity = float(
        np.clip(
            uniformity,
            0.0,
            1.0,
        )
    )

    if coverage < 0.20:
        distribution = "VERY LOW COVERAGE"
    elif coverage < 0.35:
        distribution = "LOW COVERAGE"
    elif uniformity < 0.40:
        distribution = "CLUSTERED"
    elif uniformity < 0.65:
        distribution = "MODERATELY DISTRIBUTED"
    else:
        distribution = "WELL DISTRIBUTED"

    return {
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "occupied_cells": occupied_cells,
        "total_cells": total_cells,
        "coverage": float(coverage),
        "uniformity": uniformity,
        "distribution": distribution,
    }


def select_uniform_matches(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    match_scores: np.ndarray | None,
    image_width: int,
    image_height: int,
    grid_rows: int = 4,
    grid_cols: int = 4,
    max_per_cell: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select geometrically valid matches with improved spatial distribution.

    Points are divided into a spatial grid. Each occupied source-image
    cell contributes up to `max_per_cell` matches, prioritizing higher
    LightGlue confidence scores.

    Parameters
    ----------
    source_points:
        Nx2 source-image coordinates.
    reference_points:
        Nx2 reference-image coordinates.
    match_scores:
        N confidence scores corresponding to the matches.
        If None, all matches receive equal priority.
    image_width, image_height:
        Dimensions of the source image.
    grid_rows, grid_cols:
        Spatial selection grid.
    max_per_cell:
        Maximum selected matches from one cell.

    Returns
    -------
    selected_source_points:
        Selected source coordinates.
    selected_reference_points:
        Selected reference coordinates.
    selected_indices:
        Indices into the input arrays.
    """

    source_points = np.asarray(
        source_points,
        dtype=np.float32,
    )

    reference_points = np.asarray(
        reference_points,
        dtype=np.float32,
    )

    if len(source_points) != len(reference_points):
        raise ValueError(
            "source_points and reference_points "
            "must have the same length"
        )

    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError(
            "grid_rows and grid_cols must be positive"
        )

    if max_per_cell <= 0:
        raise ValueError(
            "max_per_cell must be positive"
        )

    if len(source_points) == 0:
        return (
            np.empty(
                (0, 2),
                dtype=np.float32,
            ),
            np.empty(
                (0, 2),
                dtype=np.float32,
            ),
            np.empty(
                (0,),
                dtype=np.int32,
            ),
        )

    if match_scores is None:
        match_scores = np.ones(
            len(source_points),
            dtype=np.float32,
        )
    else:
        match_scores = np.asarray(
            match_scores,
            dtype=np.float32,
        )

        if len(match_scores) != len(source_points):
            raise ValueError(
                "match_scores must have the same "
                "length as source_points"
            )

    # Invalid scores should never outrank valid confidence values.
    safe_scores = np.nan_to_num(
        match_scores,
        nan=-np.inf,
        posinf=np.inf,
        neginf=-np.inf,
    )

    cell_width = (
        float(image_width) /
        float(grid_cols)
    )

    cell_height = (
        float(image_height) /
        float(grid_rows)
    )

    cols = np.floor(
        source_points[:, 0] /
        max(cell_width, 1e-12)
    ).astype(int)

    rows = np.floor(
        source_points[:, 1] /
        max(cell_height, 1e-12)
    ).astype(int)

    cols = np.clip(
        cols,
        0,
        grid_cols - 1,
    )

    rows = np.clip(
        rows,
        0,
        grid_rows - 1,
    )

    selected_indices = []

    for row in range(grid_rows):
        for col in range(grid_cols):

            cell_mask = (
                (rows == row)
                & (cols == col)
            )

            cell_indices = np.where(
                cell_mask
            )[0]

            if len(cell_indices) == 0:
                continue

            cell_scores = (
                safe_scores[cell_indices]
            )

            # Stable descending score order.
            order = np.argsort(
                -cell_scores,
                kind="stable",
            )

            chosen = cell_indices[
                order[:max_per_cell]
            ]

            selected_indices.extend(
                chosen.tolist()
            )

    selected_indices = np.asarray(
        selected_indices,
        dtype=np.int32,
    )

    # Restore original correspondence order so that the selected
    # source/reference arrays remain easy to trace back to LightGlue.
    selected_indices = np.sort(
        selected_indices
    )

    return (
        source_points[
            selected_indices
        ],
        reference_points[
            selected_indices
        ],
        selected_indices,
    )


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    points = rng.uniform(
        low=[0, 0],
        high=[1000, 1000],
        size=(100, 2),
    ).astype(np.float32)

    reference = (
        points + rng.normal(
            0,
            0.5,
            size=points.shape,
        )
    ).astype(np.float32)

    scores = rng.uniform(
        0.5,
        1.0,
        size=len(points),
    ).astype(np.float32)

    selected_source, selected_reference, selected_indices = (
        select_uniform_matches(
            source_points=points,
            reference_points=reference,
            match_scores=scores,
            image_width=1000,
            image_height=1000,
            grid_rows=4,
            grid_cols=4,
            max_per_cell=8,
        )
    )

    spatial = analyze_spatial_distribution(
        points=selected_source,
        image_width=1000,
        image_height=1000,
        grid_rows=4,
        grid_cols=4,
    )

    print("Spatial module self-test")
    print(f"Input matches:    {len(points)}")
    print(f"Selected matches: {len(selected_indices)}")
    print(f"Coverage:         {spatial['coverage']:.3f}")
    print(f"Uniformity:       {spatial['uniformity']:.3f}")
    print(f"Distribution:     {spatial['distribution']}")
