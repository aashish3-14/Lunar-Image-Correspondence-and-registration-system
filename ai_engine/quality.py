from __future__ import annotations


def evaluate_registration_quality(
    total_matches: int,
    inlier_count: int,
    min_inliers: int = 10,
    min_inlier_ratio: float = 0.20,
    spatial_coverage: float = 0.0,
    min_spatial_coverage: float = 0.25,
):
    """
    Evaluate correspondence quality for LUCAS registration.

    LUCAS checks three independent criteria:

    1. Minimum number of geometric inliers
    2. Minimum inlier ratio
    3. Minimum spatial coverage

    The registration is accepted only when ALL criteria pass.

    Parameters
    ----------
    total_matches:
        Total number of candidate matches produced by the matcher.

    inlier_count:
        Number of matches that remain geometrically consistent
        after RANSAC.

    min_inliers:
        Minimum number of geometric inliers required.

    min_inlier_ratio:
        Minimum acceptable ratio of inliers to total matches.

    spatial_coverage:
        Fraction of spatial grid cells occupied by inlier points.

    min_spatial_coverage:
        Minimum acceptable spatial coverage.

    Returns
    -------
    dict
        Quality assessment containing:

        success
        reason
        reasons
        inlier_count
        inlier_ratio
        spatial_coverage
    """

    # ========================================================
    # NO MATCHES
    # ========================================================

    if total_matches <= 0:

        return {
            "success": False,
            "reason": "No matches found",
            "reasons": [
                "No matches found"
            ],
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "spatial_coverage": 0.0,
        }

    # ========================================================
    # INLIER RATIO
    # ========================================================

    inlier_ratio = (
        inlier_count / total_matches
    )

    # ========================================================
    # COLLECT ALL FAILED CRITERIA
    # ========================================================

    failures = []

    # --------------------------------------------------------
    # Criterion 1: Minimum geometric inliers
    # --------------------------------------------------------

    if inlier_count < min_inliers:

        failures.append(
            f"Too few geometric inliers "
            f"({inlier_count} < {min_inliers})"
        )

    # --------------------------------------------------------
    # Criterion 2: Minimum inlier ratio
    # --------------------------------------------------------

    if inlier_ratio < min_inlier_ratio:

        failures.append(
            f"Low inlier ratio "
            f"({inlier_ratio:.3f} < "
            f"{min_inlier_ratio:.3f})"
        )

    # --------------------------------------------------------
    # Criterion 3: Minimum spatial coverage
    # --------------------------------------------------------

    if spatial_coverage < min_spatial_coverage:

        failures.append(
            f"Insufficient spatial coverage "
            f"({spatial_coverage:.3f} < "
            f"{min_spatial_coverage:.3f})"
        )

    # ========================================================
    # REGISTRATION REJECTED
    # ========================================================

    if failures:

        return {
            "success": False,

            # First failure is the primary reason.
            "reason": failures[0],

            # Complete list of failed criteria.
            "reasons": failures,

            "inlier_count": int(
                inlier_count
            ),

            "inlier_ratio": float(
                inlier_ratio
            ),

            "spatial_coverage": float(
                spatial_coverage
            ),
        }

    # ========================================================
    # REGISTRATION ACCEPTED
    # ========================================================

    return {
        "success": True,

        "reason": (
            "Geometric and spatial "
            "verification passed"
        ),

        "reasons": [],

        "inlier_count": int(
            inlier_count
        ),

        "inlier_ratio": float(
            inlier_ratio
        ),

        "spatial_coverage": float(
            spatial_coverage
        ),
    }


# ============================================================
# SIMPLE STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 50)
    print("       LUCAS QUALITY GATE TEST")
    print("=" * 50)

    # --------------------------------------------------------
    # Test case 1: Current difficult lunar pair
    # --------------------------------------------------------

    print()
    print("TEST 1: Current LUCAS baseline")

    result = evaluate_registration_quality(
        total_matches=19,
        inlier_count=5,
        spatial_coverage=0.25,
    )

    print(
        f"Status: "
        f"{'ACCEPTED' if result['success'] else 'REJECTED'}"
    )

    print(
        f"Primary reason: "
        f"{result['reason']}"
    )

    if result["reasons"]:

        print("Failure criteria:")

        for reason in result["reasons"]:
            print(f"  • {reason}")

    print(
        f"Inliers: "
        f"{result['inlier_count']}"
    )

    print(
        f"Inlier ratio: "
        f"{result['inlier_ratio']:.3f}"
    )

    print(
        f"Spatial coverage: "
        f"{result['spatial_coverage']:.3f}"
    )

    # --------------------------------------------------------
    # Test case 2: Strong hypothetical correspondence
    # --------------------------------------------------------

    print()
    print("TEST 2: Strong correspondence")

    result = evaluate_registration_quality(
        total_matches=50,
        inlier_count=30,
        spatial_coverage=0.75,
    )

    print(
        f"Status: "
        f"{'ACCEPTED' if result['success'] else 'REJECTED'}"
    )

    print(
        f"Primary reason: "
        f"{result['reason']}"
    )

    print(
        f"Inliers: "
        f"{result['inlier_count']}"
    )

    print(
        f"Inlier ratio: "
        f"{result['inlier_ratio']:.3f}"
    )

    print(
        f"Spatial coverage: "
        f"{result['spatial_coverage']:.3f}"
    )

    print()
    print("=" * 50)