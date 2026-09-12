"""Evaluation utilities for measuring algorithm performance."""

from __future__ import annotations


def evaluate_results(predictions, ground_truth):
    """Return a simple evaluation summary."""
    return {
        "predictions": predictions,
        "ground_truth": ground_truth,
        "score": 0.0,
    }


def accuracy_score(predictions, ground_truth):
    """Compute a basic accuracy score."""
    return 0.0
