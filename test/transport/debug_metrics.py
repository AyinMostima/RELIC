"""
Debug script to check grid metrics return values
"""

import torch
from relic.computation.operators.functional import compute_grid_metrics

# Test grid metrics
metrics = compute_grid_metrics((11,), 0.1, torch.device("cpu"))
print("Metrics keys:", list(metrics.keys()))

# Test 2D metrics
metrics_2d = compute_grid_metrics((5, 11), (0.1, 0.2), torch.device("cpu"))
print("2D Metrics keys:", list(metrics_2d.keys()))

