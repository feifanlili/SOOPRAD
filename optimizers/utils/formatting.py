"""
utils/formatting.py

Utility functions for JSON serialization of NumPy data structures.

This module provides a helper function to recursively convert NumPy types
(e.g., arrays, floats, ints) into native Python types (lists, float, int)
that are compatible with JSON serialization.

Typical use case:
    - Preparing data for logging or saving model results to a JSON file
      when the data includes NumPy arrays or scalars.

Functions:
    make_json_serializable(obj): Recursively transforms NumPy data into JSON-safe formats.
"""

import numpy as np

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj