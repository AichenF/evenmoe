from __future__ import annotations

from typing import Tuple

import numpy as np


def closed_form_pair_oracle(
    global_counts: np.ndarray,
    log2phy_map: np.ndarray,
    log_counts: np.ndarray,
    ep_size: int,
    num_physical_experts: int,
    max_copies: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """NumPy reference for the LP-CF pair-split algorithm."""

    num_logical = len(global_counts)
    num_local = num_physical_experts // ep_size
    out = np.zeros((num_logical, max_copies), dtype=np.float32)
    rank_load = np.zeros(ep_size, dtype=np.float32)
    total = max(float(global_counts.sum()), 1.0)
    inv_total = 1.0 / total

    for logical in range(num_logical):
        if log_counts[logical] != 1 or global_counts[logical] <= 0:
            continue
        physical = int(log2phy_map[logical, 0])
        if physical < 0:
            continue
        load = float(global_counts[logical])
        rank_load[physical // num_local] += load
        out[logical, 0] = load * inv_total

    for logical in range(num_logical):
        copies = int(log_counts[logical])
        load = float(global_counts[logical])
        if copies < 3 or load <= 0:
            continue
        share = load / copies
        for replica in range(min(copies, max_copies)):
            physical = int(log2phy_map[logical, replica])
            if physical < 0:
                continue
            rank_load[physical // num_local] += share
            out[logical, replica] = share * inv_total

    load_snapshot = rank_load.copy()
    for logical in range(num_logical):
        if int(log_counts[logical]) != 2 or float(global_counts[logical]) <= 0:
            continue
        physical_a = int(log2phy_map[logical, 0])
        physical_b = int(log2phy_map[logical, 1])
        if physical_a < 0 or physical_b < 0:
            continue
        rank_a = physical_a // num_local
        rank_b = physical_b // num_local
        load = float(global_counts[logical])
        if rank_a == rank_b:
            probability_a = 0.5
            rank_load[rank_a] += load
        else:
            probability_a = max(
                0.0,
                min(
                    1.0,
                    (load_snapshot[rank_b] - load_snapshot[rank_a] + load) / (2.0 * load),
                ),
            )
            rank_load[rank_a] += probability_a * load
            rank_load[rank_b] += (1.0 - probability_a) * load
        out[logical, 0] = probability_a * load * inv_total
        out[logical, 1] = (1.0 - probability_a) * load * inv_total

    return out, rank_load
