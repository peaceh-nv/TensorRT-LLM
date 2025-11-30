import numpy as np
from mypy_extensions import i32


def copy_batch_block_offsets(
    dst_tensor: np.ndarray,
    batch_size: i32,
    batch_cache_indices: list[np.ndarray],
    num_pools: i32,
    offset: i32,
) -> None:
    for pool_idx in range(num_pools):
        for idx in range(batch_size):
            batch_idx = pool_idx * batch_size + idx
            batch_cache_index = batch_cache_indices[batch_idx]
            dst_tensor[pool_idx, idx + offset, 0, : len(batch_cache_index)] = batch_cache_index
            dst_tensor[pool_idx, idx + offset, 1, : len(batch_cache_index)] = batch_cache_index + 1
