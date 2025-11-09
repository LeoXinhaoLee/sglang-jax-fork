# Adapted from https://github.com/jax-ml/jax/releases/tag/jax-v0.8.0
# Copyright 2025 The JAX Authors. All rights reserved.

import functools
from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.layers.gmm.megablox_gmm_kernel import common


def _validate_args(
    *,
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    expected_rhs_dims: int = 3,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.dtype]:
    """Validates the arguments for the gmm function."""
    # Validate 'lhs'.
    if lhs.ndim != 2:
        raise ValueError(f"Expected 2-tensor for 'lhs' but got {lhs.ndim}-tensor.")
    common.assert_is_supported_dtype(lhs.dtype)

    # Validate 'rhs'.
    if rhs.ndim != expected_rhs_dims:
        raise ValueError(
            f"Expected {expected_rhs_dims}-tensor for 'rhs' but got {rhs.ndim}-tensor."
        )
    common.assert_is_supported_dtype(rhs.dtype)

    # Validate 'group_sizes'.
    if group_sizes.dtype != jnp.int32:
        raise ValueError(f"Expected 32-bit integer 'group_sizes' but got {group_sizes.dtype}.")

    return lhs, group_sizes, common.select_input_dtype(lhs, rhs)


def _calculate_num_tiles(x: int, tx: int) -> int:
    tiles, rem = divmod(x, tx)
    if rem:
        raise ValueError(f"{x} must be divisible by x-dimension tile size ({tx}).")
    return tiles


def _calculate_irregular_num_tiles(x: int, tx: int) -> tuple[int, int]:
    tiles, rem = divmod(x, tx)
    if rem:
        tiles += 1
    return tiles, rem


GroupMetadata = Any  # A tuple that minimally contains (group_offsets, group_ids, m_tile_ids).
# In this implementation we add derived arrays to speed the kernel:
# (group_offsets, group_ids, m_tile_ids, group_start_rows, group_end_rows, first_visit_flags)


def make_group_metadata(
    *,
    group_sizes: jnp.ndarray,
    m: int,
    tm: int,
    start_group: jnp.ndarray,
    num_nonzero_groups: int,
    visit_empty_groups: bool = True,
) -> GroupMetadata:
    """Create the metadata needed for grouped matmul computation.

    Optimizations:
    - Use jnp.bincount (with masking) instead of jnp.histogram to count partial-tile visits.
    - Avoid large 2D iotas entirely in metadata creation.
    """
    num_groups = group_sizes.shape[0]
    end_group = start_group + num_nonzero_groups - 1

    # CSR-like offsets: group_offsets[0] = 0, group_offsets[num_groups] = m
    group_ends = jnp.cumsum(group_sizes)
    group_offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends])

    # Round group boundaries to tile edges for tile counting.
    rounded_group_ends = ((group_ends + tm - 1) // tm * tm).astype(jnp.int32)
    group_starts = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends[:-1]])
    rounded_group_starts = (group_starts // tm) * tm

    rounded_group_sizes = rounded_group_ends - rounded_group_starts
    rounded_group_sizes = jnp.where(group_sizes == 0, 0, rounded_group_sizes)

    # Convert from rows to tiles.
    group_tiles = rounded_group_sizes // tm
    if visit_empty_groups:
        group_tiles = jnp.where(group_sizes == 0, 1, group_tiles)

    tiles_m = _calculate_num_tiles(m, tm)

    # Per-grid group ids.
    group_ids = jnp.repeat(
        jnp.arange(num_groups, dtype=jnp.int32),
        group_tiles,
        total_repeat_length=tiles_m + num_groups - 1,
    )

    # Count partial visits per tile (fast path via bincount).
    starts = group_offsets[:-1]
    partial_tile_mask = jnp.logical_or((starts % tm) == 0, group_sizes == 0)
    if visit_empty_groups:
        partial_tile_mask = jnp.where(group_sizes == 0, False, partial_tile_mask)
    partial_tile_ids = jnp.where(partial_tile_mask, tiles_m, starts // tm)
    # Count only ids in [0, tiles_m), ignore sentinel == tiles_m via weights.
    safe_ids = jnp.minimum(partial_tile_ids, tiles_m - 1)
    weights = (partial_tile_ids < tiles_m).astype(jnp.int32)
    tile_visits = jnp.bincount(safe_ids, weights=weights, length=tiles_m) + 1

    # Per-grid m-tile ids.
    m_tile_ids = jnp.repeat(
        jnp.arange(tiles_m, dtype=jnp.int32),
        tile_visits.astype(jnp.int32),
        total_repeat_length=tiles_m + num_groups - 1,
    )

    # Sharding: rotate so our shard's tiles are first.
    first_tile_in_shard = (group_ids < start_group).sum()
    group_ids = jnp.roll(group_ids, shift=-first_tile_in_shard, axis=0)
    m_tile_ids = jnp.roll(m_tile_ids, shift=-first_tile_in_shard, axis=0)

    # Number of tiles for this shard.
    iota = jnp.arange(num_groups, dtype=jnp.int32)
    active_group_mask = jnp.logical_and(iota <= end_group, iota >= start_group)
    group_tiles_in_shard = jnp.where(active_group_mask, group_tiles, 0)
    num_tiles = group_tiles_in_shard.sum()

    # Pre-gathered per-grid bounds and first-visit flags (speed up kernel).
    group_start_rows = group_offsets[group_ids]
    group_end_rows = group_offsets[group_ids + 1]
    prev_m = jnp.roll(m_tile_ids, 1)
    first_visit_flags = (jnp.arange(m_tile_ids.size, dtype=jnp.int32) == 0) | (m_tile_ids != prev_m)

    return (
        group_offsets,
        group_ids,
        m_tile_ids,
        group_start_rows,
        group_end_rows,
        first_visit_flags,
    ), num_tiles


def _get_group_size(*, grid_id: jnp.ndarray, group_metadata: GroupMetadata) -> jnp.ndarray:
    """Calculate the number of rows in the current group."""
    group_offsets, group_ids = group_metadata[:2]
    group_id = group_ids[grid_id]
    group_start = group_offsets[group_id]
    group_end = group_offsets[group_id + 1]
    return group_end - group_start


def _get_store_mask(
    *,
    grid_id: jnp.ndarray,
    group_metadata: GroupMetadata,
    tm: int,
    tn: int,
) -> jnp.ndarray:
    """Mask for rows that belong to the current group in the current tile.

    NOTE: Kept for compatibility; the kernel uses a faster numeric row-mask path.
    """
    group_offsets, group_ids, m_tile_ids = group_metadata[:3]
    group_id = group_ids[grid_id]
    group_start = group_offsets[group_id]
    group_end = group_offsets[group_id + 1]
    m_id = m_tile_ids[grid_id] * tm
    iota2d = jax.lax.broadcasted_iota(jnp.int32, (tm, tn), 0) + m_id
    return (iota2d >= group_start) & (iota2d < group_end)


def _zero_uninitialized_memory(
    out: jnp.ndarray,
    *,
    start_group: jnp.ndarray,
    num_nonzero_groups: int,
    group_metadata: GroupMetadata,
) -> jnp.ndarray:
    """Zero out uninitialized memory from output."""
    group_offsets = group_metadata[0]
    group_start = group_offsets[start_group]
    group_end = group_offsets[start_group + num_nonzero_groups]
    rows = jax.lax.broadcasted_iota(jnp.int32, (out.shape[0],), 0)
    valid_mask = (rows >= group_start) & (rows < group_end)
    return jnp.where(valid_mask[:, None], out, 0)


LutFn = Callable[[int, int, int], tuple[int, int, int] | None]


@functools.partial(
    jax.jit,
    static_argnames=[
        "preferred_element_type",
        "tiling",
        "transpose_rhs",
        "interpret",
    ],
)
def gmm(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128),
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
) -> jnp.ndarray:
    """Compute lhs[sizes[i-1]:sizes[i], :] @ rhs for each group 'i'."""
    # Validate/normalize inputs.
    if existing_out is not None:
        assert isinstance(existing_out, jax.Array)
        if existing_out.dtype != preferred_element_type:
            raise ValueError("Existing output dtype must match preferred_element_type.")
    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    else:
        if group_offset.shape:
            raise ValueError(f"group_offset must be a ()-shaped array. Got: {group_offset.shape}.")
        group_offset = group_offset[None]

    num_current_groups = rhs.shape[0]
    num_total_groups = group_sizes.shape[0]
    lhs, group_sizes, input_dtype = _validate_args(lhs=lhs, rhs=rhs, group_sizes=group_sizes)

    # Shapes.
    m, k, n = (lhs.shape[0], lhs.shape[1], rhs.shape[2])
    if transpose_rhs:
        n = rhs.shape[1]

    # Tiling selection / LUT.
    if callable(tiling):
        tiling = tiling(m, k, n)
    if tiling is None:
        raise ValueError(f"No tuned tiling found for (m, k, n) = ({m}, {k}, {n})")
    tm, tk, tn = tiling
    tiles_k, k_rem = _calculate_irregular_num_tiles(k, tk)
    tiles_n, _ = _calculate_irregular_num_tiles(n, tn)

    # Metadata for grouped execution.
    group_metadata, num_active_tiles = make_group_metadata(
        group_sizes=group_sizes,
        m=m,
        tm=tm,
        start_group=group_offset[0],
        num_nonzero_groups=rhs.shape[0],
        visit_empty_groups=False,
    )

    # Dot dimension numbers (static).
    dot_general_dims = (((1,), (1,)), ((), ())) if transpose_rhs else (((1,), (0,)), ((), ()))
    acc_dtype = jnp.float32  # accumulate in f32

    # ------------------------------- Kernel ---------------------------------
    def kernel(
        group_metadata,
        group_offset,
        lhs: jax.Array,
        rhs: jax.Array,
        existing_out,
        out,
        acc_scratch,
    ):
        # Unpack metadata (arrays only; no captured 0-D JAX scalars).
        (
            _group_offsets,   # kept for structure, not used in kernel
            group_ids,
            m_tile_ids,
            group_start_rows,
            group_end_rows,
            first_visit_flags,
        ) = group_metadata

        grid_id = pl.program_id(1)
        k_i = pl.program_id(2)

        # Prologue for reduction along K.
        @pl.when(k_i == 0)
        def _k0_prologue():
            acc_scratch[...] = jnp.zeros_like(acc_scratch, dtype=acc_dtype)
            if existing_out is not None:
                is_first_visit = first_visit_flags[grid_id]
                @pl.when(is_first_visit)
                def _init_out_from_existing():
                    out[...] = existing_out[...]

        # Build a fast (tm,) numeric row mask for this grid_id, then broadcast.
        # Use float32 (32-bit) to satisfy Mosaic reshape constraints.
        def _row_mask_tm_f32() -> jax.Array:
            g_start = group_start_rows[grid_id]
            g_end = group_end_rows[grid_id]
            m_base = m_tile_ids[grid_id] * tm  # tm is a Python int captured safely
            rows = m_base + jnp.arange(tm, dtype=jnp.int32)
            row_bool = (rows >= g_start) & (rows < g_end)  # (tm,) bool
            return row_bool.astype(jnp.float32)  # cast to 32-bit numeric

        # Apply K-tail mask only on the last K tile via numeric multiply.
        def _apply_k_mask_last_tile(lhs_tile: jax.Array, rhs_tile: jax.Array) -> tuple[jax.Array, jax.Array]:
            if k_rem == 0:  # Python int, not a JAX array
                return lhs_tile, rhs_tile
            k_mask_vec = (jnp.arange(tk, dtype=jnp.int32) < k_rem).astype(lhs_tile.dtype)  # (tk,)
            lhs_masked = lhs_tile * k_mask_vec[None, :]         # (tm, tk)
            if transpose_rhs:
                rhs_masked = rhs_tile * k_mask_vec[None, :]      # (tn, tk): mask columns
            else:
                rhs_masked = rhs_tile * k_mask_vec[:, None]      # (tk, tn): mask rows
            return lhs_masked, rhs_masked

        # Store accumulator into 'out' on the last K tile using numeric row mask.
        def _store_accum():
            mask_row = _row_mask_tm_f32()[:, None]  # (tm,1) float32 (safe for reshape)
            base = out[...].astype(acc_dtype)
            to_store = acc_scratch[...]
            # Blend without boolean where (avoids i1 reshape):
            # out = mask * to_store + (1 - mask) * base
            out[...] = (mask_row * to_store + (1.0 - mask_row) * base).astype(preferred_element_type)

        # Accumulate this K tile, and store if it is the last.
        def _accum(is_last_k_tile: bool):
            lhs_tile = lhs[...].astype(input_dtype)
            rhs_tile = rhs[...].astype(input_dtype)
            if is_last_k_tile:
                lhs_tile, rhs_tile = _apply_k_mask_last_tile(lhs_tile, rhs_tile)
            acc_scratch[...] = acc_scratch[...] + jax.lax.dot_general(
                lhs_tile,
                rhs_tile,
                dimension_numbers=dot_general_dims,
                preferred_element_type=acc_dtype,
            )
            if is_last_k_tile:
                _store_accum()

        lax.cond(
            k_i == tiles_k - 1,              # uses Python int tiles_k
            lambda: _accum(True),
            lambda: _accum(False),
        )

    # ------------------------------ Block specs ------------------------------
    def lhs_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset):
        # lhs is (m, k). Load the [tm, tk] matrix for this m-tile.
        _, _, m_tile_ids, *_ = group_metadata
        del n_i, group_offset
        return m_tile_ids[grid_id], k_i

    def rhs_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset):
        # rhs is (num_groups, k, n) or (num_groups, n, k) if transposed.
        _, group_ids, _, *_ = group_metadata
        if transpose_rhs:
            k_i, n_i = n_i, k_i
        # Adjust for sharded rhs: group_ids are in the unsharded domain.
        return group_ids[grid_id] - group_offset[0], k_i, n_i

    def out_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset):
        # out is (m, n). Load the [tm, tn] matrix for this m-tile.
        _, _, m_tile_ids, *_ = group_metadata
        del k_i, group_offset
        return m_tile_ids[grid_id], n_i

    out_block_spec = pl.BlockSpec((tm, tn), out_transform_indices)
    if existing_out is None:
        in_out_block_spec: Any = None
        input_output_aliases = {}
    else:
        in_out_block_spec = out_block_spec
        # Keep index consistent with kernel arg ordering / PrefetchScalarGridSpec.
        existing_out_arg_index = 6
        input_output_aliases = {existing_out_arg_index: 0}

    lhs_block_spec = pl.BlockSpec((tm, tk), lhs_transform_indices)
    if transpose_rhs:
        rhs_block_spec = pl.BlockSpec((None, tn, tk), rhs_transform_indices)
    else:
        rhs_block_spec = pl.BlockSpec((None, tk, tn), rhs_transform_indices)

    # Cost estimate (heuristic; not used for correctness).
    lhs_bytes = lhs.size * lhs.itemsize
    rhs_bytes = (k * n) * rhs.itemsize  # upper bound
    out_bytes = (m * n) * jnp.dtype(preferred_element_type).itemsize
    max_active_tiles = group_metadata[1].size  # len(group_ids)
    bytes_accessed = (lhs_bytes * tiles_n) + (rhs_bytes * max_active_tiles) + out_bytes
    flops = 2 * m * k * n
    cost_estimate = pl.CostEstimate(flops=flops, bytes_accessed=bytes_accessed, transcendentals=0)

    call_gmm = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), preferred_element_type),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            # Two "scalar" (non-block-spec) args precede the tiled args:
            #   (1) group_metadata pytree, (2) group_offset scalar array.
            num_scalar_prefetch=2,
            in_specs=[
                lhs_block_spec,
                rhs_block_spec,
                in_out_block_spec,
            ],
            out_specs=out_block_spec,
            grid=(tiles_n, num_active_tiles, tiles_k),
            scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)],  # f32 accumulator
        ),
        input_output_aliases=input_output_aliases,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary")
        ),
        interpret=interpret,
        cost_estimate=cost_estimate,
    )

    out = call_gmm(
        group_metadata,
        group_offset,
        lhs,
        rhs,
        existing_out,
    )

    # If we computed only a shard of groups and aren't accumulating into an
    # existing output, zero rows outside our shard.
    if existing_out is None and num_current_groups < num_total_groups:
        out = _zero_uninitialized_memory(
            out,
            start_group=group_offset[0],
            num_nonzero_groups=rhs.shape[0],
            group_metadata=group_metadata,
        )
    return out
