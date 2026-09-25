"""Conservative remapping between one-dimensional non-uniform grids.

Unlike :py:meth:`DataArray.interp`, which treats values as point samples and
interpolates them, the routines here treat values as *cell quantities*
(either cell totals or per-unit-length densities) and redistribute them by
the overlap fraction between source and target cells, conserving the
integral over the domain.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, Literal

import numpy as np

from xarray.computation.apply_ufunc import apply_ufunc

if TYPE_CHECKING:
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset

    T_Data = DataArray | Dataset


def _as_cell_pairs(bounds, name: str) -> np.ndarray:
    """Coerce bounds to an (n, 2) array of (lower, upper) pairs.

    Accepts either (n, 2) per-cell bounds or (n + 1,) contiguous cell edges.
    """
    arr = np.asarray(bounds, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size < 2:
            raise ValueError(
                f"{name} must contain at least two cell edges, got {arr.size}"
            )
        arr = np.column_stack([arr[:-1], arr[1:]])
    elif arr.ndim == 2 and arr.shape[1] == 2:
        pass
    else:
        raise ValueError(
            f"{name} must be either per-cell bounds with shape (n, 2) or "
            f"contiguous cell edges with shape (n + 1,), got shape {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must describe at least one cell")
    if np.isnan(arr).any():
        raise ValueError(f"{name} must not contain NaN values")
    return arr


def _normalize_bounds(
    bounds, name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Validate cell bounds and return them in strictly ascending order.

    Returns
    -------
    lo, hi : np.ndarray
        Ascending cell bounds, reordered so cells are sorted by position.
    lengths : np.ndarray
        Cell lengths in the *original* cell order.
    reversed : bool
        Whether the original cells were ordered descending.
    """
    pairs = _as_cell_pairs(bounds, name)
    deltas = pairs[:, 1] - pairs[:, 0]
    if np.all(deltas > 0):
        lo, hi = pairs[:, 0], pairs[:, 1]
        lengths = deltas
        reversed_ = False
    elif np.all(deltas < 0):
        # descending cells: swap each pair and reverse the cell order
        lo, hi = pairs[::-1, 1], pairs[::-1, 0]
        lengths = -deltas
        reversed_ = True
    else:
        raise ValueError(
            f"{name} contains inverted (lower >= upper) or zero-width cells, "
            "or mixes ascending and descending cells"
        )
    if lo.size > 1 and np.any(lo[1:] < hi[:-1]):
        raise ValueError(
            f"{name} contains overlapping cells; cell bounds must be "
            "non-overlapping (touching edges are allowed)"
        )
    return lo, hi, lengths, reversed_


def _overlap_weights(
    src_lo: np.ndarray,
    src_hi: np.ndarray,
    tgt_lo: np.ndarray,
    tgt_hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sparse source/target intersection weights in CSR-like form.

    Returns (cols, vals, row_ptr) where vals[k] is the length of the overlap
    between target cell t and source cell cols[k], with
    row_ptr[t] : row_ptr[t + 1] selecting the entries of target t.
    Only the O(n_source + n_target) non-empty intersections are stored; the
    dense (n_target, n_source) intersection matrix is never materialized.
    """
    n_tgt = tgt_lo.shape[0]
    # first source cell whose upper edge is above the target lower edge
    starts = np.searchsorted(src_hi, tgt_lo, side="right")
    # first source cell whose lower edge is at or above the target upper edge
    ends = np.searchsorted(src_lo, tgt_hi, side="left")
    counts = np.maximum(ends - starts, 0)
    row_ptr = np.zeros(n_tgt + 1, dtype=np.intp)
    np.cumsum(counts, out=row_ptr[1:])
    nnz = int(row_ptr[-1])
    rows = np.repeat(np.arange(n_tgt), counts)
    # column ranges starts[t] .. ends[t] - 1, without a Python loop
    cols = np.repeat(starts - row_ptr[:-1], counts) + np.arange(nnz)
    lo = np.maximum(tgt_lo[rows], src_lo[cols])
    hi = np.minimum(tgt_hi[rows], src_hi[cols])
    vals = hi - lo
    return cols, vals, row_ptr


def _seg_matvec(
    y: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    row_ptr: np.ndarray,
    n_tgt: int,
) -> np.ndarray:
    """Compute out[t] = sum_k vals[k] * y[..., cols[k]] segment-wise."""
    counts = np.diff(row_ptr)
    nonempty = counts > 0
    out = np.zeros(y.shape[:-1] + (n_tgt,), dtype=y.dtype)
    if np.any(nonempty):
        contrib = y[..., cols] * vals
        starts = row_ptr[:-1][nonempty]
        out[..., nonempty] = np.add.reduceat(contrib, starts, axis=-1)
    return out


def _remap_numpy(
    x: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    row_ptr: np.ndarray,
    src_lengths: np.ndarray,
    tgt_lengths: np.ndarray,
    data_kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply sparse overlap weights along the last axis of x.

    Returns the remapped values and the valid coverage fraction of each
    target cell. Cells without any valid source coverage become NaN.
    """
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float64)
    n_tgt = tgt_lengths.shape[0]
    valid = ~np.isnan(x)
    # total source length with valid data overlapping each target cell
    covered = _seg_matvec(valid.astype(x.dtype), cols, vals, row_ptr, n_tgt)
    coverage = covered / tgt_lengths
    if data_kind == "total":
        # distribute each cell total by its per-unit-length density
        filled = np.where(valid, x / src_lengths, 0.0)
        out = _seg_matvec(filled, cols, vals, row_ptr, n_tgt)
    else:
        # average the density over the covered part of each target cell
        filled = np.where(valid, x, 0.0)
        numerator = _seg_matvec(filled, cols, vals, row_ptr, n_tgt)
        safe_covered = np.where(covered > 0, covered, 1.0)
        out = numerator / safe_covered
    out = np.where(covered > 0, out, np.nan)
    return out, coverage


def conservative_remap(
    data: T_Data,
    src_bounds,
    tgt_bounds,
    dim: Hashable,
    *,
    data_kind: Literal["total", "density"] = "total",
    return_coverage: bool = False,
) -> T_Data | tuple[T_Data, T_Data]:
    """Conservatively remap cell values between 1D non-uniform grids.

    Values are redistributed from source to target cells in proportion to
    the length of their overlap, conserving the integral over the domain.
    This contrasts with :py:meth:`DataArray.interp`, which treats values as
    point samples rather than cell quantities.

    Parameters
    ----------
    data : DataArray or Dataset
        Values living on the source cells along ``dim``. Other dimensions
        are broadcast and preserved.
    src_bounds, tgt_bounds : array-like
        Cell bounds of the source and target grids, either as (n, 2)
        per-cell (lower, upper) pairs or as (n + 1,) contiguous cell
        edges. Bounds may be ascending or descending, but must be
        non-overlapping and non-inverted. The bounds' own dimension or
        coordinate names need not match ``dim``.
    dim : Hashable
        Name of the dimension of ``data`` to remap. Its size must match the
        number of source cells.
    data_kind : {"total", "density"}, default: "total"
        Whether ``data`` represents cell *totals* (extensive quantities such
        as mass or count, redistributed by overlap fraction) or
        *per-unit-length densities* (intensive quantities, averaged over the
        covered part of each target cell).
    return_coverage : bool, default: False
        If True, also return the fraction of each target cell covered by
        valid (non-NaN) source cells, so the caller can decide how to treat
        insufficient coverage.

    Returns
    -------
    remapped : DataArray or Dataset
        Values on the target cells. Target cells with no valid source
        coverage are NaN.
    coverage : DataArray or Dataset, optional
        Valid coverage fraction in [0, 1] for each target cell, only
        returned when ``return_coverage`` is True.

    Examples
    --------
    >>> da = xr.DataArray([2.0, 4.0, 6.0], dims="x")
    >>> src = [[0.0, 1.0], [1.0, 3.0], [3.0, 6.0]]
    >>> tgt = [[0.0, 2.0], [2.0, 6.0]]
    >>> conservative_remap(da, src, tgt, dim="x", data_kind="total")
    <xarray.DataArray (x: 2)> Size: 16B
    array([4., 8.])
    Dimensions without coordinates: x
    """
    if data_kind not in ("total", "density"):
        raise ValueError(
            f"data_kind must be 'total' or 'density', got {data_kind!r}"
        )
    if dim not in data.dims:
        raise ValueError(
            f"dimension {dim!r} not found in data dimensions {data.dims}"
        )

    src_lo, src_hi, src_lengths, src_reversed = _normalize_bounds(
        src_bounds, "src_bounds"
    )
    tgt_lo, tgt_hi, tgt_lengths, tgt_reversed = _normalize_bounds(
        tgt_bounds, "tgt_bounds"
    )
    n_src = src_lo.shape[0]
    n_tgt = tgt_lo.shape[0]
    if data.sizes[dim] != n_src:
        raise ValueError(
            f"src_bounds describes {n_src} cells but dimension {dim!r} of "
            f"data has size {data.sizes[dim]}"
        )

    cols, vals, row_ptr = _overlap_weights(src_lo, src_hi, tgt_lo, tgt_hi)
    if src_reversed:
        # weights of columns refer to the ascending (reversed) source order;
        # map them back to the original cell order
        cols = n_src - 1 - cols
    if tgt_reversed:
        # weight rows follow the ascending (reversed) target order, so the
        # target lengths used inside the remapping must as well
        tgt_lengths = tgt_lengths[::-1]

    dtype = getattr(data, "dtype", None)
    if dtype is None or not np.issubdtype(dtype, np.floating):
        dtype = np.float64
    remapped, coverage = apply_ufunc(
        _remap_numpy,
        data,
        input_core_dims=[[dim]],
        output_core_dims=[[dim], [dim]],
        exclude_dims={dim},
        kwargs=dict(
            cols=cols,
            vals=vals,
            row_ptr=row_ptr,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            data_kind=data_kind,
        ),
        dask="parallelized",
        dask_gufunc_kwargs=dict(output_sizes={dim: n_tgt}, allow_rechunk=True),
        output_dtypes=[dtype, dtype],
        keep_attrs=True,
    )
    if tgt_reversed:
        # results were computed in ascending target order; restore the
        # original (descending) cell order
        indexer = {dim: slice(None, None, -1)}
        remapped = remapped.isel(indexer)
        coverage = coverage.isel(indexer)

    coverage.attrs = {}
    if return_coverage:
        return remapped, coverage
    return remapped
