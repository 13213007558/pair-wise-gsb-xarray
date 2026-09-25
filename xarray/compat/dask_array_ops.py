from __future__ import annotations

import math
from functools import partial

import numpy as np

from xarray.compat.dask_array_compat import reshape_blockwise
from xarray.core import dtypes, nputils
from xarray.namedarray.parallelcompat import get_chunked_array_type


def dask_rolling_wrapper(
    moving_func, a, window, min_count=None, axis=-1, input_dtype=None
):
    """Wrapper to apply bottleneck moving window funcs on dask arrays"""
    if input_dtype is None:
        input_dtype = a.dtype
    if input_dtype.kind in "biufc":
        dtype = moving_func(
            np.empty((1,), dtype=input_dtype), window=1, min_count=1, axis=0
        ).dtype
    else:
        dtype, _ = dtypes.maybe_promote(a.dtype)
    return a.data.map_overlap(
        moving_func,
        depth={axis: (window - 1, 0)},
        axis=axis,
        dtype=dtype,
        window=window,
        min_count=min_count,
    )


def least_squares(lhs, rhs, rcond=None, skipna=False):
    da = get_chunked_array_type(rhs).array_api

    # The trick here is that the core dimension is axis 0.
    # All other dimensions need to be reshaped down to one axis for `lstsq`
    # (which only accepts 2D input)
    # and this needs to be undone after running `lstsq`
    # The order of values in the reshaped axes is irrelevant.
    # There are big gains to be had by simply reshaping the blocks on a blockwise
    # basis, and then undoing that transform.
    # We use a specific `reshape_blockwise` method in dask for this optimization
    if rhs.ndim > 2:
        out_shape = rhs.shape
        reshape_chunks = rhs.chunks
        rhs = reshape_blockwise(rhs, (rhs.shape[0], math.prod(rhs.shape[1:])))
    else:
        out_shape = None

    lhs_da = da.from_array(lhs, chunks=(rhs.chunks[0], lhs.shape[1]))
    if skipna:
        added_dim = rhs.ndim == 1
        if added_dim:
            rhs = rhs.reshape(rhs.shape[0], 1)
        results = da.apply_along_axis(
            nputils._nanpolyfit_1d,
            0,
            rhs,
            lhs_da,
            dtype=float,
            shape=(lhs.shape[1] + 1,),
            rcond=rcond,
        )
        coeffs = results[:-1, ...]
        residuals = results[-1, ...]
        if added_dim:
            coeffs = coeffs.reshape(coeffs.shape[0])
            residuals = residuals.reshape(residuals.shape[0])
    else:
        # Residuals here are (1, 1) but should be (K,) as rhs is (N, K)
        # See issue dask/dask#6516
        coeffs, residuals, _, _ = da.linalg.lstsq(lhs_da, rhs)

    if out_shape is not None:
        coeffs = reshape_blockwise(
            coeffs,
            shape=(coeffs.shape[0], *out_shape[1:]),
            chunks=((coeffs.shape[0],), *reshape_chunks[1:]),
        )
        residuals = reshape_blockwise(
            residuals, shape=out_shape[1:], chunks=reshape_chunks[1:]
        )

    return coeffs, residuals


def gls_least_squares(
    x, cov, chol, rhs, order, rcond=None, skipna=False, scaled=True, name=None
):
    da = get_chunked_array_type(rhs).array_api

    # Same reshaping trick as least_squares above: fold all non-core
    # dimensions into a single axis and undo the transform afterwards.
    added_dim = rhs.ndim == 1
    if added_dim:
        rhs = rhs.reshape(rhs.shape[0], 1)
    if rhs.ndim > 2:
        out_shape = rhs.shape
        reshape_chunks = rhs.chunks
        rhs = reshape_blockwise(rhs, (rhs.shape[0], math.prod(rhs.shape[1:])))
    else:
        out_shape = None

    func = partial(
        nputils._gls_polyfit_1d,
        x=x,
        cov=cov,
        chol=chol,
        order=order,
        rcond=rcond,
        skipna=skipna,
        scaled=scaled,
        name=name,
    )
    width = order + 1 + order * order
    results = da.apply_along_axis(func, 0, rhs, dtype=float, shape=(width,))
    coeffs = results[:order]
    residuals = results[order]
    covariance = results[order + 1 :]

    trailing = out_shape[1:] if out_shape is not None else rhs.shape[1:]
    trailing_chunks = reshape_chunks[1:] if out_shape is not None else rhs.chunks[1:]
    covariance = reshape_blockwise(
        covariance,
        shape=(order, order, *trailing),
        chunks=((order,), (order,), *trailing_chunks),
    )
    if out_shape is not None:
        coeffs = reshape_blockwise(
            coeffs,
            shape=(order, *out_shape[1:]),
            chunks=((order,), *reshape_chunks[1:]),
        )
        residuals = reshape_blockwise(
            residuals, shape=out_shape[1:], chunks=reshape_chunks[1:]
        )
    if added_dim:
        coeffs = coeffs.reshape(coeffs.shape[0])
        residuals = residuals.reshape(())
        covariance = covariance.reshape(order, order)

    return coeffs, residuals, covariance


def _fill_with_last_one(a, b):
    import numpy as np

    # cumreduction apply the push func over all the blocks first so,
    # the only missing part is filling the missing values using the
    # last data of the previous chunk
    return np.where(np.isnan(b), a, b)


def _dtype_push(a, axis, dtype=None):
    from xarray.core.duck_array_ops import _push

    # Not sure why the blelloch algorithm force to receive a dtype
    return _push(a, axis=axis)


def push(array, n, axis, method="blelloch"):
    """
    Dask-aware bottleneck.push
    """
    import numpy as np

    from xarray.core.duck_array_ops import _push
    from xarray.core.nputils import nanlast

    chunkmanager = get_chunked_array_type(array)
    da = chunkmanager.array_api

    if n is not None and all(n <= size for size in array.chunks[axis]):
        return array.map_overlap(_push, depth={axis: (n, 0)}, n=n, axis=axis)

    # TODO: Replace all this function
    #  once https://github.com/pydata/xarray/issues/9229 being implemented

    pushed_array = chunkmanager.scan(
        func=_dtype_push,
        binop=_fill_with_last_one,
        ident=np.nan,
        arr=array,
        axis=axis,
        dtype=array.dtype,
        method=method,
        preop=nanlast,
    )

    if n is not None and 0 < n < array.shape[axis] - 1:
        # The idea is to calculate a cumulative sum of a bitmask
        # created from the isnan method, but every time a False is found the sum
        # must be restarted, and the final result indicates the amount of contiguous
        # nan values found in the original array on every position
        nan_bitmask = da.isnan(array, dtype=int)
        cumsum_nan = nan_bitmask.cumsum(axis=axis, method=method)
        valid_positions = da.where(nan_bitmask == 0, cumsum_nan, np.nan)
        valid_positions = push(valid_positions, None, axis, method=method)
        # All the NaNs at the beginning are converted to 0
        valid_positions = da.nan_to_num(valid_positions)
        valid_positions = cumsum_nan - valid_positions
        valid_positions = valid_positions <= n
        pushed_array = da.where(valid_positions, pushed_array, np.nan)

    return pushed_array
