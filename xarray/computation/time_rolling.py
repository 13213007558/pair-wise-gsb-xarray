"""Time-indexed rolling windows defined by an actual time length.

Unlike DataArray.rolling, which counts a fixed number of samples, the
windows here contain every sample whose time coordinate falls inside an
interval of fixed physical duration (e.g. the past two hours).

The implementation is shared by the DataArray and Dataset entry points
DataArray.rolling_time and Dataset.rolling_time.
"""

from __future__ import annotations

import datetime
from collections.abc import Hashable, Mapping
from numbers import Real
from typing import TYPE_CHECKING, Any, Generic, Literal, Union

import numpy as np
import pandas as pd

from xarray.core.options import _get_keep_attrs
from xarray.core.types import T_Xarray
from xarray.core.utils import is_duck_dask_array

if TYPE_CHECKING:
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset

ClosedOptions = Literal["left", "right", "both", "neither"]
TimeWindowSpec = Union[str, datetime.timedelta, np.timedelta64, pd.Timedelta, Real]
TimeRollingWeight = Literal["equal", "duration"]


def _duration_to_seconds(value: Any, name: str) -> float:
    """Convert a window/gap specification to seconds."""
    # NB: np.timedelta64 registers as numbers.Real, so check the timedelta
    # types first.
    if isinstance(value, (datetime.timedelta, np.timedelta64, pd.Timedelta)):
        seconds = pd.Timedelta(value).total_seconds()
    elif isinstance(value, Real):
        seconds = float(value)
    elif isinstance(value, str):
        try:
            seconds = pd.Timedelta(value).total_seconds()
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"Could not parse {name}={value!r} as a time duration"
            ) from error
    else:
        raise TypeError(
            f"{name} must be a duration string (e.g. '2h'), timedelta or "
            f"number of seconds, got {value!r}"
        )
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{name} must be a finite, positive duration")
    return seconds


def _times_to_seconds(values: np.ndarray, dim: Hashable) -> np.ndarray:
    """Convert a 1D datetime64 or cftime coordinate to seconds since the
    first label."""
    if values.dtype.kind == "M":
        if np.isnat(values).any():
            raise ValueError(
                f"time coordinate {dim!r} contains NaT; remove NaT values "
                f"before using rolling_time"
            )
        seconds = (values - values[0]) / np.timedelta64(1, "s")
    elif values.dtype.kind == "O":
        # cftime datetimes on the same calendar (or other datetime-like
        # objects): subtraction yields datetime.timedelta objects.
        try:
            deltas = values - values[0]
            seconds = np.array(
                [delta.total_seconds() for delta in deltas], dtype=np.float64
            )
        except (TypeError, AttributeError) as error:
            raise TypeError(
                f"coordinate {dim!r} must have datetime64 or cftime values; "
                f"the object-dtype coordinate is not datetime-like"
            ) from error
    else:
        raise TypeError(
            f"coordinate {dim!r} must be datetime64 or cftime to use "
            f"rolling_time, got dtype {values.dtype!r}"
        )

    seconds = np.asarray(seconds, dtype=np.float64)
    if np.isnan(seconds).any():
        raise ValueError(
            f"time coordinate {dim!r} contains missing labels; remove them "
            f"before using rolling_time"
        )
    if (np.diff(seconds) < 0).any():
        raise ValueError(
            f"time coordinate {dim!r} must be monotonically non-decreasing"
        )
    return seconds


def _interval_starts(times: np.ndarray, max_gap: float | None) -> np.ndarray:
    """Start of the interval represented by each sample.

    Piecewise-constant convention: the sample at t[i] is held constant over
    the interval since the previous sample, i.e. (t[i - 1], t[i]]; the
    first sample covers the interval back to the start of any window. With
    max_gap set, an interval is capped at t[i] - max_gap so that gaps
    between samples cannot extend coverage indefinitely.
    """
    starts = np.empty_like(times)
    starts[0] = -np.inf
    starts[1:] = times[:-1]
    if max_gap is not None:
        np.maximum(starts, times - max_gap, out=starts)
    return starts


def _cumsum0(values: np.ndarray) -> np.ndarray:
    """Cumulative sum along the last axis with an explicit leading zero."""
    out = np.empty(values.shape[:-1] + (values.shape[-1] + 1,), dtype=values.dtype)
    out[..., 0] = 0
    np.cumsum(values, axis=-1, out=out[..., 1:])
    return out


def _rolling_time_reduce(
    data: np.ndarray,
    times: np.ndarray,
    starts: np.ndarray,
    *,
    axis: int,
    window: float,
    closed: ClosedOptions,
    include_current: bool,
    min_periods: int,
    min_coverage: float,
    weight: TimeRollingWeight,
    op: Literal["mean", "sum", "count"],
) -> np.ndarray:
    """Numpy kernel: time-window reductions along axis.

    times and starts are 1D float arrays (seconds since the first label)
    aligned with data.shape[axis], non-decreasing in times.
    """
    data = np.moveaxis(data, axis, -1)
    original_shape = data.shape
    n = original_shape[-1]

    out_dtype = np.complex128 if data.dtype.kind == "c" else np.float64
    if data.dtype.kind not in "buifc":
        raise TypeError(
            f"rolling_time reductions require numeric data, got dtype {data.dtype!r}"
        )
    flat = np.ascontiguousarray(data).reshape(-1, n).astype(out_dtype)

    lower = times - window
    left_side: Literal["left", "right"] = (
        "left" if closed in ("left", "both") else "right"
    )
    right_side: Literal["left", "right"] = (
        "right" if (closed in ("right", "both") and include_current) else "left"
    )

    first = np.searchsorted(times, lower, side=left_side)
    last = np.searchsorted(times, times, side=right_side)

    # First sample whose own interval starts at or after the window start;
    # earlier members cover back to the window start (minus max_gap).
    split = np.searchsorted(starts, lower, side="left")
    np.clip(split, first, last, out=split)

    valid = ~np.isnan(flat) if flat.dtype.kind == "c" else np.isfinite(flat)
    valid_f = valid.astype(np.float64)
    x = np.where(valid, flat, 0)

    # Interval lengths; entries with a -inf start are never in the
    # [split, last) range where this term is used, so zero them out to
    # keep the cumulative sums finite.
    durations = times - starts
    durations = np.where(np.isfinite(durations), durations, 0.0)

    count_csum = _cumsum0(valid_f)
    counts = count_csum[..., last] - count_csum[..., first]

    # Total duration covered by valid samples inside each window.
    vt_csum = _cumsum0(valid_f * times)
    vts_csum = _cumsum0(valid_f * durations)
    with np.errstate(invalid="ignore"):
        coverage = (
            vt_csum[..., split]
            - vt_csum[..., first]
            - lower * (count_csum[..., split] - count_csum[..., first])
            + vts_csum[..., last]
            - vts_csum[..., split]
        )

    if op == "count":
        result = counts.copy()
        result[counts < min_periods] = np.nan
        out = result.reshape(original_shape)
        return np.moveaxis(out, -1, axis)

    value_csum = _cumsum0(x)
    if weight == "equal":
        values_sum = value_csum[..., last] - value_csum[..., first]
        numerator = values_sum
        denominator: np.ndarray = counts
    else:
        xt_csum = _cumsum0(x * times)
        xts_csum = _cumsum0(x * durations)
        with np.errstate(invalid="ignore"):
            weighted_sum = (
                xt_csum[..., split]
                - xt_csum[..., first]
                - lower * (value_csum[..., split] - value_csum[..., first])
                + xts_csum[..., last]
                - xts_csum[..., split]
            )
        numerator = weighted_sum
        denominator = coverage

    if op == "sum":
        result = numerator.copy()
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            result = numerator / denominator
        result = np.where(denominator > 0, result, np.nan)

    result[counts < min_periods] = np.nan
    if min_coverage > 0:
        # Tolerance absorbs floating-point round-off on otherwise full coverage.
        result[coverage < min_coverage * window - 1e-9 * window] = np.nan

    out = result.reshape(original_shape)
    return np.moveaxis(out, -1, axis)


def _required_overlap(times: np.ndarray, window: float, chunk_sizes: tuple) -> int:
    """Earlier samples a block may need from its left neighbour."""
    boundaries = np.cumsum(np.asarray(chunk_sizes, dtype=np.int64))[:-1]
    if boundaries.size == 0:
        return 0
    lower = times[boundaries] - window
    # "left" is the smallest possible first index across closed settings.
    needed_from = np.searchsorted(times, lower, side="left")
    return int(np.max(boundaries - needed_from))


def _dask_rolling_time_reduce(
    data,
    times: np.ndarray,
    starts: np.ndarray,
    *,
    axis: int,
    out_dtype: np.dtype,
    window: float,
    closed: ClosedOptions,
    include_current: bool,
    min_periods: int,
    min_coverage: float,
    weight: TimeRollingWeight,
    op: Literal["mean", "sum", "count"],
):
    import dask.array as da

    chunk_sizes = data.chunks[axis]
    overlap = _required_overlap(times, window, chunk_sizes)

    times_da = da.from_array(times, chunks=chunk_sizes)
    starts_da = da.from_array(starts, chunks=chunk_sizes)

    depths = [
        {axis: (overlap, 0)},
        {0: (overlap, 0)},
        {0: (overlap, 0)},
    ]

    def _block(data_block, times_block, starts_block):
        return _rolling_time_reduce(
            data_block,
            times_block,
            starts_block,
            axis=axis,
            window=window,
            closed=closed,
            include_current=include_current,
            min_periods=min_periods,
            min_coverage=min_coverage,
            weight=weight,
            op=op,
        )

    return da.map_overlap(
        _block,
        data,
        times_da,
        starts_da,
        depth=depths,
        boundary="none",
        trim=True,
        meta=np.empty((0,) * data.ndim, dtype=out_dtype),
    )


class TimeRolling(Generic[T_Xarray]):
    """Rolling windows whose width is an actual time duration.

    See Also
    --------
    xarray.DataArray.rolling_time
    xarray.Dataset.rolling_time
    """

    __slots__ = (
        "obj",
        "dim",
        "window",
        "closed",
        "include_current",
        "max_gap",
        "min_coverage",
        "min_periods",
        "_times",
        "_starts",
    )

    obj: T_Xarray
    dim: Hashable
    window: float
    closed: ClosedOptions
    include_current: bool
    max_gap: float | None
    min_coverage: float
    min_periods: int
    _times: np.ndarray
    _starts: np.ndarray

    def __init__(
        self,
        obj: T_Xarray,
        windows: Mapping[Hashable, TimeWindowSpec],
        *,
        closed: ClosedOptions = "right",
        include_current: bool = True,
        max_gap: TimeWindowSpec | None = None,
        min_coverage: float = 0.0,
        min_periods: int = 1,
    ) -> None:
        if len(windows) != 1:
            raise ValueError(
                f"rolling_time supports a single dimension, got {tuple(windows)!r}"
            )
        ((dim, window_spec),) = windows.items()
        if dim not in obj.dims:
            raise KeyError(
                f"Window dimension {dim!r} not found in "
                f"{obj.__class__.__name__} dimensions {tuple(obj.dims)}"
            )
        if dim not in obj.coords:
            raise ValueError(
                f"rolling_time requires a 1D time coordinate along {dim!r}"
            )
        coord = obj.coords[dim]
        if coord.ndim != 1 or coord.dims != (dim,):
            raise ValueError(
                f"rolling_time requires a 1D time coordinate along {dim!r}"
            )
        if closed not in ("left", "right", "both", "neither"):
            raise ValueError(
                "closed must be one of 'left', 'right', 'both', 'neither', "
                f"got {closed!r}"
            )
        if not 0.0 <= min_coverage <= 1.0:
            raise ValueError(
                f"min_coverage must be between 0 and 1, got {min_coverage!r}"
            )
        if min_periods < 1:
            raise ValueError("min_periods must be a positive integer")

        self.obj = obj
        self.dim = dim
        self.closed = closed
        self.include_current = include_current
        self.min_coverage = float(min_coverage)
        self.min_periods = int(min_periods)
        self.window = _duration_to_seconds(window_spec, "window")
        self.max_gap = (
            None if max_gap is None else _duration_to_seconds(max_gap, "max_gap")
        )

        times = _times_to_seconds(np.asarray(coord.values), dim)
        self._times = times
        self._starts = _interval_starts(times, self.max_gap)

    def __repr__(self) -> str:
        gap = "" if self.max_gap is None else f", max_gap={self.max_gap}s"
        return (
            f"{self.__class__.__name__} [{self.dim}->{self.window}s, "
            f"closed={self.closed!r}, "
            f"include_current={self.include_current}{gap}]"
        )

    @property
    def ndim(self) -> int:
        return 1

    def _get_keep_attrs(self, keep_attrs: bool | None) -> bool:
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)
        return keep_attrs

    def _validate_weight(self, weight: TimeRollingWeight) -> TimeRollingWeight:
        if weight not in ("equal", "duration"):
            raise ValueError(
                "weight must be 'equal' (one weight per sample) or "
                f"'duration' (piecewise-constant durations), got {weight!r}"
            )
        return weight


def _reduce_dataarray(
    da: DataArray,
    rolling: TimeRolling[Any],
    *,
    op: Literal["mean", "sum", "count"],
    weight: TimeRollingWeight,
    keep_attrs: bool,
) -> DataArray:
    axis = da.get_axis_num(rolling.dim)
    data = da.variable.data
    out_dtype = (
        np.dtype(np.complex128) if da.dtype.kind == "c" else np.dtype(np.float64)
    )

    if is_duck_dask_array(data):
        new_data = _dask_rolling_time_reduce(
            data,
            rolling._times,
            rolling._starts,
            axis=axis,
            out_dtype=out_dtype,
            window=rolling.window,
            closed=rolling.closed,
            include_current=rolling.include_current,
            min_periods=rolling.min_periods,
            min_coverage=rolling.min_coverage,
            weight=weight,
            op=op,
        )
    else:
        new_data = _rolling_time_reduce(
            np.asarray(data),
            rolling._times,
            rolling._starts,
            axis=axis,
            window=rolling.window,
            closed=rolling.closed,
            include_current=rolling.include_current,
            min_periods=rolling.min_periods,
            min_coverage=rolling.min_coverage,
            weight=weight,
            op=op,
        )

    result = da.copy(data=new_data)
    result.attrs = da.attrs if keep_attrs else {}
    return result


class DataArrayTimeRolling(TimeRolling["DataArray"]):
    """Time-duration rolling windows over a DataArray.

    Construct via DataArray.rolling_time.
    """

    __slots__ = ()

    def _reduce(
        self,
        op: Literal["mean", "sum", "count"],
        weight: TimeRollingWeight,
        keep_attrs: bool | None,
    ) -> DataArray:
        keep_attrs = self._get_keep_attrs(keep_attrs)
        return _reduce_dataarray(
            self.obj,
            self,
            op=op,
            weight=self._validate_weight(weight),
            keep_attrs=keep_attrs,
        )

    def mean(
        self,
        *,
        weight: TimeRollingWeight = "equal",
        keep_attrs: bool | None = None,
    ) -> DataArray:
        """Weighted mean of the samples in each time window.

        Parameters
        ----------
        weight : {"equal", "duration"}, default: "equal"
            "equal" gives every sample one weight; "duration" weights each
            sample by how long it is held constant inside the window (see
            the rolling_time docstring for the interval convention).
        keep_attrs : bool, optional
            If True, copy attributes from the original object.

        Returns
        -------
        result : DataArray
        """
        return self._reduce("mean", weight, keep_attrs)

    def sum(
        self,
        *,
        weight: TimeRollingWeight = "equal",
        keep_attrs: bool | None = None,
    ) -> DataArray:
        """Weighted sum of the samples in each time window.

        With weight="duration" this is the piecewise-constant integral over
        the covered duration.

        Parameters
        ----------
        weight : {"equal", "duration"}, default: "equal"
            "equal" gives every sample one weight; "duration" weights each
            sample by how long it is held constant inside the window.
        keep_attrs : bool, optional
            If True, copy attributes from the original object.

        Returns
        -------
        result : DataArray
        """
        return self._reduce("sum", weight, keep_attrs)

    def count(self, keep_attrs: bool | None = None) -> DataArray:
        """Number of valid samples in each time window.

        Windows with fewer than min_periods valid samples are NaN.
        """
        return self._reduce("count", "equal", keep_attrs)


class DatasetTimeRolling(TimeRolling["Dataset"]):
    """Time-duration rolling windows over a Dataset.

    Construct via Dataset.rolling_time.
    """

    __slots__ = ()

    def _reduce(
        self,
        op: Literal["mean", "sum", "count"],
        weight: TimeRollingWeight,
        keep_attrs: bool | None,
    ) -> Dataset:
        from xarray.core.dataset import Dataset

        keep_attrs = self._get_keep_attrs(keep_attrs)
        weight = self._validate_weight(weight)

        reduced = {}
        for name, da in self.obj.data_vars.items():
            if self.dim in da.dims:
                reduced[name] = _reduce_dataarray(
                    da,
                    self,
                    op=op,
                    weight=weight,
                    keep_attrs=keep_attrs,
                )
            else:
                reduced[name] = da.copy()
                if not keep_attrs:
                    reduced[name].attrs = {}

        attrs = self.obj.attrs if keep_attrs else {}
        return Dataset(reduced, coords=self.obj.coords, attrs=attrs)

    def mean(
        self,
        *,
        weight: TimeRollingWeight = "equal",
        keep_attrs: bool | None = None,
    ) -> Dataset:
        """Weighted mean of the samples in each time window."""
        return self._reduce("mean", weight, keep_attrs)

    def sum(
        self,
        *,
        weight: TimeRollingWeight = "equal",
        keep_attrs: bool | None = None,
    ) -> Dataset:
        """Weighted sum of the samples in each time window."""
        return self._reduce("sum", weight, keep_attrs)

    def count(self, keep_attrs: bool | None = None) -> Dataset:
        """Number of valid samples in each time window."""
        return self._reduce("count", "equal", keep_attrs)
