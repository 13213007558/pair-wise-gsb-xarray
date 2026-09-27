from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import xarray as xr
from xarray.tests import (
    assert_allclose,
    assert_identical,
    requires_cftime,
    requires_dask,
)


def reference_reduce(
    times_s: np.ndarray,
    values: np.ndarray,
    *,
    window: float,
    closed: str,
    include_current: bool,
    max_gap: float | None,
    min_coverage: float,
    min_periods: int,
    weight: str,
    op: str,
) -> np.ndarray:
    """Small, deliberately naive per-window reference implementation."""
    n = len(times_s)
    starts = np.empty(n)
    starts[0] = -np.inf
    starts[1:] = times_s[:-1]
    if max_gap is not None:
        starts = np.maximum(starts, times_s - max_gap)

    out = np.full(n, np.nan)
    for j in range(n):
        lo = times_s[j] - window
        hi = times_s[j]
        members = []
        for i in range(n):
            t = times_s[i]
            left_ok = t >= lo if closed in ("left", "both") else t > lo
            if closed in ("right", "both") and include_current:
                right_ok = t <= hi
            else:
                right_ok = t < hi
            if left_ok and right_ok:
                members.append(i)

        vals = values[members]
        valid = ~np.isnan(vals)
        count = int(valid.sum())
        cov = np.array([min(times_s[i], hi) - max(lo, starts[i]) for i in members])
        wcov = float(cov[valid].sum())

        if op == "count":
            out[j] = np.nan if count < min_periods else float(count)
            continue

        masked = count < min_periods
        if min_coverage > 0:
            masked = masked or wcov < min_coverage * window - 1e-9 * window
        if masked:
            out[j] = np.nan
        elif weight == "equal":
            total = float(np.nansum(vals)) if members else 0.0
            out[j] = total if op == "sum" else (total / count if count else np.nan)
        else:
            wsum = float(np.sum(vals[valid] * cov[valid]))
            out[j] = wsum if op == "sum" else (wsum / wcov if wcov > 0 else np.nan)
    return out


def irregular_seconds(n: int = 40, seed: int = 0) -> np.ndarray:
    """Non-decreasing irregular sample times in seconds (5min to 3h gaps)."""
    rng = np.random.default_rng(seed)
    gaps = rng.uniform(300.0, 3 * 3600.0, size=n - 1)
    return np.concatenate([[0.0], np.cumsum(gaps)])


def irregular_values(n: int = 40, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=n)
    values[rng.random(n) < 0.15] = np.nan
    return values


def datetime64_index(times_s: np.ndarray) -> pd.DatetimeIndex:
    return pd.Timestamp("2001-01-01") + pd.to_timedelta(times_s, unit="s")


@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
@pytest.mark.parametrize("include_current", [True, False])
@pytest.mark.parametrize("weight", ["equal", "duration"])
@pytest.mark.parametrize("op", ["mean", "sum", "count"])
def test_matches_reference_datetime64(closed, include_current, weight, op):
    times_s = irregular_seconds()
    values = irregular_values()
    da = xr.DataArray(values, coords={"time": datetime64_index(times_s)}, dims="time")

    kwargs = dict(
        window=2 * 3600.0,
        closed=closed,
        include_current=include_current,
        max_gap=None,
        min_coverage=0.0,
        min_periods=1,
        weight=weight,
        op=op,
    )
    rolling = da.rolling_time(time="2h", closed=closed, include_current=include_current)
    result = getattr(rolling, op)(weight=weight) if op != "count" else rolling.count()
    expected = reference_reduce(times_s, values, **kwargs)

    assert result.time.equals(da.time)
    np.testing.assert_allclose(
        result.values, expected, rtol=1e-8, atol=1e-12, equal_nan=True
    )


def hourly_da(values=(0.0, 1.0, 2.0, 3.0, 4.0)) -> xr.DataArray:
    time = pd.date_range("2000-01-01", periods=len(values), freq="h")
    return xr.DataArray(
        np.asarray(values, dtype=float), coords={"time": time}, dims="time"
    )


def test_known_values_equal_weights():
    da = hourly_da()
    rolling = da.rolling_time(time="2h")
    np.testing.assert_allclose(rolling.mean().values, [0.0, 0.5, 1.5, 2.5, 3.5])
    np.testing.assert_allclose(rolling.sum().values, [0.0, 1.0, 3.0, 5.0, 7.0])
    np.testing.assert_allclose(rolling.count().values, [1.0, 2.0, 2.0, 2.0, 2.0])


def test_known_values_duration_weights():
    da = hourly_da()
    rolling = da.rolling_time(time="2h")
    # each sample covers the hour since the previous sample (the first one
    # covers back to the window start)
    np.testing.assert_allclose(
        rolling.sum(weight="duration").values,
        [0.0, 3600.0, 10800.0, 18000.0, 25200.0],
    )
    # uniform sampling: duration-weighted mean equals the equal-weighted one
    np.testing.assert_allclose(
        rolling.mean(weight="duration").values, [0.0, 0.5, 1.5, 2.5, 3.5]
    )


@pytest.mark.parametrize(
    ("closed", "include_current", "expected"),
    [
        ("right", True, [0.0, 0.5, 1.5, 2.5, 3.5]),  # (t-2h, t]
        ("left", True, [np.nan, 0.0, 0.5, 1.5, 2.5]),  # [t-2h, t)
        ("both", True, [0.0, 0.5, 1.0, 2.0, 3.0]),  # [t-2h, t]
        ("neither", True, [np.nan, 0.0, 1.0, 2.0, 3.0]),  # (t-2h, t)
        ("right", False, [np.nan, 0.0, 1.0, 2.0, 3.0]),  # (t-2h, t)
        ("both", False, [np.nan, 0.0, 0.5, 1.5, 2.5]),  # [t-2h, t)
    ],
)
def test_closed_and_include_current(closed, include_current, expected):
    da = hourly_da()
    result = da.rolling_time(
        time="2h", closed=closed, include_current=include_current
    ).mean()
    np.testing.assert_allclose(result.values, expected, equal_nan=True)


def test_duplicate_times():
    times_s = np.array([0.0, 1, 1, 2, 2, 2, 3]) * 3600
    values = np.arange(7.0)
    da = xr.DataArray(values, coords={"time": datetime64_index(times_s)}, dims="time")
    for closed in ["left", "right", "both", "neither"]:
        for include_current in [True, False]:
            result = da.rolling_time(
                time="1h", closed=closed, include_current=include_current
            ).mean()
            expected = reference_reduce(
                times_s,
                values,
                window=3600.0,
                closed=closed,
                include_current=include_current,
                max_gap=None,
                min_coverage=0.0,
                min_periods=1,
                weight="equal",
                op="mean",
            )
            np.testing.assert_allclose(result.values, expected, equal_nan=True)


def test_duplicate_times_zero_duration():
    # the second of two identical timestamps covers no interval
    da = xr.DataArray(
        [np.nan, 5.0],
        coords={"time": datetime64_index(np.array([0.0, 0.0]))},
        dims="time",
    )
    rolling = da.rolling_time(time="1h")
    # equal weights: the valid duplicate still counts (both outputs share
    # the same window, since membership is decided by time)
    np.testing.assert_allclose(rolling.mean().values, [5.0, 5.0])
    # duration weights: the only valid sample covers nothing
    assert np.isnan(rolling.mean(weight="duration").values).all()
    np.testing.assert_allclose(rolling.sum(weight="duration").values, [0.0, 0.0])


def test_empty_windows():
    times_s = np.array([0.0, 10.0, 20.0]) * 3600
    da = xr.DataArray(
        [1.0, 2.0, 3.0], coords={"time": datetime64_index(times_s)}, dims="time"
    )
    # excluding the current sample leaves every window empty
    rolling = da.rolling_time(time="1h", include_current=False)
    assert np.isnan(rolling.mean().values).all()
    assert np.isnan(rolling.sum().values).all()
    assert np.isnan(rolling.mean(weight="duration").values).all()
    # empty windows have zero valid samples, below the default min_periods=1
    assert np.isnan(rolling.count().values).all()
    # including the current sample, each window holds exactly that sample
    result = da.rolling_time(time="1h").mean()
    np.testing.assert_allclose(result.values, [1.0, 2.0, 3.0])


def test_max_gap_caps_coverage():
    # hourly samples, then a 8h gap before the last sample
    times_s = np.array([0.0, 1.0, 2.0, 10.0]) * 3600
    da = xr.DataArray(
        [0.0, 1.0, 2.0, 10.0],
        coords={"time": datetime64_index(times_s)},
        dims="time",
    )
    # without max_gap the last sample covers the whole 3h window
    rolling = da.rolling_time(time="3h")
    np.testing.assert_allclose(
        rolling.sum(weight="duration").values[-1], 10.0 * 3 * 3600
    )
    # with max_gap=1h its coverage is capped at one hour
    rolling = da.rolling_time(time="3h", max_gap="1h")
    np.testing.assert_allclose(rolling.sum(weight="duration").values[-1], 10.0 * 3600)
    # coverage fraction is 1/3, so min_coverage=0.5 masks the window
    rolling = da.rolling_time(time="3h", max_gap="1h", min_coverage=0.5)
    assert np.isnan(rolling.mean(weight="duration").values[-1])
    rolling = da.rolling_time(time="3h", max_gap="1h", min_coverage=0.3)
    np.testing.assert_allclose(rolling.mean(weight="duration").values[-1], 10.0)


def test_min_coverage_masks_undersampled_windows():
    # a single sample per 2h window covers only half of it
    times_s = np.array([0.0, 2.0, 4.0]) * 3600
    da = xr.DataArray(
        [1.0, 2.0, 3.0], coords={"time": datetime64_index(times_s)}, dims="time"
    )
    rolling = da.rolling_time(time="2h", max_gap="1h", min_coverage=0.75)
    result = rolling.mean(weight="duration")
    assert np.isnan(result.values).all()
    rolling = da.rolling_time(time="2h", max_gap="1h", min_coverage=0.5)
    np.testing.assert_allclose(rolling.mean(weight="duration").values, [1.0, 2.0, 3.0])


@requires_cftime
@pytest.mark.parametrize("calendar", ["noleap", "360_day"])
@pytest.mark.parametrize("weight", ["equal", "duration"])
def test_cftime_non_gregorian_matches_reference(calendar, weight):
    import cftime

    cls = {
        "noleap": cftime.DatetimeNoLeap,
        "360_day": cftime.Datetime360Day,
    }[calendar]
    times_s = irregular_seconds(n=25, seed=5)
    times = [
        cls(2001, 1, 1) + pd.Timedelta(seconds=float(s)).to_pytimedelta()
        for s in times_s
    ]
    values = irregular_values(n=25, seed=6)
    da = xr.DataArray(values, coords={"time": np.array(times)}, dims="time")

    result = da.rolling_time(time="90min").mean(weight=weight)
    expected = reference_reduce(
        times_s,
        values,
        window=5400.0,
        closed="right",
        include_current=True,
        max_gap=None,
        min_coverage=0.0,
        min_periods=1,
        weight=weight,
        op="mean",
    )
    np.testing.assert_allclose(
        result.values, expected, rtol=1e-8, atol=1e-12, equal_nan=True
    )
    # the cftime coordinate is preserved
    assert result.time.equals(da.time)


def test_multidim_matches_per_row_reference():
    times_s = irregular_seconds(n=20, seed=7)
    values = np.stack([irregular_values(n=20, seed=8), irregular_values(n=20, seed=9)])
    da = xr.DataArray(
        values,
        coords={"x": [0, 1], "time": datetime64_index(times_s)},
        dims=("x", "time"),
    )
    result = da.rolling_time(time="2h", max_gap="1h").mean(weight="duration")
    for k in range(2):
        expected = reference_reduce(
            times_s,
            values[k],
            window=7200.0,
            closed="right",
            include_current=True,
            max_gap=3600.0,
            min_coverage=0.0,
            min_periods=1,
            weight="duration",
            op="mean",
        )
        np.testing.assert_allclose(
            result.isel(x=k).values,
            expected,
            rtol=1e-8,
            atol=1e-12,
            equal_nan=True,
        )


@requires_dask
@pytest.mark.parametrize("weight", ["equal", "duration"])
@pytest.mark.parametrize("op", ["mean", "sum", "count"])
def test_dask_matches_numpy_across_chunk_boundaries(weight, op):
    # irregular sampling with a large gap in the middle, chunked so that
    # windows reach across chunk boundaries and across the gap
    rng = np.random.default_rng(11)
    dense = np.cumsum(rng.uniform(60.0, 300.0, size=24))
    times_s = np.concatenate([dense, dense[-1] + 20 * 3600 + dense])
    values = irregular_values(n=len(times_s), seed=12)
    time = datetime64_index(times_s)

    da = xr.DataArray(values, coords={"time": time}, dims="time")
    expected = getattr(da.rolling_time(time="2h", max_gap="30min"), op)(
        **({} if op == "count" else {"weight": weight})
    )

    chunked = da.chunk({"time": 7})
    rolling = chunked.rolling_time(time="2h", max_gap="30min")
    result = getattr(rolling, op)(**({} if op == "count" else {"weight": weight}))
    assert result.chunks is not None
    assert_allclose(result.compute(), expected)


@requires_dask
def test_dask_window_spanning_multiple_chunks():
    # dense sampling with small chunks: windows span several chunks, so the
    # overlap is deeper than a single chunk
    times_s = np.arange(48.0) * 600  # 10min sampling
    values = np.sin(np.arange(48.0))
    da = xr.DataArray(values, coords={"time": datetime64_index(times_s)}, dims="time")
    expected = da.rolling_time(time="3h").mean()
    result = da.chunk({"time": 4}).rolling_time(time="3h").mean()
    assert_allclose(result.compute(), expected)


@requires_dask
def test_dask_2d_and_chunk_structure_preserved():
    times_s = irregular_seconds(n=16, seed=13)
    values = irregular_values(n=16, seed=14)[None, :].repeat(3, axis=0)
    da = xr.DataArray(
        values,
        coords={"x": [0, 1, 2], "time": datetime64_index(times_s)},
        dims=("x", "time"),
    )
    chunked = da.chunk({"x": 2, "time": 5})
    result = chunked.rolling_time(time="2h").mean()
    # non-time axes keep their chunking (dask may rechunk the time axis if
    # the required overlap exceeds a chunk)
    assert result.chunks[0] == chunked.chunks[0]
    expected = da.rolling_time(time="2h").mean()
    assert_allclose(result.compute(), expected)


def test_coordinates_and_attrs_preserved():
    times_s = irregular_seconds(n=15, seed=15)
    da = xr.DataArray(
        irregular_values(n=15, seed=16),
        coords={
            "time": datetime64_index(times_s),
            "label": ("time", [f"s{k}" for k in range(15)]),
            "station": "alpha",
        },
        dims="time",
        name="temperature",
        attrs={"units": "degC"},
    )
    result = da.rolling_time(time="2h").mean(keep_attrs=True)
    assert result.attrs == {"units": "degC"}
    assert result.name == "temperature"
    assert_identical(result.coords.to_dataset(), da.coords.to_dataset())

    result = da.rolling_time(time="2h").mean(keep_attrs=False)
    assert result.attrs == {}

    # default follows the global keep_attrs option (default True)
    result = da.rolling_time(time="2h").mean()
    assert result.attrs == {"units": "degC"}
    with xr.set_options(keep_attrs=False):
        result = da.rolling_time(time="2h").mean()
        assert result.attrs == {}


def test_dataset_rolling_time():
    times_s = irregular_seconds(n=15, seed=17)
    time = datetime64_index(times_s)
    ds = xr.Dataset(
        {
            "a": ("time", irregular_values(n=15, seed=18)),
            "b": ("time", irregular_values(n=15, seed=19)),
            "c": ("x", np.arange(4.0)),
        },
        coords={"time": time},
        attrs={"title": "example"},
    )
    ds["a"].attrs = {"units": "m"}

    result = ds.rolling_time(time="2h").mean()
    assert result.attrs == {"title": "example"}
    assert result["a"].attrs == {"units": "m"}
    # variables without the time dimension are passed through
    assert_identical(result["c"].drop_attrs(), ds["c"].drop_attrs())
    for var in ["a", "b"]:
        expected = ds[var].rolling_time(time="2h").mean()
        assert_identical(result[var], expected)

    result = ds.rolling_time(time="2h").sum(keep_attrs=False)
    assert result.attrs == {}
    assert result["a"].attrs == {}


def test_min_periods():
    da = hourly_da()
    # a 2h right-closed window over hourly data holds at most 2 samples
    result = da.rolling_time(time="2h", min_periods=3).mean()
    assert np.isnan(result.values).all()
    result = da.rolling_time(time="2h", min_periods=2).mean()
    np.testing.assert_allclose(
        result.values, [np.nan, 0.5, 1.5, 2.5, 3.5], equal_nan=True
    )


def test_window_spec_types():
    da = hourly_da()
    expected = da.rolling_time(time="2h").mean()
    import datetime

    for spec in [
        "2h",
        "120min",
        datetime.timedelta(hours=2),
        np.timedelta64(2, "h"),
        pd.Timedelta("2h"),
        7200,
        7200.0,
    ]:
        result = da.rolling_time(time=spec).mean()
        assert_identical(result, expected)


def test_non_nanosecond_datetime64():
    time = np.array(
        ["2001-01-01", "2001-01-01T01:00", "2001-01-01T02:00"],
        dtype="datetime64[s]",
    )
    da = xr.DataArray([1.0, 2.0, 3.0], coords={"time": time}, dims="time")
    result = da.rolling_time(time="90min").mean()
    np.testing.assert_allclose(result.values, [1.0, 1.5, 2.5])


def test_errors():
    da = hourly_da()
    with pytest.raises(ValueError, match="single dimension"):
        da.rolling_time({"time": "2h", "x": "1h"})
    with pytest.raises(KeyError, match="not found"):
        da.rolling_time(nope="2h")
    with pytest.raises(ValueError, match="closed"):
        da.rolling_time(time="2h", closed="middle")
    with pytest.raises(ValueError, match="min_coverage"):
        da.rolling_time(time="2h", min_coverage=1.5)
    with pytest.raises(ValueError, match="min_periods"):
        da.rolling_time(time="2h", min_periods=0)
    with pytest.raises(ValueError, match="positive"):
        da.rolling_time(time="0h")
    with pytest.raises(ValueError, match="positive"):
        da.rolling_time(time="2h", max_gap="-1h")
    with pytest.raises(ValueError, match="parse"):
        da.rolling_time(time="not-a-duration")
    with pytest.raises(ValueError, match="weight"):
        da.rolling_time(time="2h").mean(weight="sample")

    # no coordinate along the dimension
    da_nocoord = xr.DataArray(np.arange(5.0), dims="time")
    with pytest.raises(ValueError, match="1D time coordinate"):
        da_nocoord.rolling_time(time="2h")

    # non-datetime coordinate
    da_float = xr.DataArray(
        np.arange(5.0), coords={"time": np.arange(5.0)}, dims="time"
    )
    with pytest.raises(TypeError, match="datetime64 or cftime"):
        da_float.rolling_time(time="2h")

    # non-monotonic coordinate
    time = pd.date_range("2000-01-01", periods=5, freq="h")[[0, 2, 1, 3, 4]]
    da_unsorted = xr.DataArray(np.arange(5.0), coords={"time": time}, dims="time")
    with pytest.raises(ValueError, match="non-decreasing"):
        da_unsorted.rolling_time(time="2h")

    # NaT in the coordinate
    time_nat = pd.date_range("2000-01-01", periods=5, freq="h").values
    time_nat = time_nat.astype("datetime64[ns]")
    time_nat[2] = np.datetime64("NaT", "ns")
    da_nat = xr.DataArray(np.arange(5.0), coords={"time": time_nat}, dims="time")
    with pytest.raises(ValueError, match="NaT"):
        da_nat.rolling_time(time="2h")

    # non-numeric data
    da_text = xr.DataArray(list("abcde"), coords={"time": da.time}, dims="time")
    with pytest.raises(TypeError, match="numeric"):
        da_text.rolling_time(time="2h").mean()


@pytest.mark.parametrize("max_gap", [None, "30min"])
@pytest.mark.parametrize("min_coverage", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("weight", ["equal", "duration"])
def test_matches_reference_gap_and_coverage(max_gap, min_coverage, weight):
    times_s = irregular_seconds(n=30, seed=3)
    values = irregular_values(n=30, seed=4)
    da = xr.DataArray(values, coords={"time": datetime64_index(times_s)}, dims="time")

    max_gap_s = None if max_gap is None else 1800.0
    rolling = da.rolling_time(time="2h", max_gap=max_gap, min_coverage=min_coverage)
    result = rolling.mean(weight=weight)
    expected = reference_reduce(
        times_s,
        values,
        window=7200.0,
        closed="right",
        include_current=True,
        max_gap=max_gap_s,
        min_coverage=min_coverage,
        min_periods=1,
        weight=weight,
        op="mean",
    )
    np.testing.assert_allclose(
        result.values, expected, rtol=1e-8, atol=1e-12, equal_nan=True
    )
