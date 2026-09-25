from __future__ import annotations

import numpy as np
import pytest

import xarray as xr
from xarray.core.conservative import _overlap_weights, conservative_remap
from xarray.tests import assert_allclose, assert_identical, requires_dask


def reference_remap(values, src_bounds, tgt_bounds, data_kind):
    """Independent dense O(n_target * n_source) reference implementation."""
    values = np.asarray(values, dtype=float)
    src = np.asarray(src_bounds, dtype=float)
    tgt = np.asarray(tgt_bounds, dtype=float)
    out = np.full(len(tgt), np.nan)
    coverage = np.zeros(len(tgt))
    for t, (lo_t, hi_t) in enumerate(tgt):
        numerator = 0.0
        covered = 0.0
        for s, (lo_s, hi_s) in enumerate(src):
            overlap = min(hi_t, hi_s) - max(lo_t, lo_s)
            if overlap <= 0 or np.isnan(values[s]):
                continue
            covered += overlap
            if data_kind == "total":
                numerator += overlap * values[s] / (hi_s - lo_s)
            else:
                numerator += overlap * values[s]
        if covered > 0:
            out[t] = numerator if data_kind == "total" else numerator / covered
        coverage[t] = covered / (hi_t - lo_t)
    return out, coverage


def random_edges(n, rng):
    return np.concatenate([[0.0], np.cumsum(rng.uniform(0.5, 2.0, size=n))])


def edges_to_pairs(edges):
    return np.column_stack([edges[:-1], edges[1:]])


@pytest.fixture
def grids():
    rng = np.random.default_rng(42)
    src_edges = random_edges(7, rng)
    tgt_edges = random_edges(5, rng) * src_edges[-1] / 1.0
    # rescale target to exactly span the source domain (full coverage)
    tgt_edges = tgt_edges * src_edges[-1] / tgt_edges[-1]
    values = rng.uniform(0.5, 3.0, size=7)
    return values, edges_to_pairs(src_edges), edges_to_pairs(tgt_edges)


@pytest.mark.parametrize("data_kind", ["total", "density"])
def test_full_coverage_matches_reference_and_conserves(grids, data_kind):
    values, src, tgt = grids
    da = xr.DataArray(values, dims="x")
    result, coverage = conservative_remap(
        da, src, tgt, dim="x", data_kind=data_kind, return_coverage=True
    )
    expected, expected_cov = reference_remap(values, src, tgt, data_kind)
    assert_allclose(result, xr.DataArray(expected, dims="x"))
    assert_allclose(coverage, xr.DataArray(expected_cov, dims="x"))
    assert np.all(coverage.values == 1.0)
    if data_kind == "total":
        # total quantity over the domain is conserved
        assert result.sum().item() == pytest.approx(values.sum())
    else:
        # the integral of the density over the domain is conserved
        src_lengths = src[:, 1] - src[:, 0]
        tgt_lengths = tgt[:, 1] - tgt[:, 0]
        assert (result.values * tgt_lengths).sum() == pytest.approx(
            (values * src_lengths).sum()
        )


def test_edges_form_matches_pairs_form(grids):
    values, src, tgt = grids
    src_edges = np.concatenate([src[:, 0], src[-1:, 1]])
    tgt_edges = np.concatenate([tgt[:, 0], tgt[-1:, 1]])
    da = xr.DataArray(values, dims="x")
    from_pairs = conservative_remap(da, src, tgt, dim="x")
    from_edges = conservative_remap(da, src_edges, tgt_edges, dim="x")
    assert_identical(from_pairs, from_edges)


def test_partial_coverage():
    values = np.array([1.0, 2.0, 3.0])
    src = edges_to_pairs(np.array([0.0, 1.0, 2.0, 3.0]))
    # target extends beyond the source on both sides and has a cell fully
    # outside the source domain
    tgt = edges_to_pairs(np.array([-1.0, 0.5, 2.5, 3.0, 4.0]))
    da = xr.DataArray(values, dims="x")
    result, coverage = conservative_remap(
        da, src, tgt, dim="x", data_kind="total", return_coverage=True
    )
    expected, expected_cov = reference_remap(values, src, tgt, "total")
    assert_allclose(result, xr.DataArray(expected, dims="x"))
    assert_allclose(coverage, xr.DataArray(expected_cov, dims="x"))
    # first target cell [-1, 0.5] is covered over [0, 0.5] only
    assert coverage.values[0] == pytest.approx(0.5 / 1.5)
    # target cell [2.5, 3.0] is fully covered
    assert coverage.values[2] == pytest.approx(1.0)
    # target cell [3.0, 4.0] is fully outside the source domain
    assert coverage.values[3] == 0.0
    assert np.isnan(result.values[3])


def test_partial_coverage_density():
    values = np.array([2.0, 4.0])
    src = edges_to_pairs(np.array([0.0, 1.0, 3.0]))
    tgt = edges_to_pairs(np.array([-1.0, 0.0, 2.0, 5.0]))
    da = xr.DataArray(values, dims="x")
    result, coverage = conservative_remap(
        da, src, tgt, dim="x", data_kind="density", return_coverage=True
    )
    expected, expected_cov = reference_remap(values, src, tgt, "density")
    assert_allclose(result, xr.DataArray(expected, dims="x"))
    assert_allclose(coverage, xr.DataArray(expected_cov, dims="x"))
    # density is averaged over the covered part only
    assert np.isnan(result.values[0])
    assert coverage.values[0] == 0.0
    assert result.values[1] == pytest.approx((1.0 * 2.0 + 1.0 * 4.0) / 2.0)
    assert result.values[2] == pytest.approx(4.0)
    assert coverage.values[2] == pytest.approx(1.0 / 3.0)


def test_missing_values():
    values = np.array([1.0, np.nan, 3.0])
    src = edges_to_pairs(np.array([0.0, 1.0, 2.0, 3.0]))
    tgt = edges_to_pairs(np.array([0.0, 3.0]))
    da = xr.DataArray(values, dims="x")
    result, coverage = conservative_remap(
        da, src, tgt, dim="x", data_kind="total", return_coverage=True
    )
    # the NaN source cell contributes neither mass nor coverage
    assert result.values[0] == pytest.approx(1.0 + 3.0)
    assert coverage.values[0] == pytest.approx(2.0 / 3.0)
    expected, _ = reference_remap(values, src, tgt, "total")
    assert_allclose(result, xr.DataArray(expected, dims="x"))


def test_missing_values_density():
    values = np.array([2.0, np.nan])
    src = edges_to_pairs(np.array([0.0, 1.0, 3.0]))
    tgt = edges_to_pairs(np.array([0.0, 3.0]))
    da = xr.DataArray(values, dims="x")
    result, coverage = conservative_remap(
        da, src, tgt, dim="x", data_kind="density", return_coverage=True
    )
    # only the first unit is covered, with density 2
    assert result.values[0] == pytest.approx(2.0)
    assert coverage.values[0] == pytest.approx(1.0 / 3.0)


def test_descending_bounds(grids):
    values, src, tgt = grids
    da = xr.DataArray(values, dims="x")
    ascending = conservative_remap(da, src, tgt, dim="x", return_coverage=True)
    # reverse both grids: values and results follow the cell order
    src_desc = src[::-1, ::-1]
    tgt_desc = tgt[::-1, ::-1]
    da_desc = xr.DataArray(values[::-1], dims="x")
    descending = conservative_remap(
        da_desc, src_desc, tgt_desc, dim="x", return_coverage=True
    )
    assert_allclose(ascending[0], xr.DataArray(descending[0].values[::-1], dims="x"))
    assert_allclose(ascending[1], xr.DataArray(descending[1].values[::-1], dims="x"))


@pytest.mark.parametrize(
    "bounds",
    [
        np.array([[0.0, 2.0], [1.0, 3.0]]),  # overlapping cells
        np.array([[0.0, 1.0], [1.0, 0.5]]),  # inverted cell
        np.array([[0.0, 1.0], [1.0, 1.0]]),  # zero-width cell
        np.array([[0.0, 1.0], [2.0, 1.0]]),  # inverted cell
        np.array([0.0, 2.0, 1.0]),  # non-monotonic edges
        np.array([0.0, 1.0, 1.0]),  # repeated edges
        np.array([[0.0, np.nan], [1.0, 2.0]]),  # NaN bounds
    ],
)
def test_invalid_bounds_rejected(bounds):
    da = xr.DataArray(np.ones(2), dims="x")
    valid = np.array([[0.0, 1.0], [1.0, 2.0]])
    with pytest.raises(ValueError, match="bounds"):
        conservative_remap(da, bounds, valid, dim="x")
    with pytest.raises(ValueError, match="bounds"):
        conservative_remap(da, valid, bounds, dim="x")


def test_mismatched_source_size_rejected():
    da = xr.DataArray(np.ones(3), dims="x")
    src = edges_to_pairs(np.array([0.0, 1.0, 2.0]))
    tgt = edges_to_pairs(np.array([0.0, 2.0]))
    with pytest.raises(ValueError, match="2 cells"):
        conservative_remap(da, src, tgt, dim="x")


def test_invalid_data_kind_rejected():
    da = xr.DataArray(np.ones(2), dims="x")
    bounds = edges_to_pairs(np.array([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="data_kind"):
        conservative_remap(da, bounds, bounds, dim="x", data_kind="average")


def test_missing_dim_rejected():
    da = xr.DataArray(np.ones(2), dims="x")
    bounds = edges_to_pairs(np.array([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="dimension"):
        conservative_remap(da, bounds, bounds, dim="y")


def test_coord_names_need_not_match_dim_and_other_dims_broadcast():
    rng = np.random.default_rng(0)
    values = rng.uniform(size=(3, 4))
    da = xr.DataArray(
        values,
        dims=("time", "x"),
        coords={"time": [10, 20, 30], "lon": ("x", [1.0, 2.0, 3.0, 4.0])},
        name="mass",
    )
    # bounds carry their own dimension/coordinate names, unrelated to "x"
    src_bounds = xr.DataArray(
        edges_to_pairs(np.array([0.0, 1.0, 2.0, 3.5, 5.0])),
        dims=("cells", "nv"),
        coords={"cells": [100, 101, 102, 103]},
    )
    tgt_bounds = xr.DataArray(
        edges_to_pairs(np.array([0.0, 2.5, 5.0])), dims=("new_cells", "bnds")
    )
    result, coverage = conservative_remap(
        da, src_bounds, tgt_bounds, dim="x", return_coverage=True
    )
    assert result.dims == ("time", "x")
    assert result.sizes == {"time": 3, "x": 2}
    # non-remapped dimensions and their coordinates are preserved
    assert_identical(result["time"], da["time"])
    assert result.name == "mass"
    assert np.all(coverage.values == 1.0)
    # conservation along the remapped dimension for every other slice
    np.testing.assert_allclose(
        result.values.sum(axis=1), values.sum(axis=1), rtol=1e-12
    )
    # per-slice check against the 1D reference
    for i in range(values.shape[0]):
        expected, _ = reference_remap(
            values[i], src_bounds.values, tgt_bounds.values, "total"
        )
        np.testing.assert_allclose(result.values[i], expected, rtol=1e-12)


def test_no_coverage_returns_nan_and_zero():
    da = xr.DataArray([1.0, 2.0], dims="x")
    src = edges_to_pairs(np.array([0.0, 1.0, 2.0]))
    tgt = edges_to_pairs(np.array([5.0, 6.0]))
    result, coverage = conservative_remap(da, src, tgt, dim="x", return_coverage=True)
    assert np.isnan(result.values).all()
    assert (coverage.values == 0.0).all()


def test_overlap_weights_are_sparse():
    # contiguous grids: each target cell touches at most a few source cells,
    # so the number of stored intersections stays O(n_src + n_tgt) and the
    # dense (n_tgt, n_src) matrix is never built
    src_edges = np.linspace(0.0, 1.0, 101)
    tgt_edges = np.linspace(0.0, 1.0, 57)
    cols, vals, row_ptr = _overlap_weights(
        src_edges[:-1], src_edges[1:], tgt_edges[:-1], tgt_edges[1:]
    )
    n_src, n_tgt = 100, 56
    assert len(vals) == row_ptr[-1]
    assert len(vals) <= n_src + n_tgt
    assert len(vals) < n_src * n_tgt // 10
    # every source cell is fully covered: weights per source cell sum to its length
    sums = np.zeros(n_src)
    np.add.at(sums, cols, vals)
    np.testing.assert_allclose(sums, 1.0 / n_src, rtol=1e-12)


@requires_dask
def test_dask_stays_lazy_and_matches_numpy(grids):
    import dask.array as da_mod

    values, src, tgt = grids
    da = xr.DataArray(values, dims="x")
    expected, expected_cov = conservative_remap(
        da, src, tgt, dim="x", return_coverage=True
    )
    dask_da = da.chunk({"x": 3})
    result, coverage = conservative_remap(
        dask_da, src, tgt, dim="x", return_coverage=True
    )
    # weight application is lazy: nothing has been computed yet
    assert isinstance(result.data, da_mod.Array)
    assert isinstance(coverage.data, da_mod.Array)
    assert_identical(result.compute(), expected)
    assert_identical(coverage.compute(), expected_cov)


@requires_dask
def test_dask_broadcast_dimensions(grids):
    import dask.array as da_mod

    values, src, tgt = grids
    da2d = xr.DataArray(
        np.stack([values, 2 * values]), dims=("y", "x")
    ).chunk({"y": 1, "x": 4})
    result = conservative_remap(da2d, src, tgt, dim="x")
    assert isinstance(result.data, da_mod.Array)
    expected = conservative_remap(da2d.compute(), src, tgt, dim="x")
    assert_identical(result.compute(), expected)
