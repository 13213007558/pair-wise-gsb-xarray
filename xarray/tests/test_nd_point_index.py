import numpy as np
import pytest

import xarray as xr
from xarray.indexes import NDPointIndex
from xarray.indexes.nd_point_index import ScipyKDTreeAdapter, TreeAdapter
from xarray.tests import assert_identical, has_scipy, requires_scipy


class BruteForceTreeAdapter(TreeAdapter):
    """Pure NumPy tree adapter used as a brute-force reference in tests."""

    def __init__(self, points, *, options):
        self._points = np.asarray(points, dtype=float)

    def query(self, points):
        distances, indices = self.query_neighbors(points, k=1, radius=np.inf)
        return distances[:, 0], indices[:, 0]

    def query_neighbors(self, points, *, k, radius):
        diff = np.asarray(points, dtype=float)[:, None, :] - self._points[None, :, :]
        dist = np.sqrt((diff**2).sum(axis=-1))

        # stable sort keeps the original point position order for ties
        order = np.argsort(dist, kind="stable", axis=1)
        dist = np.take_along_axis(dist, order, axis=1)

        n_points = self._points.shape[0]
        n_keep = min(k, n_points)
        dist = dist[:, :n_keep]
        indices = order[:, :n_keep]

        missing = (dist > radius) | ~np.isfinite(dist)
        indices = np.where(missing, -1, indices)
        dist = np.where(missing, np.inf, dist)

        if n_keep < k:
            pad = k - n_keep
            dist = np.concatenate(
                [dist, np.full((points.shape[0], pad), np.inf)], axis=1
            )
            indices = np.concatenate(
                [indices, np.full((points.shape[0], pad), -1, dtype=np.intp)], axis=1
            )

        return dist, indices

    def equals(self, other):
        return np.array_equal(self._points, other._points)


class NearestOnlyTreeAdapter(TreeAdapter):
    """Adapter that does not implement the radius-limited neighbor query."""

    def __init__(self, points, *, options):
        self._points = np.asarray(points, dtype=float)

    def query(self, points):
        diff = np.asarray(points, dtype=float)[:, None, :] - self._points[None, :, :]
        dist = np.sqrt((diff**2).sum(axis=-1))
        indices = np.argmin(dist, axis=1)
        return dist[np.arange(dist.shape[0]), indices], indices


@pytest.mark.skipif(has_scipy, reason="requires scipy to be missing")
def test_scipy_kdtree_adapter_missing_scipy():
    points = np.random.rand(4, 2)

    with pytest.raises(ImportError, match=r"scipy"):
        ScipyKDTreeAdapter(points, options={})


@requires_scipy
def test_tree_index_init() -> None:
    from xarray.indexes.nd_point_index import ScipyKDTreeAdapter

    xx, yy = np.meshgrid([1.0, 2.0], [3.0, 4.0])
    ds = xr.Dataset(coords={"xx": (("y", "x"), xx), "yy": (("y", "x"), yy)})

    ds_indexed1 = ds.set_xindex(("xx", "yy"), NDPointIndex)
    assert "xx" in ds_indexed1.xindexes
    assert "yy" in ds_indexed1.xindexes
    assert isinstance(ds_indexed1.xindexes["xx"], NDPointIndex)
    assert ds_indexed1.xindexes["xx"] is ds_indexed1.xindexes["yy"]

    ds_indexed2 = ds.set_xindex(
        ("xx", "yy"), NDPointIndex, tree_adapter_cls=ScipyKDTreeAdapter
    )
    assert ds_indexed1.xindexes["xx"].equals(ds_indexed2.xindexes["yy"])


@requires_scipy
def test_tree_index_init_errors() -> None:
    xx, yy = np.meshgrid([1.0, 2.0], [3.0, 4.0])
    ds = xr.Dataset(coords={"xx": (("y", "x"), xx), "yy": (("y", "x"), yy)})

    ds2 = ds.assign_coords(yy=(("u", "v"), [[3.0, 3.0], [4.0, 4.0]]))

    with pytest.raises(ValueError, match="same dimensions"):
        ds2.set_xindex(("xx", "yy"), NDPointIndex)


@requires_scipy
def test_tree_index_sel() -> None:
    xx, yy = np.meshgrid([1.0, 2.0], [3.0, 4.0])
    ds = xr.Dataset(coords={"xx": (("y", "x"), xx), "yy": (("y", "x"), yy)}).set_xindex(
        ("xx", "yy"), NDPointIndex
    )

    # 1-dimensional labels
    actual = ds.sel(
        xx=xr.Variable("u", [1.1, 1.1, 1.1]),
        yy=xr.Variable("u", [3.1, 3.1, 3.1]),
        method="nearest",
    )
    expected = xr.Dataset(
        coords={"xx": ("u", [1.0, 1.0, 1.0]), "yy": ("u", [3.0, 3.0, 3.0])}
    )
    assert_identical(actual, expected)

    # 2-dimensional labels
    actual = ds.sel(
        xx=xr.Variable(("u", "v"), [[1.1, 1.1, 1.1], [1.9, 1.9, 1.9]]),
        yy=xr.Variable(("u", "v"), [[3.1, 3.1, 3.1], [3.9, 3.9, 3.9]]),
        method="nearest",
    )
    expected = xr.Dataset(
        coords={
            "xx": (("u", "v"), [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
            "yy": (("u", "v"), [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]),
        },
    )
    assert_identical(actual, expected)

    # all scalar labels
    actual = ds.sel(xx=1.1, yy=3.1, method="nearest")
    expected = xr.Dataset(coords={"xx": 1.0, "yy": 3.0})
    assert_identical(actual, expected)

    # broadcast scalar to label shape and dimensions
    actual = ds.sel(xx=1.1, yy=xr.Variable("u", [3.1, 3.1, 3.1]), method="nearest")
    expected = ds.sel(
        xx=xr.Variable("u", [1.1, 1.1, 1.1]),
        yy=xr.Variable("u", [3.1, 3.1, 3.1]),
        method="nearest",
    )
    assert_identical(actual, expected)

    # broadcast orthogonal 1-dimensional labels
    actual = ds.sel(
        xx=xr.Variable("u", [1.1, 1.1]),
        yy=xr.Variable("v", [3.1, 3.1]),
        method="nearest",
    )
    expected = xr.Dataset(
        coords={
            "xx": (("u", "v"), [[1.0, 1.0], [1.0, 1.0]]),
            "yy": (("u", "v"), [[3.0, 3.0], [3.0, 3.0]]),
        },
    )
    assert_identical(actual, expected)

    # implicit dimension array-like labels
    actual = ds.sel(
        xx=[[1.1, 1.1, 1.1], [1.9, 1.9, 1.9]],
        yy=[[3.1, 3.1, 3.1], [3.9, 3.9, 3.9]],
        method="nearest",
    )
    expected = ds.sel(
        xx=xr.Variable(ds.xx.dims, [[1.1, 1.1, 1.1], [1.9, 1.9, 1.9]]),
        yy=xr.Variable(ds.yy.dims, [[3.1, 3.1, 3.1], [3.9, 3.9, 3.9]]),
        method="nearest",
    )
    assert_identical(actual, expected)


@requires_scipy
def test_tree_index_sel_errors() -> None:
    xx, yy = np.meshgrid([1.0, 2.0], [3.0, 4.0])
    ds = xr.Dataset(coords={"xx": (("y", "x"), xx), "yy": (("y", "x"), yy)}).set_xindex(
        ("xx", "yy"), NDPointIndex
    )

    with pytest.raises(ValueError, match="method='nearest'"):
        ds.sel(xx=1.1, yy=3.1)

    with pytest.raises(ValueError, match="missing labels"):
        ds.sel(xx=1.1, method="nearest")

    with pytest.raises(ValueError, match="invalid label value"):
        # invalid array-like dimensions
        ds.sel(xx=[1.1, 1.9], yy=[3.1, 3.9], method="nearest")

    # error while trying to broadcast labels
    with pytest.raises(xr.AlignmentError, match=r".*conflicting dimension sizes"):
        ds.sel(
            xx=xr.Variable("u", [1.1, 1.1, 1.1]),
            yy=xr.Variable("u", [3.1, 3.1]),
            method="nearest",
        )


@requires_scipy
def test_tree_index_equals() -> None:
    xx1, yy1 = np.meshgrid([1.0, 2.0], [3.0, 4.0])
    ds1 = xr.Dataset(
        coords={"xx": (("y", "x"), xx1), "yy": (("y", "x"), yy1)}
    ).set_xindex(("xx", "yy"), NDPointIndex)

    xx2, yy2 = np.meshgrid([1.0, 2.0], [3.0, 4.0])
    ds2 = xr.Dataset(
        coords={"xx": (("y", "x"), xx2), "yy": (("y", "x"), yy2)}
    ).set_xindex(("xx", "yy"), NDPointIndex)

    xx3, yy3 = np.meshgrid([10.0, 20.0], [30.0, 40.0])
    ds3 = xr.Dataset(
        coords={"xx": (("y", "x"), xx3), "yy": (("y", "x"), yy3)}
    ).set_xindex(("xx", "yy"), NDPointIndex)

    assert ds1.xindexes["xx"].equals(ds2.xindexes["xx"])
    assert not ds1.xindexes["xx"].equals(ds3.xindexes["xx"])


def test_tree_index_rename() -> None:
    xx, yy = np.meshgrid([1.0, 2.0], [3.0, 4.0])
    ds = xr.Dataset(coords={"xx": (("y", "x"), xx), "yy": (("y", "x"), yy)}).set_xindex(
        ("xx", "yy"), NDPointIndex
    )

    ds_renamed = ds.rename_dims(y="u").rename_vars(yy="uu")
    assert "uu" in ds_renamed.xindexes
    assert isinstance(ds_renamed.xindexes["uu"], NDPointIndex)
    assert ds_renamed.xindexes["xx"] is ds_renamed.xindexes["uu"]

    # test via sel() with implicit dimension array-like labels, which relies on
    # NDPointIndex._coord_names and NDPointIndex._dims internal attrs
    actual = ds_renamed.sel(
        xx=[[1.1, 1.1, 1.1], [1.9, 1.9, 1.9]],
        uu=[[3.1, 3.1, 3.1], [3.9, 3.9, 3.9]],
        method="nearest",
    )
    expected = ds_renamed.sel(
        xx=xr.Variable(ds_renamed.xx.dims, [[1.1, 1.1, 1.1], [1.9, 1.9, 1.9]]),
        uu=xr.Variable(ds_renamed.uu.dims, [[3.1, 3.1, 3.1], [3.9, 3.9, 3.9]]),
        method="nearest",
    )
    assert_identical(actual, expected)


@requires_scipy
def test_tree_index_1d_coords() -> None:
    ds = xr.Dataset(
        {"temp": ("points", np.arange(20, dtype=float))},
        coords={
            "x": ("points", np.linspace(0, 10, 20)),
            "y": ("points", np.linspace(15, 8, 20)),
        },
    )
    ds_indexed = ds.set_xindex(("x", "y"), NDPointIndex)

    assert isinstance(ds_indexed.xindexes["x"], NDPointIndex)

    actual = ds_indexed.sel(x=5.0, y=11.5, method="nearest")
    assert actual.temp.values == 10.0

    actual = ds_indexed.sel(
        x=xr.Variable("query", [0.0, 5.0, 10.0]),
        y=xr.Variable("query", [15.0, 11.5, 8.0]),
        method="nearest",
    )
    expected = xr.Dataset(
        {"temp": ("query", [0.0, 10.0, 19.0])},
        coords={
            "x": ("query", [0.0, ds.x.values[10], 10.0]),
            "y": ("query", [15.0, ds.y.values[10], 8.0]),
        },
    )
    assert_identical(actual, expected)


def test_brute_force_adapter_nearest_sel() -> None:
    # the pure test adapter also works with the existing nearest-neighbor query
    xx, yy = np.meshgrid([1.0, 2.0], [3.0, 4.0])
    ds = xr.Dataset(coords={"xx": (("y", "x"), xx), "yy": (("y", "x"), yy)})
    ds_indexed = ds.set_xindex(
        ("xx", "yy"), NDPointIndex, tree_adapter_cls=BruteForceTreeAdapter
    )

    actual = ds_indexed.sel(xx=1.9, yy=xr.Variable("u", [3.9, 3.1]), method="nearest")
    expected = xr.Dataset(
        coords={"xx": ("u", [2.0, 2.0]), "yy": ("u", [4.0, 3.0])}
    )
    assert_identical(actual, expected)


def test_sel_neighbors_unsupported_adapter() -> None:
    xx, yy = np.meshgrid([1.0, 2.0], [3.0, 4.0])
    ds = xr.Dataset(coords={"xx": (("y", "x"), xx), "yy": (("y", "x"), yy)})
    ds_indexed = ds.set_xindex(
        ("xx", "yy"), NDPointIndex, tree_adapter_cls=NearestOnlyTreeAdapter
    )

    with pytest.raises(NotImplementedError, match="does not support"):
        ds_indexed.xindexes["xx"].sel_neighbors({"xx": 1.0, "yy": 3.0}, k=2, radius=1.0)


def test_tree_index_empty_error() -> None:
    ds = xr.Dataset(
        coords={
            "xx": ("points", np.array([], dtype=float)),
            "yy": ("points", np.array([], dtype=float)),
        }
    )

    with pytest.raises(ValueError, match="zero points"):
        ds.set_xindex(
            ("xx", "yy"), NDPointIndex, tree_adapter_cls=BruteForceTreeAdapter
        )


@requires_scipy
def test_sel_neighbors_matches_brute_force() -> None:
    rng = np.random.default_rng(42)
    n_points = 30
    xx = rng.uniform(0, 10, n_points)
    yy = rng.uniform(0, 10, n_points)
    temp = rng.uniform(250, 300, n_points)
    pres = rng.uniform(900, 1000, n_points)

    ds = xr.Dataset(
        {"temp": ("points", temp), "pres": ("points", pres)},
        coords={"xx": ("points", xx), "yy": ("points", yy)},
    ).set_xindex(("xx", "yy"), NDPointIndex)

    k = 4
    radius = 3.0
    qxx = rng.uniform(0, 10, 7)
    qyy = rng.uniform(0, 10, 7)

    index = ds.xindexes["xx"]
    result = index.sel_neighbors(
        {"xx": xr.Variable("q", qxx), "yy": xr.Variable("q", qyy)},
        k=k,
        radius=radius,
    )

    # independent brute-force reference
    dist = np.sqrt(
        (qxx[:, None] - xx[None, :]) ** 2 + (qyy[:, None] - yy[None, :]) ** 2
    )
    order = np.argsort(dist, kind="stable", axis=1)[:, :k]
    ref_dist = np.take_along_axis(dist, order, axis=1)
    ref_mask = ref_dist <= radius
    ref_dist = np.where(ref_mask, ref_dist, np.inf)

    np.testing.assert_allclose(result.distances.values, ref_dist)
    np.testing.assert_array_equal(result.mask.values, ref_mask)

    # missing indexers are clamped to 0 and never -1 (i.e., the last sample)
    for indexer in result.dim_indexers.values():
        flat = indexer.values
        assert (flat >= 0).all()
        np.testing.assert_array_equal(flat[~ref_mask], 0)

    # all variables of a multi-variable Dataset are fetched at identical
    # neighbor positions
    safe_order = np.where(ref_mask, order, 0)
    actual = ds.isel(result.dim_indexers)
    np.testing.assert_array_equal(actual.temp.values, temp[safe_order])
    np.testing.assert_array_equal(actual.pres.values, pres[safe_order])

    # same result with the pure test adapter
    ds_bf = ds.drop_indexes(("xx", "yy")).set_xindex(
        ("xx", "yy"), NDPointIndex, tree_adapter_cls=BruteForceTreeAdapter
    )
    result_bf = ds_bf.xindexes["xx"].sel_neighbors(
        {"xx": xr.Variable("q", qxx), "yy": xr.Variable("q", qyy)},
        k=k,
        radius=radius,
    )
    np.testing.assert_allclose(result_bf.distances.values, result.distances.values)
    np.testing.assert_array_equal(result_bf.mask.values, result.mask.values)
    assert_identical(ds_bf.isel(result_bf.dim_indexers), actual)


@requires_scipy
def test_sel_neighbors_ties_and_duplicates() -> None:
    # duplicated coordinates and equidistant points: neighbors are ordered by
    # original point position and duplicates keep their own data
    ds = xr.Dataset(
        {"temp": ("points", [10.0, 20.0, 30.0, 40.0])},
        coords={
            "xx": ("points", [0.0, 0.0, 1.0, 1.0]),
            "yy": ("points", [0.0, 0.0, 1.0, 1.0]),
        },
    ).set_xindex(("xx", "yy"), NDPointIndex)

    index = ds.xindexes["xx"]

    # exact duplicates
    result = index.sel_neighbors({"xx": 0.0, "yy": 0.0}, k=4, radius=2.0)
    np.testing.assert_array_equal(
        result.distances.values, [0.0, 0.0, np.sqrt(2), np.sqrt(2)]
    )
    np.testing.assert_array_equal(result.dim_indexers["points"].values, [0, 1, 2, 3])
    actual = ds.isel(result.dim_indexers)
    np.testing.assert_array_equal(actual.temp.values, [10.0, 20.0, 30.0, 40.0])

    # all points equidistant from the query
    result = index.sel_neighbors({"xx": 0.5, "yy": 0.5}, k=4, radius=2.0)
    np.testing.assert_array_equal(result.dim_indexers["points"].values, [0, 1, 2, 3])


@requires_scipy
def test_sel_neighbors_missing_mask() -> None:
    xx, yy = np.meshgrid([1.0, 2.0], [3.0, 4.0])
    ds = xr.Dataset(
        {"temp": (("y", "x"), [[1.0, 2.0], [3.0, 4.0]])},
        coords={"xx": (("y", "x"), xx), "yy": (("y", "x"), yy)},
    ).set_xindex(("xx", "yy"), NDPointIndex)

    index = ds.xindexes["xx"]
    result = index.sel_neighbors({"xx": 1.1, "yy": 3.1}, k=3, radius=0.5)

    np.testing.assert_array_equal(result.mask.values, [True, False, False])
    np.testing.assert_allclose(
        result.distances.values, [np.sqrt(0.02), np.inf, np.inf]
    )

    # missing entries are clamped to the first sample, never to the last one
    np.testing.assert_array_equal(result.dim_indexers["y"].values, [0, 0, 0])
    np.testing.assert_array_equal(result.dim_indexers["x"].values, [0, 0, 0])

    actual = ds.isel(result.dim_indexers)
    np.testing.assert_array_equal(actual.temp.values, [1.0, 1.0, 1.0])

    # the explicit mask flags the missing entries
    masked = actual.where(result.mask)
    np.testing.assert_array_equal(masked.temp.values, [1.0, np.nan, np.nan])

    # the radius is inclusive
    result = index.sel_neighbors({"xx": 1.0, "yy": 3.0}, k=3, radius=1.0)
    np.testing.assert_array_equal(result.mask.values, [True, True, True])


@requires_scipy
def test_sel_neighbors_k_larger_than_n_points() -> None:
    xx, yy = np.meshgrid([1.0, 2.0], [3.0, 4.0])
    ds = xr.Dataset(
        coords={"xx": (("y", "x"), xx), "yy": (("y", "x"), yy)}
    ).set_xindex(("xx", "yy"), NDPointIndex)

    index = ds.xindexes["xx"]
    result = index.sel_neighbors({"xx": 1.0, "yy": 3.0}, k=6, radius=np.inf)

    assert result.mask.values.sum() == 4
    np.testing.assert_array_equal(result.mask.values[4:], [False, False])
    np.testing.assert_array_equal(result.distances.values[4:], [np.inf, np.inf])


@requires_scipy
def test_sel_neighbors_invalid_labels() -> None:
    xx, yy = np.meshgrid([1.0, 2.0], [3.0, 4.0])
    ds = xr.Dataset(
        coords={"xx": (("y", "x"), xx), "yy": (("y", "x"), yy)}
    ).set_xindex(("xx", "yy"), NDPointIndex)

    index = ds.xindexes["xx"]

    # non-finite query coordinates return all-missing rows
    result = index.sel_neighbors(
        {
            "xx": xr.Variable("u", [np.nan, np.inf, 1.0]),
            "yy": xr.Variable("u", [3.0, 3.0, 3.0]),
        },
        k=2,
        radius=1.0,
    )
    np.testing.assert_array_equal(
        result.mask.values, [[False, False], [False, False], [True, True]]
    )
    np.testing.assert_array_equal(result.distances.values[0], [np.inf, np.inf])
    np.testing.assert_array_equal(result.distances.values[1], [np.inf, np.inf])

    # invalid k / radius / neighbor_dim values
    for bad_k in [0, -1, 2.5, True]:
        with pytest.raises(ValueError, match="k must be a positive integer"):
            index.sel_neighbors({"xx": 1.0, "yy": 3.0}, k=bad_k, radius=1.0)

    for bad_radius in [-1.0, np.nan, "foo"]:
        with pytest.raises(ValueError, match="radius must be a non-negative number"):
            index.sel_neighbors({"xx": 1.0, "yy": 3.0}, k=2, radius=bad_radius)

    with pytest.raises(ValueError, match="neighbor dimension"):
        index.sel_neighbors(
            {"xx": xr.Variable("u", [1.0]), "yy": xr.Variable("u", [3.0])},
            k=2,
            radius=1.0,
            neighbor_dim="u",
        )


@requires_scipy
def test_sel_neighbors_broadcast_and_k1() -> None:
    xx, yy = np.meshgrid([1.0, 2.0], [3.0, 4.0])
    ds = xr.Dataset(
        coords={"xx": (("y", "x"), xx), "yy": (("y", "x"), yy)}
    ).set_xindex(("xx", "yy"), NDPointIndex)

    index = ds.xindexes["xx"]

    # broadcast label dimensions are preserved, neighbor dimension is appended
    result = index.sel_neighbors(
        {"xx": xr.Variable("u", [1.1, 1.9]), "yy": xr.Variable("v", [3.1, 3.9])},
        k=2,
        radius=1.0,
        neighbor_dim="nb",
    )
    assert result.distances.dims == ("u", "v", "nb")
    assert result.mask.dims == ("u", "v", "nb")
    for indexer in result.dim_indexers.values():
        assert indexer.dims == ("u", "v", "nb")

    # k=1 keeps the neighbor dimension
    result = index.sel_neighbors({"xx": 1.1, "yy": 3.1}, k=1, radius=1.0)
    assert result.distances.dims == ("neighbor",)
    assert result.distances.shape == (1,)
    np.testing.assert_array_equal(result.mask.values, [True])
