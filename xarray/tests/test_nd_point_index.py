from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

import xarray as xr
from xarray.indexes import NDPointIndex
from xarray.indexes.nd_point_index import ScipyKDTreeAdapter, TreeAdapter
from xarray.tests import assert_identical, has_scipy, requires_scipy


class BruteForceTreeAdapter(TreeAdapter):
    """Pure NumPy brute-force tree adapter used as a reference in tests."""

    def __init__(self, points: np.ndarray, *, options: Mapping[str, Any]):
        self._points = np.asarray(points, dtype=float)

    def query(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distances, indices = self.query_neighbors(points, 1, np.inf)
        return distances[:, 0], indices[:, 0]

    def query_neighbors(
        self, points: np.ndarray, k: int, radius: float
    ) -> tuple[np.ndarray, np.ndarray]:
        distances = np.full((len(points), k), np.inf)
        indices = np.full((len(points), k), -1, dtype=np.intp)

        if self._points.shape[0]:
            dist = np.sqrt(
                (
                    (np.asarray(points, dtype=float)[:, None, :] - self._points[None])
                    ** 2
                ).sum(axis=2)
            )
            dist = np.where(dist <= radius, dist, np.inf)
            # stable sort: equidistant points keep their original position order
            n_found = min(k, self._points.shape[0])
            order = np.argsort(dist, axis=1, kind="stable")[:, :n_found]
            dist = np.take_along_axis(dist, order, axis=1)
            hit = np.isfinite(dist)
            distances[:, :n_found][hit] = dist[hit]
            indices[:, :n_found][hit] = order[hit]

        return distances, indices

    def equals(self, other) -> bool:
        return np.array_equal(self._points, other._points)


class NearestOnlyTreeAdapter(TreeAdapter):
    """Minimal adapter that does not implement neighbor queries."""

    def __init__(self, points: np.ndarray, *, options: Mapping[str, Any]):
        self._points = np.asarray(points, dtype=float)

    def query(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(len(points)), np.zeros(len(points), dtype=np.intp)

    def equals(self, other) -> bool:
        return np.array_equal(self._points, other._points)


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


@requires_scipy
def test_query_neighbors_vs_brute_force() -> None:
    rng = np.random.default_rng(42)
    points = rng.random((50, 2)) * 10
    # include duplicated points
    points[5] = points[10]
    points[40:] = points[:10]
    queries = rng.random((30, 2)) * 10

    ds = xr.Dataset(coords={"x": ("p", points[:, 0]), "y": ("p", points[:, 1])})
    ds_scipy = ds.set_xindex(("x", "y"), NDPointIndex)
    ds_brute = ds.set_xindex(
        ("x", "y"), NDPointIndex, tree_adapter_cls=BruteForceTreeAdapter
    )

    labels = {
        "x": xr.Variable("q", queries[:, 0]),
        "y": xr.Variable("q", queries[:, 1]),
    }

    for k, radius in [(1, np.inf), (3, 2.0), (7, 5.0), (60, np.inf), (4, 0.5)]:
        actual = ds_scipy.xindexes["x"].query_neighbors(labels, k=k, radius=radius)
        expected = ds_brute.xindexes["x"].query_neighbors(labels, k=k, radius=radius)

        assert actual.distances.dims == ("q", "neighbor")
        np.testing.assert_array_equal(actual.mask.values, expected.mask.values)
        np.testing.assert_allclose(actual.distances.values, expected.distances.values)
        # neighbor positions must be identical
        np.testing.assert_array_equal(
            np.where(actual.mask.values, actual.dim_indexers["p"].values, -1),
            np.where(expected.mask.values, expected.dim_indexers["p"].values, -1),
        )


@requires_scipy
def test_query_neighbors_dims() -> None:
    xx, yy = np.meshgrid([1.0, 2.0], [3.0, 4.0])
    ds = xr.Dataset(coords={"xx": (("y", "x"), xx), "yy": (("y", "x"), yy)}).set_xindex(
        ("xx", "yy"), NDPointIndex
    )
    index = ds.xindexes["xx"]

    # scalar labels: only the neighbor dimension remains
    result = index.query_neighbors({"xx": 1.1, "yy": 3.1}, k=2)
    assert result.distances.dims == ("neighbor",)
    assert result.mask.dims == ("neighbor",)
    assert all(v.dims == ("neighbor",) for v in result.dim_indexers.values())

    # broadcast label dimensions are preserved, neighbor dimension is appended
    result = index.query_neighbors(
        {"xx": xr.Variable("u", [1.1, 1.9]), "yy": xr.Variable("v", [3.1, 3.9])},
        k=3,
        neighbor_dim="n",
    )
    assert result.distances.dims == ("u", "v", "n")
    assert result.distances.shape == (2, 2, 3)
    assert result.mask.dims == ("u", "v", "n")
    assert all(v.dims == ("u", "v", "n") for v in result.dim_indexers.values())
    assert set(result.dim_indexers) == {"y", "x"}


@requires_scipy
def test_query_neighbors_stable_ties_and_duplicates() -> None:
    # points all at distance exactly 1 from the origin, with duplicates
    x = np.array([1.0, 0.0, -1.0, 0.0, 1.0, 0.0])
    y = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0])
    data = np.arange(len(x), dtype=float)
    ds = xr.Dataset({"v": ("p", data)}, coords={"x": ("p", x), "y": ("p", y)})
    ds = ds.set_xindex(("x", "y"), NDPointIndex)

    result = ds.xindexes["x"].query_neighbors({"x": 0.0, "y": 0.0}, k=6)

    # equidistant neighbors are ordered by their original position
    np.testing.assert_array_equal(result.dim_indexers["p"].values, np.arange(6))
    np.testing.assert_allclose(result.distances.values, np.ones(6))
    # duplicated coordinates each keep their own data
    np.testing.assert_array_equal(ds.isel(result.dim_indexers).v.values, data)

    # also holds when truncating equidistant neighbors
    result = ds.xindexes["x"].query_neighbors({"x": 0.0, "y": 0.0}, k=3)
    np.testing.assert_array_equal(result.dim_indexers["p"].values, [0, 1, 2])


@requires_scipy
def test_query_neighbors_missing_mask() -> None:
    ds = xr.Dataset(
        {"v": ("p", [10.0, 20.0, 30.0])},
        coords={"x": ("p", [0.0, 1.0, 2.0]), "y": ("p", [0.0, 0.0, 0.0])},
    ).set_xindex(("x", "y"), NDPointIndex)

    result = ds.xindexes["x"].query_neighbors({"x": 0.1, "y": 0.0}, k=3, radius=0.5)

    # only the first point is within radius
    np.testing.assert_array_equal(result.mask.values, [True, False, False])
    np.testing.assert_allclose(result.distances.values, [0.1, np.inf, np.inf])

    # missing entries must never use a -1 index (which would silently select
    # the last indexed point)
    for indexer in result.dim_indexers.values():
        assert not (indexer.values < 0).any()

    # the explicit mask discards the missing neighbors
    actual = ds.isel(result.dim_indexers).where(result.mask)
    np.testing.assert_array_equal(actual.v.values, [10.0, np.nan, np.nan])


@requires_scipy
def test_query_neighbors_k_larger_than_npoints() -> None:
    ds = xr.Dataset(coords={"x": ("p", [0.0, 1.0]), "y": ("p", [0.0, 0.0])}).set_xindex(
        ("x", "y"), NDPointIndex
    )

    result = ds.xindexes["x"].query_neighbors({"x": 0.0, "y": 0.0}, k=5)

    np.testing.assert_array_equal(result.mask.values, [True, True, False, False, False])
    np.testing.assert_allclose(
        result.distances.values, [0.0, 1.0, np.inf, np.inf, np.inf]
    )
    np.testing.assert_array_equal(result.dim_indexers["p"].values[:2], [0, 1])


@requires_scipy
@pytest.mark.parametrize(
    "tree_adapter_cls", [ScipyKDTreeAdapter, BruteForceTreeAdapter]
)
def test_query_neighbors_empty_index(tree_adapter_cls) -> None:
    ds = xr.Dataset(
        {"v": ("p", [])},
        coords={"x": ("p", []), "y": ("p", [])},
    ).set_xindex(("x", "y"), NDPointIndex, tree_adapter_cls=tree_adapter_cls)

    result = ds.xindexes["x"].query_neighbors({"x": 1.0, "y": 2.0}, k=3)

    np.testing.assert_array_equal(result.mask.values, [False, False, False])
    np.testing.assert_array_equal(result.distances.values, [np.inf, np.inf, np.inf])
    for indexer in result.dim_indexers.values():
        assert not (indexer.values < 0).any()


@requires_scipy
def test_query_neighbors_invalid_labels() -> None:
    ds = xr.Dataset(
        coords={"x": ("p", [0.0, 1.0, 2.0]), "y": ("p", [0.0, 0.0, 0.0])}
    ).set_xindex(("x", "y"), NDPointIndex)

    result = ds.xindexes["x"].query_neighbors(
        {"x": xr.Variable("q", [0.1, np.nan]), "y": xr.Variable("q", [0.0, 0.0])},
        k=2,
    )

    # non-finite query labels yield only missing neighbors
    np.testing.assert_array_equal(result.mask.values[0], [True, True])
    np.testing.assert_array_equal(result.mask.values[1], [False, False])
    np.testing.assert_array_equal(result.distances.values[1], [np.inf, np.inf])


@requires_scipy
def test_query_neighbors_errors() -> None:
    xx, yy = np.meshgrid([1.0, 2.0], [3.0, 4.0])
    ds = xr.Dataset(coords={"xx": (("y", "x"), xx), "yy": (("y", "x"), yy)}).set_xindex(
        ("xx", "yy"), NDPointIndex
    )
    index = ds.xindexes["xx"]

    for bad_k in (0, -1, 1.5, "2"):
        with pytest.raises(ValueError, match="k must be a positive integer"):
            index.query_neighbors({"xx": 1.0, "yy": 3.0}, k=bad_k)

    for bad_radius in (-1.0, np.nan):
        with pytest.raises(ValueError, match="radius must be a non-negative number"):
            index.query_neighbors({"xx": 1.0, "yy": 3.0}, k=1, radius=bad_radius)

    with pytest.raises(ValueError, match="radius must be a scalar"):
        index.query_neighbors({"xx": 1.0, "yy": 3.0}, k=1, radius=[1.0, 2.0])

    with pytest.raises(ValueError, match="missing labels"):
        index.query_neighbors({"xx": 1.0}, k=1)

    with pytest.raises(ValueError, match="conflicts with the broadcast"):
        index.query_neighbors({"xx": xr.Variable("neighbor", [1.0]), "yy": 3.0}, k=1)


def test_query_neighbors_not_implemented() -> None:
    ds = xr.Dataset(coords={"x": ("p", [0.0, 1.0]), "y": ("p", [0.0, 1.0])}).set_xindex(
        ("x", "y"), NDPointIndex, tree_adapter_cls=NearestOnlyTreeAdapter
    )

    with pytest.raises(NotImplementedError, match="does not support neighbor queries"):
        ds.xindexes["x"].query_neighbors({"x": 0.0, "y": 0.0}, k=2)


@requires_scipy
def test_query_neighbors_multivariable_dataset() -> None:
    rng = np.random.default_rng(123)
    n = 40
    x = rng.random(n) * 5
    y = rng.random(n) * 5
    ds = xr.Dataset(
        {
            "temp": ("p", rng.random(n)),
            "pres": ("p", rng.random(n)),
            "meta": ("p", np.arange(n).astype(str)),
        },
        coords={"x": ("p", x), "y": ("p", y)},
    ).set_xindex(("x", "y"), NDPointIndex)

    queries = rng.random((6, 2)) * 5
    k, radius = 4, 2.5
    result = ds.xindexes["x"].query_neighbors(
        {"x": xr.Variable("q", queries[:, 0]), "y": xr.Variable("q", queries[:, 1])},
        k=k,
        radius=radius,
    )

    # brute-force reference
    points = np.stack([x, y]).T
    dist = np.sqrt(((queries[:, None, :] - points[None]) ** 2).sum(axis=2))
    dist = np.where(dist <= radius, dist, np.inf)
    order = np.argsort(dist, axis=1, kind="stable")[:, :k]
    ref_dist = np.take_along_axis(dist, order, axis=1)
    ref_mask = np.isfinite(ref_dist)
    ref_idx = np.where(ref_mask, order, -1)

    np.testing.assert_array_equal(result.mask.values, ref_mask)
    np.testing.assert_allclose(
        result.distances.values, np.where(ref_mask, ref_dist, np.inf)
    )
    np.testing.assert_array_equal(
        np.where(ref_mask, result.dim_indexers["p"].values, -1), ref_idx
    )

    # every variable of the dataset is retrieved at the exact same
    # neighbor positions
    actual = ds.isel(result.dim_indexers)
    for name in ("temp", "pres", "meta"):
        np.testing.assert_array_equal(
            actual[name].values[ref_mask], ds[name].values[ref_idx[ref_mask]]
        )
