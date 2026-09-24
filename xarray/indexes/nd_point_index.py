from __future__ import annotations

import abc
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

from xarray.core.dataarray import DataArray
from xarray.core.indexes import Index
from xarray.core.indexing import IndexSelResult
from xarray.core.utils import is_scalar
from xarray.core.variable import Variable
from xarray.structure.alignment import broadcast

if TYPE_CHECKING:
    from scipy.spatial import KDTree

    from xarray.core.types import Self


class TreeAdapter(abc.ABC):
    """Lightweight adapter abstract class for plugging in 3rd-party structures
    like :py:class:`scipy.spatial.KDTree` or :py:class:`sklearn.neighbors.KDTree`
    into :py:class:`~xarray.indexes.NDPointIndex`.

    """

    @abc.abstractmethod
    def __init__(self, points: np.ndarray, *, options: Mapping[str, Any]):
        """
        Parameters
        ----------
        points : ndarray of shape (n_points, n_coordinates)
            Two-dimensional array of points/samples (rows) and their
            corresponding coordinate labels (columns) to index.
        """
        ...

    @abc.abstractmethod
    def query(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Query points.

        Parameters
        ----------
        points: ndarray of shape (n_points, n_coordinates)
            Two-dimensional array of points/samples (rows) and their
            corresponding coordinate labels (columns) to query.

        Returns
        -------
        distances : ndarray of shape (n_points)
            Distances to the nearest neighbors.
        indices : ndarray of shape (n_points)
            Indices of the nearest neighbors in the array of the indexed
            points.
        """
        ...

    def equals(self, other: Self) -> bool:
        """Check equality with another TreeAdapter of the same kind.

        Parameters
        ----------
        other :
            The other TreeAdapter object to compare with this object.

        """
        raise NotImplementedError

    def query_neighbors(
        self, points: np.ndarray, *, k: int, radius: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Query up to k nearest neighbors within a radius for each point.

        Parameters
        ----------
        points: ndarray of shape (n_points, n_coordinates)
            Two-dimensional array of points/samples (rows) and their
            corresponding coordinate labels (columns) to query.
        k : int
            Maximum number of neighbors to return for each query point.
        radius : float
            Maximum distance (inclusive) of the returned neighbors.

        Returns
        -------
        distances : ndarray of shape (n_points, k)
            Distances to the neighbors, sorted by increasing distance.
            Missing entries (fewer than k neighbors within the radius)
            must be set to inf.
        indices : ndarray of shape (n_points, k)
            Indices of the neighbors in the array of the indexed points.
            Neighbors at equal distance must be ordered by increasing
            index (stable sort). Missing entries must be set to -1.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support radius-limited "
            "k-nearest neighbor queries"
        )


class ScipyKDTreeAdapter(TreeAdapter):
    """:py:class:`scipy.spatial.KDTree` adapter for :py:class:`~xarray.indexes.NDPointIndex`."""

    _kdtree: KDTree

    def __init__(self, points: np.ndarray, options: Mapping[str, Any]):
        try:
            from scipy.spatial import KDTree
        except ImportError as err:
            raise ImportError(
                "`NDPointIndex` requires `scipy` when used with `ScipyKDTreeAdapter`. "
                "Please ensure that `scipy` is installed and importable."
            ) from err

        self._kdtree = KDTree(points, **options)

    def query(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._kdtree.query(points)  # type: ignore[return-value,unused-ignore]

    def equals(self, other: Self) -> bool:
        return np.array_equal(self._kdtree.data, other._kdtree.data)

    def query_neighbors(
        self, points: np.ndarray, *, k: int, radius: float
    ) -> tuple[np.ndarray, np.ndarray]:
        n_indexed = len(self._kdtree.data)
        # scipy's distance_upper_bound excludes points at exactly the given
        # distance: slightly bump it to make the radius inclusive
        distances, indices = self._kdtree.query(
            points, k=k, distance_upper_bound=np.nextafter(radius, np.inf)
        )

        # scipy drops the neighbor axis when k=1
        if k == 1:
            distances = distances[:, np.newaxis]
            indices = indices[:, np.newaxis]

        # break distance ties by original point position (stable sort)
        order = np.lexsort((indices, distances), axis=1)
        distances = np.take_along_axis(distances, order, axis=1)
        indices = np.take_along_axis(indices, order, axis=1)

        # scipy flags missing neighbors with an out-of-bounds index: convert
        # it to an explicit -1 missing indicator
        missing = (indices >= n_indexed) | (distances > radius)
        indices = np.where(missing, -1, indices)
        distances = np.where(missing, np.inf, distances)

        return distances, indices


def get_points(coords: Iterable[Variable | Any]) -> np.ndarray:
    """Re-arrange data from a sequence of xarray coordinate variables or
    labels into a 2-d array of shape (n_points, n_coordinates).

    """
    data = [c.values if isinstance(c, Variable | DataArray) else c for c in coords]
    return np.stack([np.ravel(d) for d in data]).T


T_TreeAdapter = TypeVar("T_TreeAdapter", bound=TreeAdapter)


@dataclass(frozen=True)
class NeighborsResult:
    """Result of :py:meth:`NDPointIndex.sel_neighbors`.

    Attributes
    ----------
    dim_indexers : dict of Hashable to Variable
        Dimension indexers suitable for :py:meth:`Dataset.isel` or
        :py:meth:`DataArray.isel`. Each indexer has the broadcast dimensions
        of the query labels plus a trailing neighbor dimension. Positions
        with no neighbor are clamped to index 0; use ``mask`` to identify
        them.
    distances : DataArray
        Distances to the neighbors with the broadcast dimensions of the
        query labels plus a trailing neighbor dimension. Missing entries
        are set to inf.
    mask : DataArray
        Boolean array (same dimensions as ``distances``) that is True
        where a neighbor was found within the radius and False elsewhere.
    """

    dim_indexers: dict[Hashable, Variable]
    distances: DataArray
    mask: DataArray


class NDPointIndex(Index, Generic[T_TreeAdapter]):
    """Xarray index for irregular, n-dimensional data.

    This index may be associated with a set of coordinate variables representing
    the arbitrary location of data points in an n-dimensional space. All
    coordinates must have the same shape and dimensions. The number of
    associated coordinate variables must correspond to the number of dimensions
    of the space.

    This index supports label-based selection (nearest neighbor lookup). It also
    has limited support for alignment.

    By default, this index relies on :py:class:`scipy.spatial.KDTree` for fast
    lookup.

    Do not use :py:meth:`~xarray.indexes.NDPointIndex.__init__` directly. Instead
    use :py:meth:`xarray.Dataset.set_xindex` or
    :py:meth:`xarray.DataArray.set_xindex` to create and set the index from
    existing coordinates (see the example below).

    Examples
    --------
    An example using a dataset with 2-dimensional coordinates.

    >>> xx = [[1.0, 2.0], [3.0, 0.0]]
    >>> yy = [[11.0, 21.0], [29.0, 9.0]]
    >>> ds = xr.Dataset(coords={"xx": (("y", "x"), xx), "yy": (("y", "x"), yy)})
    >>> ds
    <xarray.Dataset> Size: 64B
    Dimensions:  (y: 2, x: 2)
    Coordinates:
        xx       (y, x) float64 32B 1.0 2.0 3.0 0.0
        yy       (y, x) float64 32B 11.0 21.0 29.0 9.0
    Dimensions without coordinates: y, x
    Data variables:
        *empty*

    Creation of an NDPointIndex from the "xx" and "yy" coordinate variables:

    >>> ds = ds.set_xindex(("xx", "yy"), xr.indexes.NDPointIndex)
    >>> ds
    <xarray.Dataset> Size: 64B
    Dimensions:  (y: 2, x: 2)
    Coordinates:
      * xx       (y, x) float64 32B 1.0 2.0 3.0 0.0
      * yy       (y, x) float64 32B 11.0 21.0 29.0 9.0
    Dimensions without coordinates: y, x
    Data variables:
        *empty*
    Indexes:
      ┌ xx       NDPointIndex (ScipyKDTreeAdapter)
      └ yy

    Point-wise (nearest-neighbor) data selection using Xarray's advanced
    indexing, i.e., using arbitrary dimension(s) for the Variable objects passed
    as labels:

    >>> ds.sel(
    ...     xx=xr.Variable("points", [1.9, 0.1]),
    ...     yy=xr.Variable("points", [13.0, 8.0]),
    ...     method="nearest",
    ... )
    <xarray.Dataset> Size: 32B
    Dimensions:  (points: 2)
    Coordinates:
        xx       (points) float64 16B 1.0 0.0
        yy       (points) float64 16B 11.0 9.0
    Dimensions without coordinates: points
    Data variables:
        *empty*

    Data selection with scalar labels:

    >>> ds.sel(xx=1.9, yy=13.0, method="nearest")
    <xarray.Dataset> Size: 16B
    Dimensions:  ()
    Coordinates:
        xx       float64 8B 1.0
        yy       float64 8B 11.0
    Data variables:
        *empty*

    Data selection with broadcasting the input labels:

    >>> ds.sel(xx=1.9, yy=xr.Variable("points", [13.0, 8.0]), method="nearest")
    <xarray.Dataset> Size: 32B
    Dimensions:  (points: 2)
    Coordinates:
        xx       (points) float64 16B 1.0 0.0
        yy       (points) float64 16B 11.0 9.0
    Dimensions without coordinates: points
    Data variables:
        *empty*

    >>> da = xr.DataArray(
    ...     [[45.1, 53.3], [65.4, 78.2]],
    ...     coords={"u": [1.9, 0.1], "v": [13.0, 8.0]},
    ...     dims=("u", "v"),
    ... )
    >>> ds.sel(xx=da.u, yy=da.v, method="nearest")
    <xarray.Dataset> Size: 64B
    Dimensions:  (u: 2, v: 2)
    Coordinates:
        xx       (u, v) float64 32B 1.0 0.0 1.0 0.0
        yy       (u, v) float64 32B 11.0 9.0 11.0 9.0
    Dimensions without coordinates: u, v
    Data variables:
        *empty*

    Data selection with array-like labels (implicit dimensions):

    >>> ds.sel(xx=[[1.9], [0.1]], yy=[[13.0], [8.0]], method="nearest")
    <xarray.Dataset> Size: 32B
    Dimensions:  (y: 2, x: 1)
    Coordinates:
        xx       (y, x) float64 16B 1.0 0.0
        yy       (y, x) float64 16B 11.0 9.0
    Dimensions without coordinates: y, x
    Data variables:
        *empty*

    """

    _tree_obj: T_TreeAdapter
    _coord_names: tuple[Hashable, ...]
    _dims: tuple[Hashable, ...]
    _shape: tuple[int, ...]

    def __init__(
        self,
        tree_obj: T_TreeAdapter,
        *,
        coord_names: tuple[Hashable, ...],
        dims: tuple[Hashable, ...],
        shape: tuple[int, ...],
    ):
        # this constructor is "private"
        assert isinstance(tree_obj, TreeAdapter)
        self._tree_obj = tree_obj

        assert len(dims) == len(shape)
        self._coord_names = coord_names
        self._dims = dims
        self._shape = shape

    @classmethod
    def from_variables(
        cls,
        variables: Mapping[Any, Variable],
        *,
        options: Mapping[str, Any],
    ) -> Self:
        if len({var.dims for var in variables.values()}) > 1:
            var_names = ",".join(vn for vn in variables)
            raise ValueError(
                f"variables {var_names} must all have the same dimensions and the same shape"
            )

        var0 = next(iter(variables.values()))

        opts = dict(options)

        tree_adapter_cls: type[T_TreeAdapter] = opts.pop("tree_adapter_cls", None)
        if tree_adapter_cls is None:
            tree_adapter_cls = ScipyKDTreeAdapter

        points = get_points(variables.values())

        if points.shape[0] == 0:
            raise ValueError(
                "cannot create an NDPointIndex from coordinates with zero points"
            )

        return cls(
            tree_adapter_cls(points, options=opts),
            coord_names=tuple(variables),
            dims=var0.dims,
            shape=var0.shape,
        )

    def create_variables(
        self, variables: Mapping[Any, Variable] | None = None
    ) -> dict[Any, Variable]:
        if variables is not None:
            for var in variables.values():
                # maybe re-sync variable dimensions with the index object
                # returned by NDPointIndex.rename()
                if var.dims != self._dims:
                    var.dims = self._dims
            return dict(**variables)
        else:
            return {}

    def equals(
        self, other: Index, *, exclude: frozenset[Hashable] | None = None
    ) -> bool:
        if not isinstance(other, NDPointIndex):
            return False
        if type(self._tree_obj) is not type(other._tree_obj):
            return False
        return self._tree_obj.equals(other._tree_obj)

    def _get_dim_indexers(
        self,
        indices: np.ndarray,
        dims: tuple[Hashable, ...],
        shape: tuple[int, ...],
    ) -> dict[Hashable, Variable]:
        """Returns dimension indexers based on the query results (indices) and
        the given indexer dimensions and shape.

        1. Unravel the flat indices returned from the query
        2. Reshape the unraveled indices according to indexers shapes
        3. Wrap the indices in xarray.Variable objects.

        """
        dim_indexers = {}

        u_indices = list(np.unravel_index(indices.ravel(), self._shape))

        for dim, ind in zip(self._dims, u_indices, strict=False):
            dim_indexers[dim] = Variable(dims, ind.reshape(shape))

        return dim_indexers

    def sel(
        self, labels: dict[Any, Any], method=None, tolerance=None
    ) -> IndexSelResult:
        if method != "nearest":
            raise ValueError(
                "NDPointIndex only supports selection with method='nearest'"
            )

        xr_labels, label_dims, label_shape = self._normalize_labels(labels)

        # get and return dimension indexers
        points = get_points(xr_labels[name] for name in self._coord_names)
        _, indices = self._tree_obj.query(points)

        dim_indexers = self._get_dim_indexers(indices, label_dims, label_shape)

        return IndexSelResult(dim_indexers=dim_indexers)

    def _normalize_labels(
        self, labels: dict[Any, Any]
    ) -> tuple[dict[Any, DataArray], tuple[Hashable, ...], tuple[int, ...]]:
        """Validate labels, convert them to broadcasted DataArray objects and
        return them along with their broadcast dimensions and shape.

        """
        missing_labels = set(self._coord_names) - set(labels)
        if missing_labels:
            missing_labels_str = ",".join([f"{name}" for name in missing_labels])
            raise ValueError(f"missing labels for coordinate(s): {missing_labels_str}.")

        # maybe convert labels into xarray DataArray objects
        xr_labels: dict[Any, DataArray] = {}

        for name, lbl in labels.items():
            if isinstance(lbl, DataArray):
                xr_labels[name] = lbl
            elif isinstance(lbl, Variable):
                xr_labels[name] = DataArray(lbl)
            elif is_scalar(lbl):
                xr_labels[name] = DataArray(lbl, dims=())
            elif np.asarray(lbl).ndim == len(self._dims):
                xr_labels[name] = DataArray(lbl, dims=self._dims)
            else:
                raise ValueError(
                    "invalid label value. NDPointIndex only supports advanced (point-wise) indexing "
                    "with the following label value kinds:\n"
                    "- xarray.DataArray or xarray.Variable objects\n"
                    "- scalar values\n"
                    "- unlabelled array-like objects with the same number of dimensions "
                    f"than the {self._coord_names} coordinate variables ({len(self._dims)})"
                )

        # broadcast xarray labels against one another and determine labels shape and dimensions
        broadcasted = broadcast(*xr_labels.values())
        label_dims = broadcasted[0].dims
        label_shape = broadcasted[0].shape
        xr_labels = dict(zip(xr_labels, broadcasted, strict=True))

        return xr_labels, label_dims, label_shape

    def sel_neighbors(
        self,
        labels: dict[Any, Any],
        *,
        k: int,
        radius: float,
        neighbor_dim: Hashable = "neighbor",
    ) -> NeighborsResult:
        """Query up to k nearest neighbors within a radius for each label.

        The query labels are broadcast against one another like in
        :py:meth:`NDPointIndex.sel`. The returned indexers, distances and
        mask all have the broadcast label dimensions plus a trailing
        neighbor dimension of size k.

        Parameters
        ----------
        labels : dict
            Mapping of coordinate names to query labels (same kinds of
            values as accepted by :py:meth:`NDPointIndex.sel`).
        k : int
            Maximum number of neighbors to return for each query point.
            It may be larger than the total number of indexed points, in
            which case the remaining entries are flagged as missing.
        radius : float
            Maximum distance (inclusive) of the returned neighbors.
        neighbor_dim : Hashable, default: "neighbor"
            Name of the extra neighbor dimension in the returned indexers,
            distances and mask.

        Returns
        -------
        NeighborsResult
            Dimension indexers (missing entries clamped to index 0),
            distances (missing entries set to inf) and a boolean mask
            indicating which entries are actual neighbors.

        """
        if isinstance(k, bool) or not isinstance(k, int | np.integer) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        k = int(k)

        try:
            radius = float(radius)
        except (TypeError, ValueError):
            raise ValueError(
                f"radius must be a non-negative number, got {radius!r}"
            ) from None
        if np.isnan(radius) or radius < 0:
            raise ValueError(f"radius must be a non-negative number, got {radius!r}")

        xr_labels, label_dims, label_shape = self._normalize_labels(labels)

        if neighbor_dim in label_dims:
            raise ValueError(
                f"neighbor dimension {neighbor_dim!r} conflicts with the label "
                "dimensions, please choose another neighbor_dim value"
            )

        points = get_points(xr_labels[name] for name in self._coord_names)

        # non-finite query coordinates have no neighbor by definition
        valid = np.isfinite(points).all(axis=1)

        n_queries = points.shape[0]
        distances = np.full((n_queries, k), np.inf)
        indices = np.full((n_queries, k), -1, dtype=np.intp)

        if valid.any():
            d, i = self._tree_obj.query_neighbors(points[valid], k=k, radius=radius)
            distances[valid] = d
            indices[valid] = i

        mask = indices >= 0

        # clamp missing entries so that the indexers can never accidentally
        # select the last indexed sample (e.g., via a -1 index)
        safe_indices = np.where(mask, indices, 0)

        result_dims = tuple(label_dims) + (neighbor_dim,)
        result_shape = tuple(label_shape) + (k,)

        dim_indexers = self._get_dim_indexers(safe_indices, result_dims, result_shape)
        distances_da = DataArray(distances.reshape(result_shape), dims=result_dims)
        mask_da = DataArray(mask.reshape(result_shape), dims=result_dims)

        return NeighborsResult(
            dim_indexers=dim_indexers, distances=distances_da, mask=mask_da
        )

    def rename(
        self,
        name_dict: Mapping[Any, Hashable],
        dims_dict: Mapping[Any, Hashable],
    ) -> Self:
        if not set(self._coord_names) & set(name_dict) and not set(self._dims) & set(
            dims_dict
        ):
            return self

        new_coord_names = tuple(name_dict.get(n, n) for n in self._coord_names)
        new_dims = tuple(dims_dict.get(d, d) for d in self._dims)

        return type(self)(
            self._tree_obj,
            coord_names=new_coord_names,
            dims=new_dims,
            shape=self._shape,
        )

    def _repr_inline_(self, max_width: int) -> str:
        tree_obj_type = self._tree_obj.__class__.__name__
        return f"{self.__class__.__name__} ({tree_obj_type})"
