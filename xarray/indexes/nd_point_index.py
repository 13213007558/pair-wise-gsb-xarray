from __future__ import annotations

import abc
import math
from collections.abc import Hashable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, TypeVar

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

    def query_neighbors(
        self, points: np.ndarray, k: int, radius: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Query the up-to ``k`` nearest neighbors of points within a radius.

        This method is optional: 3rd-party tree adapters are not required to
        implement it. The default implementation raises
        :py:class:`NotImplementedError`.

        Parameters
        ----------
        points: ndarray of shape (n_points, n_coordinates)
            Two-dimensional array of points/samples (rows) and their
            corresponding coordinate labels (columns) to query. Every row is
            assumed to contain only finite values.
        k : int
            Maximum number of neighbors to return for each query point.
        radius : float
            Maximum distance (inclusive) between a query point and its
            returned neighbors. May be `np.inf` for an unlimited radius.

        Returns
        -------
        distances : ndarray of shape (n_points, k)
            Distances to the neighbors, sorted in ascending order for each
            query point. Ties are ordered by the position of the neighbors in
            the array of the indexed points. Missing entries (fewer than ``k``
            neighbors found within ``radius``) are set to `np.inf`.
        indices : ndarray of shape (n_points, k)
            Indices of the neighbors in the array of the indexed points.
            Missing entries are set to `-1`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support neighbor queries "
            "(query_neighbors is not implemented for this tree adapter)"
        )

    def equals(self, other: Self) -> bool:
        """Check equality with another TreeAdapter of the same kind.

        Parameters
        ----------
        other :
            The other TreeAdapter object to compare with this object.

        """
        raise NotImplementedError


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

    def query_neighbors(
        self, points: np.ndarray, k: int, radius: float
    ) -> tuple[np.ndarray, np.ndarray]:
        n_indexed = self._kdtree.n
        if n_indexed == 0:
            return (
                np.full((len(points), k), np.inf),
                np.full((len(points), k), -1, dtype=np.intp),
            )

        # query one extra neighbor (if any) in order to detect (nearly)
        # equidistant neighbors at the k-th position boundary
        detect_ambiguity = k < n_indexed
        k_query = k + 1 if detect_ambiguity else k

        # scipy.spatial.KDTree.query() excludes points at exactly
        # distance_upper_bound: use the next representable float towards
        # infinity to make the radius limit inclusive
        distance_upper_bound = np.nextafter(radius, np.inf)

        distances, indices = self._kdtree.query(
            points, k=k_query, distance_upper_bound=distance_upper_bound
        )

        distances = np.asarray(distances, dtype=float).reshape(len(points), k_query)
        indices = np.asarray(indices, dtype=np.intp).reshape(len(points), k_query)

        # scipy.spatial.KDTree.query() marks missing neighbors with an index
        # equal to the number of indexed points: normalize those entries
        missing = indices >= n_indexed
        indices = np.where(missing, -1, indices)
        distances = np.where(missing, np.inf, distances)

        # order neighbors by (distance, original position) so that
        # equidistant neighbors are sorted by their position in the array of
        # the indexed points (stable two-pass sort)
        order = np.argsort(indices, axis=1, kind="stable")
        distances = np.take_along_axis(distances, order, axis=1)
        indices = np.take_along_axis(indices, order, axis=1)
        order = np.argsort(distances, axis=1, kind="stable")
        distances = np.take_along_axis(distances, order, axis=1)
        indices = np.take_along_axis(indices, order, axis=1)

        if detect_ambiguity:
            # the distances computed internally by the k-d tree may differ by
            # a few ulps from an explicit computation: re-select the neighbors
            # of the rows with (nearly) tied distances using an exact
            # brute-force computation, so that equidistant neighbors are
            # selected and ordered deterministically by their position in the
            # array of the indexed points
            with np.errstate(invalid="ignore"):
                # inf - inf may produce nan gaps, which never compare <= tol
                gaps = np.diff(distances, axis=1)
            tol = (
                16 * np.finfo(distances.dtype).eps * np.maximum(distances[:, :-1], 1.0)
            )
            ambiguous = np.nonzero(
                (gaps <= tol).any(axis=1) & np.isfinite(distances[:, -2])
            )[0]
            if len(ambiguous):
                self._reselect_ambiguous_rows(
                    points, distances, indices, ambiguous, radius
                )

        distances = distances[:, :k]
        indices = indices[:, :k]

        return distances, indices

    def _reselect_ambiguous_rows(
        self,
        points: np.ndarray,
        distances: np.ndarray,
        indices: np.ndarray,
        rows: np.ndarray,
        radius: float,
    ) -> None:
        """Re-select the neighbors of the given rows using an exact
        brute-force distance computation.

        """
        data = self._kdtree.data
        k = indices.shape[1]
        for row in rows:
            dist_row = np.sqrt(((data - points[row]) ** 2).sum(axis=1))
            dist_row = np.where(dist_row <= radius, dist_row, np.inf)
            # stable sort: equidistant neighbors are selected and ordered by
            # their position in the array of the indexed points
            selected = np.argsort(dist_row, kind="stable")[:k]
            selected_dist = dist_row[selected]
            hit = np.isfinite(selected_dist)
            distances[row] = np.where(hit, selected_dist, np.inf)
            indices[row] = np.where(hit, selected, -1)

    def equals(self, other: Self) -> bool:
        return np.array_equal(self._kdtree.data, other._kdtree.data)


def get_points(coords: Iterable[Variable | Any]) -> np.ndarray:
    """Re-arrange data from a sequence of xarray coordinate variables or
    labels into a 2-d array of shape (n_points, n_coordinates).

    """
    data = [c.values if isinstance(c, Variable | DataArray) else c for c in coords]
    return np.stack([np.ravel(d) for d in data]).T


T_TreeAdapter = TypeVar("T_TreeAdapter", bound=TreeAdapter)


class NeighborQueryResult(NamedTuple):
    """Result of :py:meth:`NDPointIndex.query_neighbors`.

    Attributes
    ----------
    dim_indexers : dict
        Mapping of dimension names to :py:class:`~xarray.Variable` objects
        that can be passed to :py:meth:`xarray.Dataset.isel` or
        :py:meth:`xarray.DataArray.isel` in order to extract the data of the
        neighbors. Each variable has the broadcast dimensions of the query
        labels plus an extra neighbor dimension (in last position). Missing
        neighbors (fewer than ``k`` neighbors found within
        ``radius``) are arbitrarily assigned to the first
        indexed point: use ``mask`` to identify them.
    distances : DataArray
        Distances to the neighbors, with the broadcast dimensions of the
        query labels plus the neighbor dimension. Missing neighbors have a
        distance of `np.inf`.
    mask : DataArray
        Boolean array (same dimensions as ``distances``)
        indicating which neighbor entries are actual hits. It is `False`
        for missing neighbors, e.g., where fewer than ``k``
        neighbors were found within ``radius``.
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
        label_dims: tuple[Hashable, ...],
        label_shape: tuple[int, ...],
    ) -> dict[Hashable, Variable]:
        """Returns dimension indexers based on the query results (indices) and
        the original label dimensions and shape.

        1. Unravel the flat indices returned from the query
        2. Reshape the unraveled indices according to indexers shapes
        3. Wrap the indices in xarray.Variable objects.

        """
        dim_indexers = {}

        u_indices = list(np.unravel_index(indices.ravel(), self._shape))

        for dim, ind in zip(self._dims, u_indices, strict=False):
            dim_indexers[dim] = Variable(label_dims, ind.reshape(label_shape))

        return dim_indexers

    def _get_points_and_label_meta(
        self, labels: dict[Any, Any]
    ) -> tuple[np.ndarray, tuple[Hashable, ...], tuple[int, ...]]:
        """Convert query labels into a 2-d array of points and return it along
        with the broadcast label dimensions and shape.

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

        points = get_points(xr_labels[name] for name in self._coord_names)

        return points, label_dims, label_shape

    def sel(
        self, labels: dict[Any, Any], method=None, tolerance=None
    ) -> IndexSelResult:
        if method != "nearest":
            raise ValueError(
                "NDPointIndex only supports selection with method='nearest'"
            )

        points, label_dims, label_shape = self._get_points_and_label_meta(labels)

        # get and return dimension indexers
        _, indices = self._tree_obj.query(points)

        dim_indexers = self._get_dim_indexers(indices, label_dims, label_shape)

        return IndexSelResult(dim_indexers=dim_indexers)

    def query_neighbors(
        self,
        labels: dict[Any, Any],
        *,
        k: int,
        radius: float = np.inf,
        neighbor_dim: Hashable = "neighbor",
    ) -> NeighborQueryResult:
        """Query the up-to ``k`` nearest neighbors of the given
        labels within ``radius``.

        Unlike :py:meth:`NDPointIndex.sel`, this method is not limited to
        the single nearest neighbor of each query point. It accepts the same
        kinds of labels as :py:meth:`NDPointIndex.sel` (scalars,
        array-likes or xarray objects, broadcast against each other) and
        preserves their broadcast dimensions, to which it appends an extra
        neighbor dimension of size ``k``.

        Parameters
        ----------
        labels : dict
            Mapping of coordinate names to query labels, like
            :py:meth:`NDPointIndex.sel`. Labels for all the coordinate
            variables associated with this index must be provided.
        k : int
            Maximum number of neighbors to return for each query point. Must
            be a positive integer. It may be greater than the total number of
            indexed points, in which case the extra neighbor entries are
            marked as missing.
        radius : float, default: np.inf
            Maximum distance (inclusive) between a query point and its
            returned neighbors. Must be a non-negative scalar. The default
            (infinite radius) corresponds to a plain k-nearest-neighbor query.
        neighbor_dim : Hashable, default: "neighbor"
            Name of the extra neighbor dimension in the returned result. Must
            not conflict with any of the broadcast label dimensions.

        Returns
        -------
        NeighborQueryResult
            A named tuple with the dimension indexers of the neighbors (to
            pass to :py:meth:`xarray.Dataset.isel` or
            :py:meth:`xarray.DataArray.isel`), the distances to the
            neighbors and a boolean mask indicating which neighbor entries
            are actual hits.

            Neighbors are ordered by increasing distance for each query
            point. Equidistant neighbors are ordered by their position in
            the indexed coordinates (stable sort), so duplicated coordinate
            points each keep their own data. Missing neighbors (fewer than
            ``k`` neighbors found within ``radius``,
            non-finite query labels or an empty index) have a distance of
            `np.inf` and are flagged as `False` in the mask. Their
            dimension indexers are arbitrarily set to the first indexed
            point: always use the mask to discard them (e.g., via
            :py:meth:`xarray.Dataset.where`) instead of relying on the
            indexed values.

        Examples
        --------
        >>> xx = [[1.0, 2.0], [3.0, 0.0]]
        >>> yy = [[11.0, 21.0], [29.0, 9.0]]
        >>> ds = xr.Dataset(
        ...     {"temp": (("y", "x"), [[45.1, 53.3], [65.4, 78.2]])},
        ...     coords={"xx": (("y", "x"), xx), "yy": (("y", "x"), yy)},
        ... )
        >>> ds = ds.set_xindex(("xx", "yy"), xr.indexes.NDPointIndex)
        >>> index = ds.xindexes["xx"]
        >>> result = index.query_neighbors(
        ...     {
        ...         "xx": xr.Variable("points", [1.9, 0.1]),
        ...         "yy": xr.Variable("points", [13.0, 8.0]),
        ...     },
        ...     k=2,
        ...     radius=20.0,
        ... )
        >>> result.distances
        <xarray.DataArray (points: 2, neighbor: 2)> Size: 32B
        array([[2.19317122, 4.42831797],
               [1.00498756, 3.13209195]])
        Dimensions without coordinates: points, neighbor
        >>> result.mask
        <xarray.DataArray (points: 2, neighbor: 2)> Size: 4B
        array([[ True,  True],
               [ True,  True]])
        Dimensions without coordinates: points, neighbor
        >>> ds.isel(result.dim_indexers).where(result.mask).temp
        <xarray.DataArray 'temp' (points: 2, neighbor: 2)> Size: 32B
        array([[45.1, 78.2],
               [78.2, 45.1]])
        Coordinates:
            xx       (points, neighbor) float64 32B 1.0 0.0 0.0 1.0
            yy       (points, neighbor) float64 32B 11.0 9.0 9.0 11.0
        Dimensions without coordinates: points, neighbor

        """
        if isinstance(k, bool) or not isinstance(k, int | np.integer) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        k = int(k)

        if not is_scalar(radius):
            raise ValueError(f"radius must be a scalar number, got {radius!r}")
        radius = float(radius)
        if np.isnan(radius) or radius < 0:
            raise ValueError(f"radius must be a non-negative number, got {radius!r}")

        points, label_dims, label_shape = self._get_points_and_label_meta(labels)

        if neighbor_dim in label_dims:
            raise ValueError(
                f"neighbor_dim {neighbor_dim!r} conflicts with the broadcast "
                f"label dimensions {label_dims}"
            )

        index_dims = (*label_dims, neighbor_dim)
        index_shape = (*label_shape, k)

        n_indexed = math.prod(self._shape)
        n_queries = points.shape[0]

        distances = np.full((n_queries, k), np.inf)
        indices = np.full((n_queries, k), -1, dtype=np.intp)

        if n_indexed > 0:
            # only query finite points: non-finite query labels are invalid
            # and always yield missing neighbors
            valid = np.isfinite(points).all(axis=1)
            if valid.any():
                q_distances, q_indices = self._tree_obj.query_neighbors(
                    points[valid], k, radius
                )
                expected_shape = (int(valid.sum()), k)
                if (
                    q_distances.shape != expected_shape
                    or q_indices.shape != expected_shape
                ):
                    raise ValueError(
                        f"tree adapter {type(self._tree_obj).__name__} returned "
                        f"invalid shapes {q_distances.shape} and {q_indices.shape}, "
                        f"expected {expected_shape}"
                    )
                distances[valid] = q_distances
                indices[valid] = q_indices

        mask = indices != -1

        # never keep -1 indices (which would silently select the last indexed
        # point): use a safe dummy index for missing neighbors and rely on the
        # explicit mask to identify them
        safe_indices = np.where(mask, indices, 0)

        if n_indexed > 0:
            dim_indexers = self._get_dim_indexers(safe_indices, index_dims, index_shape)
        else:
            dim_indexers = {
                dim: Variable(index_dims, np.zeros(index_shape, dtype=np.intp))
                for dim in self._dims
            }

        return NeighborQueryResult(
            dim_indexers=dim_indexers,
            distances=DataArray(distances.reshape(index_shape), dims=index_dims),
            mask=DataArray(mask.reshape(index_shape), dims=index_dims),
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
