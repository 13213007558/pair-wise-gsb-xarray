from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from xarray.core.dataarray import DataArray
from xarray.core.indexes import Index, PandasIndex, as_scalar, normalize_label
from xarray.core.indexing import IndexSelResult
from xarray.core.utils import is_dict_like, is_scalar
from xarray.core.variable import Variable


class PeriodicIndex(Index):
    """Xarray index for periodic (circular) 1-dimensional coordinates such as
    longitude or day-of-year.

    A ``PeriodicIndex`` declares a ``period`` (e.g. 360 for longitude in
    degrees) and an ``origin`` (the label where the period starts, default 0).
    Label-based selection then wraps around the seam of the period instead of
    requiring the user to ``roll`` the whole dataset first:

    - slices that cross the seam (e.g. ``slice(350, 10)`` on a 0..359
      longitude coordinate) return the matching labels in the requested
      direction,
    - scalar and array-like labels are normalized into the period (e.g. label
      ``360`` matches coordinate label ``0`` when ``period=360`` and
      ``origin=0``),
    - ``method="nearest"`` computes distances on the circle, with ties
      resolved in favor of the smaller normalized label.

    Ascending, descending and non-uniform (but circularly monotonic)
    coordinates are supported. Coordinate labels must be unique modulo the
    period: equivalent labels such as ``0`` and ``360`` cannot both be
    present and raise an error when the index is created.

    Do not use :py:meth:`~xarray.indexes.PeriodicIndex.__init__` directly.
    Instead use :py:meth:`xarray.Dataset.set_xindex` with the ``period``
    (and optionally ``origin``) options::

        ds = ds.set_xindex("lon", PeriodicIndex, period=360.0)

    """

    _index: PandasIndex
    period: float
    origin: float

    def __init__(
        self,
        index: PandasIndex,
        period: float,
        origin: float = 0.0,
        *,
        _validate: bool = True,
    ) -> None:
        period = float(period)
        origin = float(origin)

        if not np.isfinite(period) or period <= 0:
            raise ValueError(
                f"period must be a positive, finite number, got {period!r}"
            )
        if not np.isfinite(origin):
            raise ValueError(f"origin must be a finite number, got {origin!r}")

        if not isinstance(index, PandasIndex):
            raise TypeError(
                "PeriodicIndex must be built from a PandasIndex, "
                f"got {type(index)!r}"
            )

        if index.index.dtype.kind not in "iuf":
            raise ValueError(
                "PeriodicIndex only supports numeric coordinate labels, "
                f"got dtype {index.index.dtype!r}"
            )

        self._index = index
        self.period = period
        self.origin = origin

        self._phase = self._compute_phase(index.index.values)
        if _validate:
            self._check_unique()
        self._direction = self._compute_direction()
        # helpers for exact and nearest lookup
        self._positions_by_phase = {
            phase: pos for pos, phase in enumerate(self._phase.tolist())
        }
        self._sorted_order = np.argsort(self._phase, kind="stable")

    @classmethod
    def from_variables(
        cls,
        variables: Mapping[Any, Variable],
        *,
        options: Mapping[str, Any],
    ) -> PeriodicIndex:
        if "period" not in options:
            raise TypeError(
                "PeriodicIndex requires a 'period' option, e.g., "
                "`ds.set_xindex('lon', PeriodicIndex, period=360.0)`"
            )
        period = options["period"]
        origin = options.get("origin", 0.0)

        pd_index = PandasIndex.from_variables(variables, options={})
        return cls(pd_index, period=period, origin=origin)

    @property
    def index(self) -> pd.Index:
        """The wrapped pandas.Index with the original (unwrapped) labels."""
        return self._index.index

    @property
    def dim(self) -> Hashable:
        return self._index.dim

    @property
    def coord_name(self) -> Hashable:
        return self._index.index.name

    @property
    def coord_dtype(self) -> Any:
        return self._index.coord_dtype

    def _compute_phase(self, values: np.ndarray) -> np.ndarray:
        """Normalize labels to their phase in the half-open interval
        ``[0, period)`` relative to ``origin``.
        """
        return np.mod(np.asarray(values, dtype=np.float64) - self.origin, self.period)

    def _check_unique(self) -> None:
        if len(self._phase) == 0:
            return
        _, counts = np.unique(self._phase, return_counts=True)
        if np.any(counts > 1):
            dup_phase = self._phase[np.argmax(counts > 1)]
            dup_labels = self._index.index.values[self._phase == dup_phase]
            raise ValueError(
                "coordinate labels are ambiguous for a periodic index: labels "
                f"{dup_labels.tolist()!r} are all equivalent to "
                f"{dup_phase + self.origin!r} modulo period {self.period!r}. "
                "Remove or disambiguate equivalent labels (e.g., both 0 and "
                "360) before creating a PeriodicIndex."
            )

    def _compute_direction(self) -> int:
        """Return +1 (-1) if the coordinate is circularly ascending
        (descending) along the array.
        """
        values = np.asarray(self._index.index.values, dtype=np.float64)
        if values.size < 2:
            return 1
        diffs = np.diff(values)
        forward = np.sum(np.mod(diffs, self.period))
        backward = np.sum(np.mod(-diffs, self.period))
        return 1 if forward <= backward else -1

    def _wrap(self, index: PandasIndex) -> PeriodicIndex:
        # no re-validation: indexing a valid periodic coordinate may yield
        # repeated labels (e.g., selecting the same label twice), which the
        # user explicitly requested
        return type(self)(
            index=index, period=self.period, origin=self.origin, _validate=False
        )

    def create_variables(
        self, variables: Mapping[Any, Variable] | None = None
    ) -> dict[Hashable, Variable]:
        return self._index.create_variables(variables)

    def to_pandas_index(self) -> pd.Index:
        return self._index.index

    def isel(
        self, indexers: Mapping[Any, int | slice | np.ndarray | Variable]
    ) -> PeriodicIndex | None:
        idxer = indexers[self.dim]
        if isinstance(idxer, Variable):
            if idxer.dims != (self.dim,):
                # can't preserve an index if result has new dimensions
                return None
            idxer = idxer.data
        if not isinstance(idxer, slice) and is_scalar(idxer):
            # scalar indexer: drop index
            return None
        if isinstance(idxer, slice) and idxer == slice(None):
            return self

        new_pd_index = self._index._replace(self._index.index[idxer])
        return self._wrap(new_pd_index)

    def _slice_indexer(self, label: slice) -> slice | np.ndarray:
        size = self._phase.size
        start, stop, step = label.start, label.stop, label.step

        if start is None and stop is None:
            return slice(None, None, step)
        if size == 0:
            return np.array([], dtype=np.intp)

        direction = self._direction
        period = self.period
        # arc position of each label along the coordinate direction, relative
        # to the first label, in [0, period)
        arc = ((self._phase - self._phase[0]) * direction) % period
        tol = period * 1e-9

        if start is None:
            # from the start of the array to the stop label
            high = (
                (self._compute_phase(np.asarray(stop)) - self._phase[0])
                * direction
            ) % period
            mask = arc <= high + tol
            order_key = arc
        else:
            low = (
                (self._compute_phase(np.asarray(start)) - self._phase[0])
                * direction
            ) % period
            if stop is None:
                # from the start label to the end of the array
                mask = arc >= low - tol
                order_key = arc
            else:
                span_raw = (float(stop) - float(start)) * direction
                if span_raw < 0:
                    span = span_raw % period
                else:
                    span = min(span_raw, period)
                high = low + span
                in_arc = (arc >= low - tol) & (arc <= high + tol)
                # labels just past the seam also match the arc shifted by one
                # full period
                wrapped = (~in_arc) & (arc <= high - period + tol)
                mask = in_arc | wrapped
                order_key = np.where(in_arc, arc, arc + period)

        positions = np.nonzero(mask)[0]
        positions = positions[np.argsort(order_key[positions], kind="stable")]
        if step is not None:
            positions = positions[::step]
        return positions

    def _lookup_exact(self, labels: np.ndarray) -> np.ndarray:
        phases = self._compute_phase(labels)
        flat = np.ravel(phases)
        indexer = np.array(
            [self._positions_by_phase.get(phase, -1) for phase in flat],
            dtype=np.intp,
        )
        return indexer.reshape(phases.shape)

    def _lookup_nearest(self, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the positions of the nearest labels and their circular
        distances to the given labels.

        Distances are computed on the circle (in ``[0, period / 2]``). Ties
        are resolved in favor of the smaller normalized label.
        """
        phases = self._compute_phase(labels)
        flat = np.ravel(phases)
        sorted_phase = self._phase[self._sorted_order]
        size = sorted_phase.size

        insert = np.searchsorted(sorted_phase, flat)
        succ = insert % size
        pred = (insert - 1) % size
        dist_succ = (sorted_phase[succ] - flat) % self.period
        dist_pred = (flat - sorted_phase[pred]) % self.period

        # on a tie (dist_pred == dist_succ) the predecessor, i.e., the smaller
        # normalized label, wins
        choose_pred = dist_pred <= dist_succ
        nearest_sorted = np.where(choose_pred, pred, succ)
        distances = np.where(choose_pred, dist_pred, dist_succ)

        indexer = self._sorted_order[nearest_sorted].reshape(phases.shape)
        return indexer, distances.reshape(phases.shape)

    def sel(
        self, labels: dict[Any, Any], method=None, tolerance=None
    ) -> IndexSelResult:
        if method is not None and not isinstance(method, str):
            raise TypeError("``method`` must be a string")
        if method not in (None, "nearest"):
            raise ValueError(
                "PeriodicIndex only supports selection with method='nearest', "
                f"got method={method!r}"
            )

        assert len(labels) == 1
        coord_name, label = next(iter(labels.items()))

        if isinstance(label, slice):
            if method is not None or tolerance is not None:
                raise ValueError(
                    "cannot use ``method`` or ``tolerance`` with slice selection"
                )
            slice_indexer = self._slice_indexer(label)
            return IndexSelResult({self.dim: slice_indexer})

        if is_dict_like(label):
            raise ValueError(
                "cannot use a dict-like object for selection on "
                "a dimension that does not have a MultiIndex"
            )

        if self._phase.size == 0:
            raise KeyError(f"not all values found in index {coord_name!r}")

        label_array = normalize_label(label, dtype=self.coord_dtype)
        indexer: Any

        if label_array.ndim == 0:
            if method is None:
                indexer = self._lookup_exact(label_array)
                if indexer < 0:
                    raise KeyError(
                        f"not all values found in index {coord_name!r}. "
                        "Try setting the `method` keyword argument "
                        "(example: method='nearest')."
                    )
                indexer = int(indexer)
            else:
                indexer_1d, distances = self._lookup_nearest(
                    np.atleast_1d(label_array)
                )
                if tolerance is not None and distances[0] > tolerance:
                    raise KeyError(f"not all values found in index {coord_name!r}")
                indexer = int(indexer_1d[0])
        elif label_array.dtype.kind == "b":
            indexer = label_array
        else:
            if method is None:
                indexer = self._lookup_exact(label_array)
            else:
                indexer, distances = self._lookup_nearest(label_array)
                if tolerance is not None:
                    indexer = np.where(distances <= tolerance, indexer, -1)
            if np.any(indexer < 0):
                raise KeyError(f"not all values found in index {coord_name!r}")

        # attach dimension names and/or coordinates to positional indexer
        if isinstance(label, Variable):
            indexer = Variable(label.dims, indexer)
        elif isinstance(label, DataArray):
            indexer = DataArray(indexer, coords=label._coords, dims=label.dims)

        return IndexSelResult({self.dim: indexer})

    def equals(
        self, other: Index, *, exclude: frozenset[Hashable] | None = None
    ) -> bool:
        if not isinstance(other, PeriodicIndex):
            return False
        return (
            self.period == other.period
            and self.origin == other.origin
            and self._index.equals(other._index)
        )

    def join(self, other: PeriodicIndex, how: str = "inner") -> PeriodicIndex:
        if not isinstance(other, PeriodicIndex):
            raise TypeError(
                f"cannot join a PeriodicIndex with {type(other)!r}"
            )
        if self.period != other.period or self.origin != other.origin:
            raise ValueError(
                "cannot join PeriodicIndex objects with different period or "
                "origin"
            )
        return self._wrap(self._index.join(other._index, how=how))

    def reindex_like(
        self, other: PeriodicIndex, method=None, tolerance=None
    ) -> dict[Hashable, Any]:
        if not isinstance(other, PeriodicIndex):
            raise TypeError(
                f"cannot reindex a PeriodicIndex against {type(other)!r}"
            )
        if method is None:
            indexer = self._lookup_exact(other.index.values)
        else:
            indexer, distances = self._lookup_nearest(other.index.values)
            if tolerance is not None:
                indexer = np.where(distances <= tolerance, indexer, -1)
        return {self.dim: indexer}

    def roll(self, shifts: Mapping[Any, int]) -> PeriodicIndex:
        return self._wrap(self._index.roll(shifts))

    def rename(self, name_dict, dims_dict) -> PeriodicIndex:
        new_index = self._index.rename(name_dict, dims_dict)
        if new_index is self._index:
            return self
        return self._wrap(new_index)

    def __getitem__(self, indexer: Any) -> PeriodicIndex:
        return self._wrap(self._index[indexer])

    def _repr_inline_(self, max_width: int) -> str:
        params_fmt = f"period={self.period:.6g}, origin={self.origin:.6g}"
        return f"{self.__class__.__name__} ({params_fmt})"

    def __repr__(self) -> str:
        params_fmt = (
            f"period={self.period:.6g}, origin={self.origin:.6g}, "
            f"coord_name={self.coord_name!r}, dim={self.dim!r}"
        )
        return f"{self.__class__.__name__} ({params_fmt})"
