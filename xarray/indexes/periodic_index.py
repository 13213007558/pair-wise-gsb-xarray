from __future__ import annotations

import operator
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from xarray.core.dataarray import DataArray
from xarray.core.indexes import (
    Index,
    PandasIndex,
    as_scalar,
    normalize_label,
    safe_cast_to_index,
    _sanitize_slice_element,
)
from xarray.core.indexing import IndexSelResult
from xarray.core.utils import is_scalar
from xarray.core.variable import Variable


class PeriodicIndex(PandasIndex):
    """Wrap a one-dimensional :class:`pandas.Index` with periodic labels.

    Labels are interpreted on a circle with a fixed ``period`` and ``origin``.
    Positive label slices follow increasing positions around that circle;
    negative slices follow decreasing positions. This allows a slice to cross
    the period boundary without calling :py:meth:`~xarray.Dataset.roll` first.

    Query labels are normalized to the half-open interval
    ``[origin, origin + period)``. Coordinate labels are returned unchanged.
    For example, with ``period=360`` and ``origin=0``, selecting ``360`` or
    ``-360`` returns a coordinate labeled ``0``.

    Coordinate labels must be unique after periodic normalization. A
    coordinate containing both ``0`` and ``360`` is rejected. The ambiguity
    must be removed before creating the index.

    Circular distances are used with ``method='nearest'``. If two labels are
    equally close, the label in the positive direction around the circle is
    selected.
    """

    period: Any
    origin: Any

    __slots__ = ('origin', 'period')

    def __init__(
        self,
        array: Any,
        dim: Hashable,
        coord_dtype: Any = None,
        *,
        period: Any,
        origin: Any = 0,
        fastpath: bool = False,
    ) -> None:
        super().__init__(array, dim, coord_dtype, fastpath=fastpath)

        if isinstance(self.index, pd.MultiIndex):
            raise ValueError('PeriodicIndex only supports a one-dimensional index')
        if self.index.dtype.kind not in 'iuf':
            raise TypeError('PeriodicIndex only supports numeric labels')

        dtype = np.result_type(self.index.dtype, period, origin)
        if np.asarray(period, dtype=dtype) <= 0:
            raise ValueError('period must be greater than zero')

        self.period = dtype.type(period)
        self.origin = dtype.type(origin)

        if not fastpath:
            self._validate_labels()

    def _replace(self, index, dim=None, coord_dtype=None, period=None, origin=None):
        return type(self)(
            index,
            self.dim if dim is None else dim,
            self.coord_dtype if coord_dtype is None else coord_dtype,
            period=self.period if period is None else period,
            origin=self.origin if origin is None else origin,
            fastpath=True,
        )

    @classmethod
    def from_variables(
        cls,
        variables: Mapping[Any, Variable],
        *,
        options: Mapping[str, Any],
    ) -> 'PeriodicIndex':
        if len(variables) != 1:
            raise ValueError(
                f'PeriodicIndex accepts one variable, found {len(variables)}'
            )

        name, var = next(iter(variables.items()))
        if var.ndim == 0:
            raise ValueError(f'cannot create PeriodicIndex from scalar {name!r}')
        if var.ndim != 1:
            raise ValueError(f'PeriodicIndex requires 1-dimensional variable {name!r}')
        if 'period' not in options:
            raise ValueError('PeriodicIndex requires a period option')

        index = safe_cast_to_index(var.data)
        obj = cls(
            index,
            var.dims[0],
            coord_dtype=var.dtype,
            period=options['period'],
            origin=options.get('origin', 0),
        )
        obj.index = obj.index.copy()
        obj.index.name = name
        return obj

    @classmethod
    def concat(
        cls,
        indexes: Sequence['PeriodicIndex'],
        dim: Hashable,
        positions=None,
    ) -> 'PeriodicIndex':
        if indexes:
            period = indexes[0].period
            origin = indexes[0].origin
            if any(item.period != period or item.origin != origin for item in indexes):
                raise ValueError(
                    'cannot concatenate PeriodicIndex objects with different '
                    'periods or origins'
                )
        else:
            period, origin = 1, 0

        new_index = PandasIndex._concat_indexes(indexes, dim, positions)
        return cls(new_index, dim, period=period, origin=origin)

    @property
    def _numeric_dtype(self):
        return np.result_type(self.coord_dtype, self.period, self.origin)

    def _comparison_dtype(self, values=0):
        return np.result_type(
            np.float64, self._numeric_dtype, np.asarray(values).dtype
        )

    def _canonicalize(self, values, *, floating: bool = False) -> np.ndarray:
        if floating:
            dtype = self._comparison_dtype(values)
        else:
            dtype = np.result_type(self._numeric_dtype, np.asarray(values).dtype)
        values = np.asarray(values, dtype=dtype)
        period = np.asarray(self.period, dtype=dtype)
        origin = np.asarray(self.origin, dtype=dtype)
        return origin + np.mod(values - origin, period)

    def _canonicalize_for_comparison(self, values) -> np.ndarray:
        return self._canonicalize(values, floating=True)

    def _validate_labels(self) -> None:
        values = self.index.to_numpy(dtype=self._numeric_dtype, copy=False)
        if not np.all(np.isfinite(values)):
            raise ValueError('PeriodicIndex labels must all be finite')

        canonical = self._canonicalize_for_comparison(values)
        duplicated = pd.Index(canonical).duplicated(keep=False)
        if np.any(duplicated):
            raise ValueError(
                'PeriodicIndex requires labels unique on the declared period; '
                f'found equivalent labels {values[duplicated].tolist()} at '
                f'positions {np.flatnonzero(duplicated).tolist()}. Resolve this '
                'ambiguity before creating the index.'
            )

    def _ordered_labels(self):
        values = self.index.to_numpy(dtype=self._numeric_dtype, copy=False)
        canonical = self._canonicalize_for_comparison(values)
        order = np.argsort(canonical, kind='stable')
        return canonical, order

    def _query_exact(self, query: np.ndarray) -> np.ndarray:
        canonical, order = self._ordered_labels()
        sorted_labels = canonical[order]
        locations = np.searchsorted(sorted_labels, query)
        in_range = locations < len(order)
        safe_locations = np.where(in_range, locations, 0)
        matches = np.zeros(query.shape, dtype=bool)
        matches[in_range] = sorted_labels[safe_locations[in_range]] == query[in_range]

        indexer = np.full(query.shape, -1, dtype=np.intp)
        indexer[matches] = order[safe_locations[matches]]
        return indexer

    def _query_nearest(self, query, method, tolerance):
        if self.index.size == 0:
            raise ValueError('cannot select from an empty PeriodicIndex')

        canonical, order = self._ordered_labels()
        sorted_labels = canonical[order]
        insertions = np.searchsorted(sorted_labels, query)
        next_positions = order[insertions % len(order)]
        previous_positions = order[(insertions - 1) % len(order)]
        period = np.asarray(self.period, dtype=query.dtype)
        forward = np.mod(canonical[next_positions] - query, period)
        backward = np.mod(query - canonical[previous_positions], period)

        if method in {'pad', 'ffill'}:
            positions, distances = previous_positions, backward
        elif method in {'backfill', 'bfill'}:
            positions, distances = next_positions, forward
        else:
            choose_next = forward <= backward
            positions = np.where(choose_next, next_positions, previous_positions)
            distances = np.minimum(forward, backward)

        if tolerance is not None:
            tolerance_array = np.asarray(tolerance, dtype=distances.dtype)
            if np.any(tolerance_array < 0):
                raise ValueError('tolerance must be non-negative')
            if np.any(distances > tolerance_array):
                raise KeyError('not all values found within the requested tolerance')
        return positions

    def _attach_dimensions(self, label, indexer):
        if isinstance(label, Variable):
            return Variable(label.dims, indexer)
        if isinstance(label, DataArray):
            return DataArray(indexer, coords=label._coords, dims=label.dims)
        return indexer

    def _slice_bound(self, value):
        scalar = _sanitize_slice_element(value)
        return self._canonicalize_for_comparison(scalar)[()]

    def _query_slice(self, label: slice):
        if label.step is None:
            step = 1
        else:
            step = operator.index(label.step)
            if step == 0:
                raise ValueError('slice step cannot be zero')

        canonical, order = self._ordered_labels()
        selected = np.ones(canonical.shape, dtype=bool)
        if step > 0:
            if label.start is None and label.stop is None:
                result = order
            else:
                start = stop = None
                if label.start is not None:
                    start = self._slice_bound(label.start)
                    selected &= canonical >= start
                if label.stop is not None:
                    stop = self._slice_bound(label.stop)
                    selected &= canonical <= stop

                if start is not None and stop is not None and start > stop:
                    first = order[canonical[order] >= start]
                    second = order[canonical[order] <= stop]
                    result = np.concatenate([first, second])
                else:
                    result = order[selected[order]]
        else:
            reverse = order[::-1]
            if label.start is None and label.stop is None:
                result = reverse
            else:
                    start = stop = None
                    if label.start is not None:
                        start = self._slice_bound(label.start)
                        selected &= canonical <= start
                    if label.stop is not None:
                        stop = self._slice_bound(label.stop)
                        selected &= canonical >= stop

                    if start is not None and stop is not None and start < stop:
                        first = reverse[canonical[reverse] <= start]
                        second = reverse[canonical[reverse] >= stop]
                        result = np.concatenate([first, second])
                    else:
                        result = reverse[selected[reverse]]

        return result[:: abs(step)]

    def sel(self, labels, method=None, tolerance=None):
        if method is not None and not isinstance(method, str):
            raise TypeError('method must be a string')
        allowed = {None, 'nearest', 'pad', 'ffill', 'backfill', 'bfill'}
        if method not in allowed:
            raise ValueError(
                "PeriodicIndex supports nearest, pad/ffill and backfill/bfill"
            )

        coord_name, label = next(iter(labels.items()))

        if isinstance(label, slice):
            if method is not None or tolerance is not None:
                raise ValueError('method and tolerance are not supported with slices')
            indexer = self._query_slice(label)
        else:
            values = normalize_label(label, dtype=self.coord_dtype)
            if values.dtype.kind == 'b':
                indexer = values
            else:
                query = self._canonicalize_for_comparison(values)
                if not np.all(np.isfinite(query)):
                    raise ValueError('periodic selection labels must all be finite')

                if method is None:
                    indexer = self._query_exact(query)
                    if np.any(indexer < 0):
                        raise KeyError(
                            f'not all values found in periodic index {coord_name!r}'
                        )
                else:
                    indexer = self._query_nearest(query, method, tolerance)

                if query.ndim == 0 and not isinstance(label, Variable | DataArray):
                    indexer = int(as_scalar(indexer))
                else:
                    indexer = self._attach_dimensions(label, indexer)

        return IndexSelResult({self.dim: indexer})

    def isel(self, indexers):
        indexer = indexers[self.dim]
        if isinstance(indexer, Variable):
            if indexer.dims != (self.dim,):
                return None
            indexer = indexer.data

        if not isinstance(indexer, slice) and is_scalar(indexer):
            return None
        if isinstance(indexer, slice) and indexer == slice(None):
            return self

        return self._replace(self.index[indexer])

    def equals(self, other: Index, *, exclude=None) -> bool:
        return (
            isinstance(other, PeriodicIndex)
            and super().equals(other, exclude=exclude)
            and self.period == other.period
            and self.origin == other.origin
        )

    def join(self, other: 'PeriodicIndex', how: str = 'inner') -> 'PeriodicIndex':
        if self.period != other.period or self.origin != other.origin:
            raise ValueError(
                'cannot align PeriodicIndex objects with different periods or origins'
            )

        self_values = self.index.to_numpy()
        other_values = other.index.to_numpy()
        self_canonical = self._canonicalize_for_comparison(self_values)
        other_canonical = other._canonicalize_for_comparison(other_values)

        if how == 'outer':
            values = np.concatenate(
                [self_values, other_values[~np.isin(other_canonical, self_canonical)]]
            )
        elif how == 'inner':
            values = self_values[np.isin(self_canonical, other_canonical)]
        else:
            raise ValueError(f'unsupported join method {how!r}')

        return type(self)(
            pd.Index(values, name=self.index.name),
            self.dim,
            self.coord_dtype,
            period=self.period,
            origin=self.origin,
        )

    def reindex_like(self, other, method=None, tolerance=None):
        if method is None:
            query = other._canonicalize_for_comparison(other.index.to_numpy())
            return {self.dim: self._query_exact(query)}
        return {
            self.dim: self._query_nearest(
                other._canonicalize_for_comparison(other.index.to_numpy()),
                method,
                tolerance,
            )
        }

    def roll(self, shifts):
        shift = shifts[self.dim] % self.index.shape[0]
        if shift:
            new_index = self.index[-shift:].append(self.index[:-shift])
        else:
            new_index = self.index[:]
        return self._replace(new_index)

    def rename(self, name_dict, dims_dict):
        if self.index.name not in name_dict and self.dim not in dims_dict:
            return self

        new_name = name_dict.get(self.index.name, self.index.name)
        new_dim = dims_dict.get(self.dim, self.dim)
        return self._replace(self.index.rename(new_name), dim=new_dim)

    def _copy(self, deep: bool = True, memo=None):
        index = self.index.copy(deep=True) if deep else self.index
        return self._replace(index)

    def __getitem__(self, indexer):
        return self._replace(self.index[indexer])

    def __repr__(self):
        return (
            f'PeriodicIndex({self.index!r}, period={self.period!r}, '
            f'origin={self.origin!r})'
        )
