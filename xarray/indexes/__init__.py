"""Xarray index objects for label-based selection and alignment of Dataset /
DataArray objects.

"""

from xarray.core.coordinate_transform import CoordinateTransform
from xarray.core.indexes import (
    CoordinateTransformIndex,
    Index,
    PandasIndex,
    PandasMultiIndex,
)
from xarray.indexes.nd_point_index import (
    NDPointIndex,
    NeighborQueryResult,
    TreeAdapter,
)
from xarray.indexes.range_index import RangeIndex

__all__ = [
    "CoordinateTransform",
    "CoordinateTransformIndex",
    "Index",
    "NDPointIndex",
    "NeighborQueryResult",
    "PandasIndex",
    "PandasMultiIndex",
    "RangeIndex",
    "TreeAdapter",
]
