from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

import xarray as xr
from xarray.indexes import PandasIndex, PeriodicIndex


def make_dataset(values=np.arange(0, 360, 10), **options):
    ds = xr.Dataset(
        {
            'data': ('lon', np.arange(len(values))),
            'aux': ('lon', np.arange(len(values)) * 10),
        },
        coords={'lon': values},
    )
    return ds.set_xindex('lon', PeriodicIndex, period=360, **options)


def test_cross_prime_meridian_slice_matches_positions() -> None:
    ds = make_dataset()

    actual = ds.sel(lon=slice(350, 10))
    expected = ds.isel(lon=[35, 0, 1])
    xr.testing.assert_identical(actual, expected)
    np.testing.assert_array_equal(actual.lon.values, [350, 0, 10])


def test_regular_slice_matches_positions() -> None:
    ds = make_dataset()

    xr.testing.assert_identical(
        ds.sel(lon=slice(10, 30)), ds.isel(lon=[1, 2, 3])
    )
    xr.testing.assert_identical(
        ds.sel(lon=slice(30, 10, -1)), ds.isel(lon=[3, 2, 1])
    )


def test_reverse_cross_prime_meridian_slice_matches_positions() -> None:
    ds = make_dataset()

    actual = ds.sel(lon=slice(10, 350, -1))
    expected = ds.isel(lon=[1, 0, 35])
    xr.testing.assert_identical(actual, expected)
    np.testing.assert_array_equal(actual.lon.values, [10, 0, 350])


@pytest.mark.parametrize(
    ('query', 'expected_position'),
    [
        (360, 0),
        (-10, 35),
        (710, 35),
    ],
)
def test_scalar_labels_are_periodically_normalized(
    query: float, expected_position: int
) -> None:
    ds = make_dataset()
    xr.testing.assert_identical(ds.sel(lon=query), ds.isel(lon=expected_position))


def test_array_labels_are_periodically_normalized() -> None:
    ds = make_dataset()

    actual = ds.sel(lon=[360, -10, 350])
    expected = ds.isel(lon=[0, 35, 35])
    xr.testing.assert_identical(actual, expected)
    np.testing.assert_array_equal(actual.lon.values, [0, 350, 350])


def test_nearest_uses_circular_distance_and_deterministic_tie() -> None:
    ds = make_dataset(values=np.array([358, 2]))

    actual = ds.sel(lon=0, method='nearest')
    expected = ds.isel(lon=1)
    xr.testing.assert_identical(actual, expected)

    # 357 is one degree from 358 and five degrees from 2.
    xr.testing.assert_identical(
        ds.sel(lon=357, method='nearest'), ds.isel(lon=0)
    )

    # 0 is two degrees from both labels. The positive-period neighbor wins.
    np.testing.assert_array_equal(
        ds.sel(lon=[0, 356], method='nearest').lon.values, [2, 358]
    )


def test_nearest_tolerance_is_circular() -> None:
    ds = make_dataset(values=np.array([358, 2]))
    actual = ds.sel(lon=359, method='nearest', tolerance=1)
    xr.testing.assert_identical(actual, ds.isel(lon=0))

    with pytest.raises(KeyError, match='tolerance'):
        ds.sel(lon=359, method='nearest', tolerance=0.5)


def test_nearest_array_tolerance_and_methods() -> None:
    ds = make_dataset()

    np.testing.assert_array_equal(
        ds.sel(
            lon=np.array([352, 4]), method='nearest', tolerance=np.array([3, 6])
        ).lon.values,
        [350, 0],
    )
    np.testing.assert_array_equal(
        ds.sel(lon=np.array([359, 1]), method='pad').lon.values, [350, 0]
    )
    np.testing.assert_array_equal(
        ds.sel(lon=np.array([359, 1]), method='bfill').lon.values, [0, 10]
    )

    labels = xr.DataArray([355, 5], dims='query')
    result = ds.sel(lon=labels, method='nearest')
    assert tuple(result.dims) == ('query',)
    np.testing.assert_array_equal(result.lon.values, [0, 10])


def test_descending_and_nonuniform_coordinates() -> None:
    values = np.array([350, 20, 355, 5])
    ds = make_dataset(values=values)

    actual = ds.sel(lon=slice(355, 5))
    expected = ds.isel(lon=[2, 3])
    xr.testing.assert_identical(actual, expected)
    np.testing.assert_array_equal(actual.lon.values, [355, 5])

    actual = ds.sel(lon=slice(5, 355, -1))
    expected = ds.isel(lon=[3, 2])
    xr.testing.assert_identical(actual, expected)
    np.testing.assert_array_equal(actual.lon.values, [5, 355])

    strided = xr.Dataset(coords={'lon': np.array([350, 20, 355, 5, 10])})
    strided = strided.set_xindex('lon', PeriodicIndex, period=360)
    np.testing.assert_array_equal(
        strided.sel(lon=slice(355, 5, 2)).lon.values, [355]
    )
    np.testing.assert_array_equal(
        strided.sel(lon=slice(5, 355, -2)).lon.values, [5]
    )
    np.testing.assert_array_equal(
        strided.sel(lon=slice(None, None, -1)).lon.values,
        [355, 350, 20, 10, 5],
    )

    xr.testing.assert_identical(ds.sel(lon=365), ds.isel(lon=3))


def test_origin_is_honored() -> None:
    values = np.array([185, 355])
    ds = make_dataset(values=values, origin=180)

    actual = ds.sel(lon=slice(355, 185))
    expected = ds.isel(lon=[1, 0])
    xr.testing.assert_identical(actual, expected)
    np.testing.assert_array_equal(actual.lon.values, [355, 185])

    xr.testing.assert_identical(
        ds.sel(lon=545), ds.isel(lon=0)
    )


def test_equivalent_duplicate_labels_must_be_explicit() -> None:
    ds = xr.Dataset(coords={'lon': [0, 360]})

    with pytest.raises(ValueError, match='unique on the declared period'):
        ds.set_xindex('lon', PeriodicIndex, period=360)

    ds_other = xr.Dataset(coords={'lon': [-180, 180]})
    with pytest.raises(ValueError, match='unique on the declared period'):
        ds_other.set_xindex('lon', PeriodicIndex, period=360, origin=-180)


def test_undeclared_axis_remains_linear() -> None:
    ds = xr.Dataset(
        {'data': ('lon', np.arange(36))}, coords={'lon': np.arange(0, 360, 10)}
    )
    assert isinstance(ds.xindexes['lon'], PandasIndex)

    result = ds.sel(lon=slice(350, 10))
    assert result.sizes['lon'] == 0
    np.testing.assert_array_equal(result.lon.values, [])


def test_periodic_index_follows_rename_copy_and_pickle() -> None:
    ds = make_dataset()

    sliced = ds.sel(lon=slice(350, 10))
    assert sliced.xindexes['lon'].period == 360
    assert sliced.xindexes['lon'].origin == 0

    renamed = ds.rename({'lon': 'longitude'})
    assert isinstance(renamed.xindexes['longitude'], PeriodicIndex)
    assert renamed.xindexes['longitude'].period == 360

    copied = copy.deepcopy(ds)
    assert copied.xindexes['lon'].equals(ds.xindexes['lon'])

    restored = pickle.loads(pickle.dumps(ds))
    xr.testing.assert_identical(restored, ds)
    assert isinstance(restored.xindexes['lon'], PeriodicIndex)


def test_concat_preserves_periodic_configuration() -> None:
    left = make_dataset(np.array([0, 10]))
    right = make_dataset(np.array([350]))

    result = xr.concat([left, right], dim='lon')
    assert isinstance(result.xindexes['lon'], PeriodicIndex)
    assert result.xindexes['lon'].period == 360
    np.testing.assert_array_equal(result.lon.values, [0, 10, 350])


def test_periodic_index_requires_period_option() -> None:
    ds = xr.Dataset(coords={'lon': np.arange(0, 360, 10)})
    with pytest.raises(ValueError, match='requires a period'):
        ds.set_xindex('lon', PeriodicIndex)


def test_dask_selection_stays_lazy() -> None:
    dask = pytest.importorskip('dask')

    ds = make_dataset().chunk({'lon': (18, 18)})
    sliced = ds.sel(lon=slice(350, 10))
    fancy = ds.sel(lon=[350, 0, 10])

    assert dask.is_dask_collection(sliced.data)
    assert dask.is_dask_collection(fancy.data)
    assert sliced.chunks['lon'] == (3,)
    assert fancy.chunks['lon'] == (3,)


def test_roll_preserves_periodic_configuration() -> None:
    ds = make_dataset()
    rolled = ds.roll(lon=1, roll_coords=True)

    assert isinstance(rolled.xindexes['lon'], PeriodicIndex)
    assert rolled.xindexes['lon'].period == 360
