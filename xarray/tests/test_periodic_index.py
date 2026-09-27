import copy
import pickle

import numpy as np
import pytest

import xarray as xr
from xarray.indexes import PandasIndex, PeriodicIndex
from xarray.tests import assert_equal, assert_identical, requires_dask


def create_dataset(lon=None, period=360.0, origin=0.0) -> xr.Dataset:
    if lon is None:
        lon = np.arange(360.0)
    lon = np.asarray(lon, dtype=np.float64)
    data = np.arange(lon.size, dtype=np.float64)
    ds = xr.Dataset(
        {
            "temp": ("lon", data * 10),
            "pres": ("lon", data * 100),
        },
        coords={"lon": lon, "lon_aux": ("lon", data + 1000)},
    )
    return ds.set_xindex("lon", PeriodicIndex, period=period, origin=origin)


def test_create() -> None:
    ds = create_dataset()
    index = ds.xindexes["lon"]
    assert isinstance(index, PeriodicIndex)
    assert index.period == 360.0
    assert index.origin == 0.0
    assert index.dim == "lon"
    assert index.coord_name == "lon"


def test_create_requires_period() -> None:
    ds = xr.Dataset(coords={"lon": np.arange(360.0)})
    with pytest.raises(TypeError, match="requires a 'period' option"):
        ds.set_xindex("lon", PeriodicIndex)


@pytest.mark.parametrize("period", [0.0, -360.0, np.inf, np.nan])
def test_create_invalid_period(period) -> None:
    ds = xr.Dataset(coords={"lon": np.arange(360.0)})
    with pytest.raises(ValueError, match="period must be a positive"):
        ds.set_xindex("lon", PeriodicIndex, period=period)


def test_create_non_numeric_coordinate() -> None:
    ds = xr.Dataset(coords={"x": list("abcd")})
    with pytest.raises(ValueError, match="only supports numeric"):
        ds.set_xindex("x", PeriodicIndex, period=360.0)


def test_create_ambiguous_equivalent_labels() -> None:
    # both 0 and 360 are present: they are equivalent modulo the period and
    # the ambiguity must be resolved explicitly by the user
    ds = xr.Dataset(coords={"lon": [0.0, 180.0, 360.0]})
    with pytest.raises(ValueError, match="ambiguous.*equivalent"):
        ds.set_xindex("lon", PeriodicIndex, period=360.0)


def test_sel_slice_cross_seam() -> None:
    ds = create_dataset()
    actual = ds.sel(lon=slice(350, 10))

    expected_positions = np.r_[np.arange(350, 360), np.arange(0, 11)]
    expected = ds.isel(lon=expected_positions)

    assert_identical(actual, expected, check_default_indexes=False)
    # the periodic index is preserved after selection
    index = actual.xindexes["lon"]
    assert isinstance(index, PeriodicIndex)
    assert index.period == 360.0


def test_sel_slice_plain_interval() -> None:
    ds = create_dataset()
    actual = ds.sel(lon=slice(10, 20))
    expected = ds.isel(lon=np.arange(10, 21))
    assert_identical(actual, expected, check_default_indexes=False)


def test_sel_slice_full_period() -> None:
    ds = create_dataset()
    actual = ds.sel(lon=slice(0, 360))
    assert_identical(actual, ds, check_default_indexes=False)


def test_sel_slice_point() -> None:
    ds = create_dataset()
    actual = ds.sel(lon=slice(350, 350))
    expected = ds.isel(lon=[350])
    assert_identical(actual, expected, check_default_indexes=False)


def test_sel_slice_step() -> None:
    ds = create_dataset()
    actual = ds.sel(lon=slice(350, 10, 2))
    expected_positions = np.r_[np.arange(350, 360), np.arange(0, 11)][::2]
    expected = ds.isel(lon=expected_positions)
    assert_identical(actual, expected, check_default_indexes=False)


def test_sel_slice_open_ended() -> None:
    ds = create_dataset()
    assert_identical(ds.sel(lon=slice(None, 10)), ds.isel(lon=np.arange(0, 11)), check_default_indexes=False)
    assert_identical(ds.sel(lon=slice(350, None)), ds.isel(lon=np.arange(350, 360)), check_default_indexes=False)
    assert_identical(ds.sel(lon=slice(None)), ds, check_default_indexes=False)


def test_sel_slice_descending() -> None:
    lon = np.arange(359.0, -1, -1)
    ds = create_dataset(lon=lon)

    # plain interval follows the (descending) coordinate direction
    actual = ds.sel(lon=slice(350, 10))
    expected = ds.isel(lon=np.arange(9, 350))
    assert_identical(actual, expected, check_default_indexes=False)

    # cross-seam slice returns labels in the requested direction
    actual = ds.sel(lon=slice(10, 350))
    expected_positions = np.r_[np.arange(349, 360), np.arange(0, 10)]
    expected = ds.isel(lon=expected_positions)
    assert_identical(actual, expected, check_default_indexes=False)


def test_sel_slice_rolled_coordinate() -> None:
    # non-monotonic (rolled) coordinate: selection still wraps the seam
    lon = np.r_[np.arange(180.0, 360), np.arange(0.0, 180)]
    ds = create_dataset(lon=lon)

    actual = ds.sel(lon=slice(350, 10))
    expected_labels = np.r_[np.arange(350.0, 360), np.arange(0.0, 11)]
    assert np.array_equal(actual.lon.values, expected_labels)
    # data and auxiliary coordinates stay in sync with the labels
    expected_positions = [np.where(lon == label)[0][0] for label in expected_labels]
    assert np.array_equal(actual["temp"].values, ds["temp"].values[expected_positions])
    assert np.array_equal(
        actual["lon_aux"].values, ds["lon_aux"].values[expected_positions]
    )


def test_sel_slice_non_uniform() -> None:
    lon = np.array([0.0, 30.0, 90.0, 150.0, 210.0, 300.0])
    ds = create_dataset(lon=lon)

    actual = ds.sel(lon=slice(300, 30))
    assert np.array_equal(actual.lon.values, [300.0, 0.0, 30.0])
    assert np.array_equal(actual["temp"].values, [50.0, 0.0, 10.0])


def test_sel_slice_origin() -> None:
    lon = np.arange(-180.0, 180)
    ds = create_dataset(lon=lon, origin=-180.0)

    actual = ds.sel(lon=slice(170, -170))
    expected_labels = np.r_[np.arange(170.0, 180), np.arange(-180.0, -169)]
    assert np.array_equal(actual.lon.values, expected_labels)


def test_sel_scalar_normalized() -> None:
    ds = create_dataset()
    assert ds.sel(lon=360.0)["temp"].item() == 0.0
    assert ds.sel(lon=-10.0)["temp"].item() == 3500.0
    assert ds.sel(lon=720.0 + 90.0)["temp"].item() == 900.0


def test_sel_scalar_not_found() -> None:
    ds = create_dataset()
    with pytest.raises(KeyError, match="not all values found"):
        ds.sel(lon=0.5)


def test_sel_array_normalized() -> None:
    ds = create_dataset()
    actual = ds.sel(lon=[350.0, 360.0, -10.0])
    assert np.array_equal(actual.lon.values, [350.0, 0.0, 350.0])
    assert np.array_equal(actual["temp"].values, [3500.0, 0.0, 3500.0])


def test_sel_array_not_found() -> None:
    ds = create_dataset()
    with pytest.raises(KeyError, match="not all values found"):
        ds.sel(lon=[10.0, 10.5])


def test_sel_nearest_circular_distance() -> None:
    ds = create_dataset()
    # 359.6 is closer to 0 (distance 0.4 across the seam) than to 359 (0.6)
    assert ds.sel(lon=359.6, method="nearest").lon.item() == 0.0
    assert ds.sel(lon=-0.4, method="nearest").lon.item() == 0.0
    assert ds.sel(lon=100.4, method="nearest").lon.item() == 100.0


def test_sel_nearest_tie_breaking() -> None:
    # ties are deterministically resolved in favor of the smaller
    # normalized label
    ds = create_dataset(lon=[0.0, 10.0])
    assert ds.sel(lon=5.0, method="nearest").lon.item() == 0.0

    ds = create_dataset(lon=[0.0, 180.0])
    assert ds.sel(lon=90.0, method="nearest").lon.item() == 0.0


def test_sel_nearest_array_and_tolerance() -> None:
    ds = create_dataset()
    actual = ds.sel(lon=[359.6, 100.4], method="nearest")
    assert np.array_equal(actual.lon.values, [0.0, 100.0])

    with pytest.raises(KeyError, match="not all values found"):
        ds.sel(lon=[100.4], method="nearest", tolerance=0.1)

    actual = ds.sel(lon=[100.4], method="nearest", tolerance=0.5)
    assert np.array_equal(actual.lon.values, [100.0])


def test_sel_invalid_method() -> None:
    ds = create_dataset()
    with pytest.raises(ValueError, match="method='nearest'"):
        ds.sel(lon=10.0, method="pad")


def test_isel_preserves_periodic_config() -> None:
    ds = create_dataset()
    actual = ds.isel(lon=slice(0, 10))
    index = actual.xindexes["lon"]
    assert isinstance(index, PeriodicIndex)
    assert index.period == 360.0
    assert index.origin == 0.0

    # label-based selection still wraps on the sliced dataset
    assert_identical(ds.isel(lon=[359, 0, 1]), create_dataset().sel(lon=[359, 0, 1]), check_default_indexes=False)


def test_rename_preserves_periodic_config() -> None:
    ds = create_dataset()
    actual = ds.rename(lon="longitude")
    index = actual.xindexes["longitude"]
    assert isinstance(index, PeriodicIndex)
    assert index.period == 360.0
    assert index.dim == "longitude"

    selected = actual.sel(longitude=slice(350, 10))
    assert len(selected.longitude) == 21


def test_copy_preserves_periodic_config() -> None:
    ds = create_dataset()
    for copied in (ds.copy(), copy.deepcopy(ds), ds.copy(deep=False)):
        index = copied.xindexes["lon"]
        assert isinstance(index, PeriodicIndex)
        assert index.period == 360.0
        assert index.origin == 0.0


def test_pickle_roundtrip() -> None:
    ds = create_dataset()
    restored = pickle.loads(pickle.dumps(ds))
    index = restored.xindexes["lon"]
    assert isinstance(index, PeriodicIndex)
    assert index.period == 360.0
    assert index.origin == 0.0
    assert_identical(restored.sel(lon=slice(350, 10)), ds.sel(lon=slice(350, 10)), check_default_indexes=False)


def test_roll_preserves_periodic_config() -> None:
    ds = create_dataset()
    rolled = ds.roll(lon=180)
    index = rolled.xindexes["lon"]
    assert isinstance(index, PeriodicIndex)
    assert index.period == 360.0
    # wrap selection works the same on the rolled dataset
    actual = rolled.sel(lon=slice(350, 10))
    expected_labels = np.r_[np.arange(350.0, 360), np.arange(0.0, 11)]
    assert np.array_equal(actual.lon.values, expected_labels)


def test_equals() -> None:
    ds = create_dataset()
    index = ds.xindexes["lon"]
    assert index.equals(create_dataset().xindexes["lon"])
    assert not index.equals(create_dataset(period=720.0).xindexes["lon"])
    assert not index.equals(create_dataset(origin=1.0).xindexes["lon"])
    assert not index.equals(PandasIndex(np.arange(360.0), "lon"))


def test_alignment_identical_indexes() -> None:
    ds = create_dataset()
    other = create_dataset()
    result = ds + other
    assert isinstance(result.xindexes["lon"], PeriodicIndex)


def test_non_periodic_dimension_unchanged() -> None:
    # dimensions without a declared period keep their linear interpretation
    ds = create_dataset().expand_dims(lat=[0.0, 1.0, 2.0])
    assert not isinstance(ds.xindexes["lat"], PeriodicIndex)
    actual = ds.sel(lat=slice(0, 1))
    assert np.array_equal(actual.lat.values, [0.0, 1.0])


def test_to_pandas_index() -> None:
    ds = create_dataset()
    pd_index = ds.xindexes["lon"].to_pandas_index()
    assert np.array_equal(pd_index.values, np.arange(360.0))


def test_repr() -> None:
    ds = create_dataset()
    index = ds.xindexes["lon"]
    assert "period=360" in repr(index)
    assert "origin=0" in repr(index)
    assert "PeriodicIndex" in index._repr_inline_(80)


@requires_dask
def test_sel_cross_seam_lazy_dask() -> None:
    ds = create_dataset().chunk({"lon": 90})
    actual = ds.sel(lon=slice(350, 10))

    # selection stays lazy
    assert isinstance(actual["temp"].data, type(ds["temp"].data))
    assert actual["temp"].chunks is not None

    expected_positions = np.r_[np.arange(350, 360), np.arange(0, 11)]
    expected = ds.isel(lon=expected_positions)
    assert_equal(actual.compute(), expected.compute(), check_default_indexes=False)
