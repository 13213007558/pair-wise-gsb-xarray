from __future__ import annotations

import numpy as np
import pytest

from xarray import Dataset
from xarray.coding.ragged import (
    RAGGED_ENCODING_KEY,
    ragged_specs,
    select_ragged_instances,
)
from xarray.conventions import cf_encoder, decode_cf
from xarray.testing import assert_identical
from xarray.tests import requires_dask


def contiguous_dataset() -> Dataset:
    # CF contiguous ragged layout: 3 instances with 3, 0 and 2 observations
    return Dataset(
        {
            "row_size": (
                "trajectory",
                np.array([3, 0, 2], dtype="int32"),
                {"sample_dimension": "obs", "long_name": "number of observations"},
            ),
            "traj": ("trajectory", [10, 11, 12]),
            "temp": ("obs", [1.0, -999.0, 3.0, 4.0, 5.0], {"_FillValue": -999.0}),
            "lat": ("obs", [40.0, 41.0, 42.0, 50.0, 51.0]),
        }
    )


def indexed_dataset() -> Dataset:
    # CF indexed ragged layout: interleaved instances, instance 1 is empty
    return Dataset(
        {
            "traj_index": (
                "obs",
                [0, 2, 0, 2],
                {"instance_dimension": "trajectory"},
            ),
            "traj": ("trajectory", [10, 11, 12]),
            "temp": ("obs", [1.0, -999.0, 3.0, 4.0], {"_FillValue": -999.0}),
            "lat": ("obs", [40.0, 50.0, 41.0, 51.0]),
        }
    )


def roundtrip(decoded: Dataset, layout: str | None = None) -> Dataset:
    variables, attributes = cf_encoder(
        decoded._variables, dict(decoded.attrs), ragged_layout=layout
    )
    return decode_cf(Dataset(variables, attrs=attributes))


def test_decode_contiguous_ragged() -> None:
    decoded = decode_cf(contiguous_dataset())
    assert dict(decoded.sizes) == {"trajectory": 3, "obs": 5}
    # unified representation: instance-id coordinate on the obs dimension
    assert "trajectory_id" in decoded.coords
    np.testing.assert_array_equal(decoded["trajectory_id"], [0, 0, 0, 2, 2])
    # the count variable is an encoding artifact and removed on decode
    assert "row_size" not in decoded.variables
    # instance-level and observation-level variables are preserved
    np.testing.assert_array_equal(decoded["traj"], [10, 11, 12])
    np.testing.assert_array_equal(decoded["lat"], [40.0, 41.0, 42.0, 50.0, 51.0])
    # missing values are still decoded
    np.testing.assert_array_equal(
        decoded["temp"], [1.0, np.nan, 3.0, 4.0, 5.0]
    )


def test_decode_indexed_ragged() -> None:
    decoded = decode_cf(indexed_dataset())
    assert dict(decoded.sizes) == {"trajectory": 3, "obs": 4}
    # the index variable becomes the instance-id coordinate
    assert "traj_index" in decoded.coords
    assert "instance_dimension" not in decoded["traj_index"].attrs
    np.testing.assert_array_equal(decoded["traj_index"], [0, 2, 0, 2])
    np.testing.assert_array_equal(decoded["traj"], [10, 11, 12])
    np.testing.assert_array_equal(decoded["temp"], [1.0, np.nan, 3.0, 4.0])


def test_decode_ragged_opt_out() -> None:
    decoded = decode_cf(contiguous_dataset(), decode_ragged=False)
    assert "row_size" in decoded.variables
    assert "trajectory_id" not in decoded.variables


@pytest.mark.parametrize("layout", ["contiguous", "indexed"])
def test_ragged_roundtrip_same_layout(layout: str) -> None:
    original = contiguous_dataset() if layout == "contiguous" else indexed_dataset()
    decoded = decode_cf(original)
    assert_identical(roundtrip(decoded), decoded)


def test_ragged_roundtrip_indexed_to_contiguous() -> None:
    decoded = decode_cf(indexed_dataset())
    redecoded = roundtrip(decoded, layout="contiguous")
    # the contiguous layout groups observations by instance, preserving the
    # observation order within each instance
    order = np.argsort(decoded["traj_index"].values, kind="stable")
    expected = decoded.isel(obs=order).rename(traj_index="trajectory_id")
    assert_identical(redecoded, expected)


def test_ragged_roundtrip_contiguous_to_indexed() -> None:
    decoded = decode_cf(contiguous_dataset())
    redecoded = roundtrip(decoded, layout="indexed")
    assert_identical(redecoded, decoded)
    # re-encoding to contiguous again restores a count variable
    variables, _ = cf_encoder(
        redecoded._variables, dict(redecoded.attrs), ragged_layout="contiguous"
    )
    np.testing.assert_array_equal(variables["trajectory_count"].data, [3, 0, 2])
    assert variables["trajectory_count"].attrs["sample_dimension"] == "obs"


def test_ragged_roundtrip_restores_count_variable() -> None:
    decoded = decode_cf(contiguous_dataset())
    variables, _ = cf_encoder(decoded._variables, dict(decoded.attrs))
    np.testing.assert_array_equal(variables["row_size"].data, [3, 0, 2])
    assert variables["row_size"].attrs["sample_dimension"] == "obs"
    assert variables["row_size"].attrs["long_name"] == "number of observations"
    assert variables["row_size"].dtype == np.dtype("int32")


def test_ragged_encode_default_keeps_original_layout() -> None:
    decoded = decode_cf(indexed_dataset())
    variables, _ = cf_encoder(decoded._variables, dict(decoded.attrs))
    assert variables["traj_index"].attrs["instance_dimension"] == "trajectory"
    assert "row_size" not in variables

    decoded = decode_cf(contiguous_dataset())
    variables, _ = cf_encoder(decoded._variables, dict(decoded.attrs))
    assert variables["row_size"].attrs["sample_dimension"] == "obs"
    assert "trajectory_id" not in variables


def test_ragged_encode_invalid_layout() -> None:
    decoded = decode_cf(contiguous_dataset())
    with pytest.raises(ValueError, match="Invalid ragged layout"):
        cf_encoder(decoded._variables, dict(decoded.attrs), ragged_layout="dense")


def test_select_ragged_instances_contiguous() -> None:
    decoded = decode_cf(contiguous_dataset())
    selected = select_ragged_instances(decoded, [2, 0])
    assert dict(selected.sizes) == {"trajectory": 2, "obs": 5}
    np.testing.assert_array_equal(selected["trajectory_id"], [1, 1, 1, 0, 0])
    np.testing.assert_array_equal(selected["traj"], [12, 10])
    np.testing.assert_array_equal(
        selected["lat"], [40.0, 41.0, 42.0, 50.0, 51.0]
    )


def test_select_ragged_instances_indexed() -> None:
    decoded = decode_cf(indexed_dataset())
    # selecting only the empty instance yields no observations
    selected = select_ragged_instances(decoded, [1])
    assert dict(selected.sizes) == {"trajectory": 1, "obs": 0}
    np.testing.assert_array_equal(selected["traj"], [11])
    # interleaved observations keep their relative order per instance
    selected = select_ragged_instances(decoded, [2, 0])
    np.testing.assert_array_equal(selected["traj_index"], [1, 0, 1, 0])
    np.testing.assert_array_equal(selected["lat"], [40.0, 50.0, 41.0, 51.0])


def test_select_ragged_instances_roundtrip() -> None:
    decoded = decode_cf(contiguous_dataset())
    selected = select_ragged_instances(decoded, [2, 0])
    redecoded = roundtrip(selected)
    assert dict(redecoded.sizes) == {"trajectory": 2, "obs": 5}
    np.testing.assert_array_equal(
        redecoded["temp"], [4.0, 5.0, 1.0, np.nan, 3.0]
    )
    np.testing.assert_array_equal(redecoded["traj"], [12, 10])


def test_select_ragged_instances_errors() -> None:
    decoded = decode_cf(contiguous_dataset())
    with pytest.raises(ValueError, match="out of bounds"):
        select_ragged_instances(decoded, [3])
    with pytest.raises(ValueError, match="unique"):
        select_ragged_instances(decoded, [0, 0])
    with pytest.raises(ValueError, match="does not contain a decoded ragged"):
        select_ragged_instances(Dataset({"a": ("x", [1, 2])}), [0])


def test_decode_ragged_count_sum_mismatch() -> None:
    ds = Dataset(
        {
            "row_size": ("trajectory", [2, 2], {"sample_dimension": "obs"}),
            "temp": ("obs", [1.0, 2.0, 3.0]),
        }
    )
    with pytest.raises(ValueError, match="'row_size'.*sums to 4"):
        decode_cf(ds)


def test_decode_ragged_negative_count() -> None:
    ds = Dataset(
        {
            "row_size": ("trajectory", [3, -1], {"sample_dimension": "obs"}),
            "temp": ("obs", [1.0, 2.0]),
        }
    )
    with pytest.raises(ValueError, match="'row_size'.*negative"):
        decode_cf(ds)


def test_decode_ragged_non_integer_count() -> None:
    ds = Dataset(
        {
            "row_size": ("trajectory", [1.5, 0.5], {"sample_dimension": "obs"}),
            "temp": ("obs", [1.0, 2.0]),
        }
    )
    with pytest.raises(ValueError, match="'row_size'.*integer"):
        decode_cf(ds)


def test_decode_ragged_index_out_of_bounds() -> None:
    ds = Dataset(
        {
            "idx": ("obs", [0, 5], {"instance_dimension": "trajectory"}),
            "traj": ("trajectory", [1, 2]),
            "temp": ("obs", [1.0, 2.0]),
        }
    )
    with pytest.raises(ValueError, match="'idx'.*instance index 5"):
        decode_cf(ds)


def test_decode_ragged_negative_index() -> None:
    ds = Dataset(
        {
            "idx": ("obs", [0, -1], {"instance_dimension": "trajectory"}),
            "temp": ("obs", [1.0, 2.0]),
        }
    )
    with pytest.raises(ValueError, match="'idx'.*negative"):
        decode_cf(ds)


def test_decode_ragged_conflicting_attributes_on_same_variable() -> None:
    ds = Dataset(
        {
            "x": (
                "obs",
                [0, 1],
                {"sample_dimension": "obs", "instance_dimension": "trajectory"},
            ),
            "temp": ("obs", [1.0, 2.0]),
        }
    )
    with pytest.raises(ValueError, match="'x'.*both"):
        decode_cf(ds)


def test_decode_ragged_conflicting_layouts() -> None:
    ds = Dataset(
        {
            "row_size": ("trajectory", [2], {"sample_dimension": "obs"}),
            "idx": ("obs", [0, 0], {"instance_dimension": "trajectory"}),
            "temp": ("obs", [1.0, 2.0]),
        }
    )
    with pytest.raises(ValueError, match="'row_size'.*'idx'.*conflicting"):
        decode_cf(ds)


def test_decode_ragged_unknown_sample_dimension() -> None:
    ds = Dataset(
        {
            "row_size": ("trajectory", [2], {"sample_dimension": "nosuchdim"}),
            "temp": ("obs", [1.0, 2.0]),
        }
    )
    with pytest.raises(ValueError, match="'row_size'.*'nosuchdim'"):
        decode_cf(ds)


def test_decode_ragged_empty_instances_exist() -> None:
    decoded = decode_cf(contiguous_dataset())
    # instance 1 has no observations but still exists
    assert decoded.sizes["trajectory"] == 3
    selected = select_ragged_instances(decoded, [1])
    assert selected.sizes["trajectory"] == 1
    assert selected.sizes["obs"] == 0
    # and it survives re-encoding in both layouts
    for layout in ("contiguous", "indexed"):
        redecoded = roundtrip(decoded, layout=layout)
        spec = next(iter(ragged_specs(redecoded).values()))
        assert spec["n_instances"] == 3


@requires_dask
def test_decode_ragged_dask_stays_lazy() -> None:
    import dask.array

    original = contiguous_dataset().chunk()
    decoded = decode_cf(original)
    assert isinstance(decoded["temp"].data, dask.array.Array)
    assert isinstance(decoded["lat"].data, dask.array.Array)
    assert_identical(decoded.compute(), decode_cf(contiguous_dataset()))


@requires_dask
def test_encode_ragged_dask() -> None:
    decoded = decode_cf(contiguous_dataset().chunk())
    redecoded = roundtrip(decoded, layout="indexed")
    expected = decode_cf(contiguous_dataset())
    assert_identical(redecoded.compute(), expected)


def test_ragged_specs_of_plain_dataset() -> None:
    ds = Dataset({"a": ("x", [1, 2]), "b": ("y", [3.0, 4.0])})
    assert ragged_specs(ds) == {}
    # plain datasets are unaffected by ragged decoding
    assert_identical(decode_cf(ds), ds)
    assert RAGGED_ENCODING_KEY not in ds["a"].encoding
