from __future__ import annotations

import numpy as np
import pytest

from xarray import Dataset, decode_cf, encode_cf
from xarray.coding.ragged import RAGGED_ENCODING_KEY
from xarray.testing import assert_identical
from xarray.tests import dask_array_type, requires_dask


def contiguous_ds() -> Dataset:
    """Contiguous ragged layout: 3 trajectories, counts [2, 0, 3].

    The second trajectory is an empty instance.
    """
    return Dataset(
        {
            "row_size": (
                "trajectory",
                np.array([2, 0, 3]),
                {"sample_dimension": "obs"},
            ),
            "traj_id": ("trajectory", np.array([10, 11, 12])),
            "temp": ("obs", np.array([1.0, 2.0, 3.0, 4.0, 5.0])),
        },
        coords={"lat": ("obs", np.array([10.0, 20.0, 30.0, 40.0, 50.0]))},
    )


def indexed_ds() -> Dataset:
    """Indexed ragged layout with interleaved observations.

    Instance indices are [0, 2, 0, 2, 2]; the second trajectory is empty.
    """
    return Dataset(
        {
            "trajectory_index": (
                "obs",
                np.array([0, 2, 0, 2, 2]),
                {"instance_dimension": "trajectory"},
            ),
            "traj_id": ("trajectory", np.array([10, 11, 12])),
            "temp": ("obs", np.array([1.0, 3.0, 2.0, 4.0, 5.0])),
        },
        coords={"lat": ("obs", np.array([10.0, 30.0, 20.0, 40.0, 50.0]))},
    )


class TestDecodeRagged:
    def test_decode_contiguous(self) -> None:
        decoded = decode_cf(contiguous_ds())
        assert decoded.sizes["trajectory"] == 3
        assert decoded.sizes["obs"] == 5
        # instance-level and observation-level variables keep their dimensions
        assert decoded["traj_id"].dims == ("trajectory",)
        assert decoded["temp"].dims == ("obs",)
        assert decoded["row_size"].dims == ("trajectory",)
        # the synthesized per-observation instance index
        assert "trajectory_index" in decoded.coords
        np.testing.assert_array_equal(
            decoded["trajectory_index"], np.array([0, 0, 2, 2, 2])
        )
        # auxiliary coordinates are preserved
        assert "lat" in decoded.coords
        # ragged metadata is recorded for re-encoding
        info = decoded.encoding[RAGGED_ENCODING_KEY]
        assert info["layout"] == "contiguous"
        assert info["instance_dim"] == "trajectory"
        assert info["obs_dim"] == "obs"

    def test_decode_indexed(self) -> None:
        decoded = decode_cf(indexed_ds())
        assert decoded.sizes["trajectory"] == 3
        # interleaved observation order is preserved
        np.testing.assert_array_equal(
            decoded["temp"], np.array([1.0, 3.0, 2.0, 4.0, 5.0])
        )
        # the index variable becomes a coordinate on the observation dimension
        assert "trajectory_index" in decoded.coords
        np.testing.assert_array_equal(
            decoded["trajectory_index"], np.array([0, 2, 0, 2, 2])
        )
        info = decoded.encoding[RAGGED_ENCODING_KEY]
        assert info["layout"] == "indexed"

    def test_decode_indexed_without_instance_variables(self) -> None:
        raw = indexed_ds().drop_vars("traj_id")
        decoded = decode_cf(raw)
        # the instance dimension is created from the index values
        assert decoded.sizes["trajectory"] == 3
        np.testing.assert_array_equal(decoded["trajectory"], np.array([0, 1, 2]))

    def test_empty_instance_exists(self) -> None:
        for raw in (contiguous_ds(), indexed_ds()):
            decoded = decode_cf(raw)
            # trajectory 1 has no observations but still exists
            assert decoded.sizes["trajectory"] == 3
            assert decoded["traj_id"].isel(trajectory=1).item() == 11

    def test_uniform_instance_selection(self) -> None:
        # both layouts decode to the same per-instance selectable form
        for raw, expected_temp, expected_lat in [
            (contiguous_ds(), [3.0, 4.0, 5.0], [30.0, 40.0, 50.0]),
            (indexed_ds(), [3.0, 4.0, 5.0], [30.0, 40.0, 50.0]),
        ]:
            decoded = decode_cf(raw)
            instance = decoded.isel(obs=decoded["trajectory_index"] == 2)
            np.testing.assert_array_equal(instance["temp"], expected_temp)
            np.testing.assert_array_equal(instance["lat"], expected_lat)
            groups = decoded.groupby("trajectory_index")
            np.testing.assert_array_equal(
                groups[2]["temp"],
                expected_temp,  # type: ignore[index]
            )

    def test_decode_ragged_disabled(self) -> None:
        decoded = decode_cf(contiguous_ds(), decode_ragged=False)
        assert "trajectory_index" not in decoded.variables
        assert decoded["row_size"].attrs["sample_dimension"] == "obs"
        assert RAGGED_ENCODING_KEY not in decoded.encoding

    def test_plain_cf_dataset_unchanged(self) -> None:
        raw = Dataset(
            {"temp": ("x", np.array([1.0, 2.0, 3.0]))},
            coords={"x": np.arange(3)},
        )
        assert_identical(decode_cf(raw), decode_cf(raw, decode_ragged=False))
        assert RAGGED_ENCODING_KEY not in decode_cf(raw).encoding


class TestRaggedErrors:
    def test_count_sum_mismatch(self) -> None:
        raw = contiguous_ds()
        raw["row_size"] = (
            "trajectory",
            np.array([2, 0, 2]),
            {"sample_dimension": "obs"},
        )
        with pytest.raises(ValueError, match="'row_size'.*sums to 4.*'obs'"):
            decode_cf(raw)

    def test_count_negative(self) -> None:
        raw = contiguous_ds()
        raw["row_size"] = (
            "trajectory",
            np.array([2, -1, 4]),
            {"sample_dimension": "obs"},
        )
        with pytest.raises(ValueError, match="'row_size'.*negative"):
            decode_cf(raw)

    def test_count_non_integer(self) -> None:
        raw = contiguous_ds()
        raw["row_size"] = (
            "trajectory",
            np.array([2.0, 0.0, 3.0]),
            {"sample_dimension": "obs"},
        )
        with pytest.raises(ValueError, match="'row_size'.*integer dtype"):
            decode_cf(raw)

    def test_index_out_of_bounds(self) -> None:
        raw = indexed_ds()
        raw["trajectory_index"] = (
            "obs",
            np.array([0, 2, 0, 2, 3]),
            {"instance_dimension": "trajectory"},
        )
        with pytest.raises(ValueError, match="'trajectory_index'.*out of bounds"):
            decode_cf(raw)

    def test_index_negative(self) -> None:
        raw = indexed_ds()
        raw["trajectory_index"] = (
            "obs",
            np.array([0, 2, 0, 2, -1]),
            {"instance_dimension": "trajectory"},
        )
        with pytest.raises(ValueError, match="'trajectory_index'.*negative"):
            decode_cf(raw)

    def test_conflicting_attributes_on_same_variable(self) -> None:
        raw = contiguous_ds()
        raw["row_size"].attrs["instance_dimension"] = "trajectory"
        with pytest.raises(ValueError, match="'row_size' declares both"):
            decode_cf(raw)

    def test_conflicting_layouts_same_obs_dimension(self) -> None:
        raw = contiguous_ds()
        raw["trajectory_index"] = (
            "obs",
            np.array([0, 0, 2, 2, 2]),
            {"instance_dimension": "trajectory"},
        )
        with pytest.raises(
            ValueError, match="'row_size' and 'trajectory_index'.*conflicting"
        ):
            decode_cf(raw)

    def test_duplicate_count_variables(self) -> None:
        raw = contiguous_ds()
        raw["row_size_2"] = (
            "trajectory",
            np.array([2, 0, 3]),
            {"sample_dimension": "obs"},
        )
        with pytest.raises(ValueError, match="'row_size' and 'row_size_2'"):
            decode_cf(raw)

    def test_sample_dimension_missing(self) -> None:
        raw = contiguous_ds()
        raw["row_size"].attrs["sample_dimension"] = "nonexistent"
        with pytest.raises(ValueError, match="'row_size'.*'nonexistent'"):
            decode_cf(raw)

    def test_encode_without_ragged_structure_raises(self) -> None:
        ds = Dataset({"temp": ("obs", np.array([1.0, 2.0]))})
        with pytest.raises(ValueError, match="no ragged structure"):
            encode_cf(ds, ragged_layout="contiguous")

    def test_invalid_layout_raises(self) -> None:
        decoded = decode_cf(contiguous_ds())
        with pytest.raises(ValueError, match="invalid ragged layout"):
            encode_cf(decoded, ragged_layout="dense")  # type: ignore[arg-type]


class TestEncodeRagged:
    def test_roundtrip_contiguous(self) -> None:
        decoded = decode_cf(contiguous_ds())
        encoded = encode_cf(decoded)
        assert encoded["row_size"].attrs["sample_dimension"] == "obs"
        np.testing.assert_array_equal(encoded["row_size"], np.array([2, 0, 3]))
        # the structural index coordinate is not written
        assert "trajectory_index" not in encoded.variables
        assert_identical(decode_cf(encoded), decoded)

    def test_roundtrip_indexed(self) -> None:
        decoded = decode_cf(indexed_ds())
        encoded = encode_cf(decoded)
        assert encoded["trajectory_index"].attrs["instance_dimension"] == "trajectory"
        np.testing.assert_array_equal(
            encoded["trajectory_index"], np.array([0, 2, 0, 2, 2])
        )
        # interleaved observation order is preserved
        np.testing.assert_array_equal(
            encoded["temp"], np.array([1.0, 3.0, 2.0, 4.0, 5.0])
        )
        assert_identical(decode_cf(encoded), decoded)

    def test_encode_contiguous_from_indexed(self) -> None:
        decoded = decode_cf(indexed_ds())
        encoded = encode_cf(decoded, ragged_layout="contiguous")
        np.testing.assert_array_equal(encoded["trajectory_count"], np.array([2, 0, 3]))
        assert encoded["trajectory_count"].attrs["sample_dimension"] == "obs"
        assert "trajectory_index" not in encoded.variables
        # observations are reordered contiguously, preserving per-instance order
        np.testing.assert_array_equal(
            encoded["temp"], np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        )
        np.testing.assert_array_equal(
            encoded["lat"], np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        )
        # the result decodes back to the same uniform representation
        redecoded = decode_cf(encoded)
        np.testing.assert_array_equal(
            redecoded["trajectory_index"], np.array([0, 0, 2, 2, 2])
        )
        instance = redecoded.isel(obs=redecoded["trajectory_index"] == 2)
        np.testing.assert_array_equal(instance["temp"], np.array([3.0, 4.0, 5.0]))

    def test_encode_indexed_from_contiguous(self) -> None:
        decoded = decode_cf(contiguous_ds())
        encoded = encode_cf(decoded, ragged_layout="indexed")
        assert "row_size" not in encoded.variables
        assert encoded["trajectory_index"].attrs["instance_dimension"] == "trajectory"
        np.testing.assert_array_equal(
            encoded["trajectory_index"], np.array([0, 0, 2, 2, 2])
        )
        np.testing.assert_array_equal(
            encoded["temp"], np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        )
        # re-decoding yields the same uniform representation as an indexed file
        redecoded = decode_cf(encoded)
        assert "row_size" not in redecoded.variables
        np.testing.assert_array_equal(
            redecoded["trajectory_index"], np.array([0, 0, 2, 2, 2])
        )
        np.testing.assert_array_equal(
            redecoded["temp"], np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        )

    def test_encode_detects_structure_from_attributes(self) -> None:
        # no decode first: structure is detected from variable attributes
        encoded = encode_cf(contiguous_ds())
        assert encoded["row_size"].attrs["sample_dimension"] == "obs"
        np.testing.assert_array_equal(encoded["row_size"], np.array([2, 0, 3]))

    def test_encode_plain_cf_dataset(self) -> None:
        ds = Dataset(
            {"temp": ("x", np.array([1.0, 2.0, 3.0]))},
            coords={"x": np.arange(3)},
        )
        encoded = encode_cf(ds)
        assert_identical(decode_cf(encoded), ds)

    def test_missing_values_roundtrip(self) -> None:
        raw = contiguous_ds()
        raw["temp"] = (
            "obs",
            np.array([1.0, -999.0, 3.0, -999.0, 5.0]),
            {"_FillValue": -999.0},
        )
        decoded = decode_cf(raw)
        np.testing.assert_array_equal(
            decoded["temp"].isnull(), np.array([False, True, False, True, False])
        )
        encoded = encode_cf(decoded)
        assert "_FillValue" in encoded["temp"].attrs
        assert_identical(decode_cf(encoded), decoded)


@requires_dask
class TestRaggedDask:
    def test_decode_stays_lazy(self) -> None:
        import dask.array as da

        raw = contiguous_ds()
        raw["temp"] = ("obs", da.from_array(np.arange(5.0), chunks=2))
        decoded = decode_cf(raw)
        assert isinstance(decoded["temp"].data, dask_array_type)
        np.testing.assert_array_equal(decoded["temp"].compute(), np.arange(5.0))

    def test_encode_contiguous_reorder_stays_lazy(self) -> None:
        import dask.array as da

        raw = indexed_ds()
        raw["temp"] = (
            "obs",
            da.from_array(np.array([1.0, 3.0, 2.0, 4.0, 5.0]), chunks=2),
        )
        decoded = decode_cf(raw)
        assert isinstance(decoded["temp"].data, dask_array_type)
        encoded = encode_cf(decoded, ragged_layout="contiguous")
        assert isinstance(encoded["temp"].data, dask_array_type)
        np.testing.assert_array_equal(
            encoded["temp"].compute(), np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        )
        np.testing.assert_array_equal(encoded["trajectory_count"], np.array([2, 0, 3]))
