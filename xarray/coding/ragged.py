"""Explicit support for CF ragged array layouts (contiguous and indexed).

See the CF conventions chapter on "Discrete Sampling Geometries":
https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#discrete-sampling-geometries

In a *contiguous ragged* layout a count variable (dimensioned by the instance
dimension) carries a ``sample_dimension`` attribute naming the observation
dimension; its values give the number of observations of each instance, stored
contiguously. In an *indexed ragged* layout an index variable (dimensioned by
the observation dimension) carries an ``instance_dimension`` attribute; its
values give the instance each observation belongs to and may be interleaved.

Both layouts decode to the same in-memory representation: observation-level
variables keep the observation dimension, instance-level variables keep the
instance dimension, and a coordinate on the observation dimension holds the
integer instance index of each observation, enabling uniform per-instance
selection (e.g. ``ds.isel(obs=ds["trajectory_index"] == i)`` or
``ds.groupby("trajectory_index")``) without padding to a dense matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from xarray.core.variable import Variable

if TYPE_CHECKING:
    from xarray.core.dataset import Dataset

RAGGED_ENCODING_KEY = "ragged"

_RaggedInfo = dict[str, Any]


def _free_name(ds: Dataset, base: str) -> str:
    """Return ``base`` or a suffixed variant that is not a variable in ``ds``."""
    name = base
    while name in ds.variables:
        name = f"{name}_"
    return name


def _ragged_declarations(ds: Dataset) -> tuple[dict[str, str], dict[str, str]]:
    """Find variables declaring CF ragged structure.

    Returns two dicts mapping the observation dimension name to the declaring
    variable name: one for count variables (``sample_dimension`` attribute,
    contiguous layout) and one for index variables (``instance_dimension``
    attribute, indexed layout).
    """
    count_vars: dict[str, str] = {}
    index_vars: dict[str, str] = {}
    for name, var in ds.variables.items():
        sample_dim = var.attrs.get("sample_dimension")
        instance_dim = var.attrs.get("instance_dimension")
        if sample_dim is not None and instance_dim is not None:
            raise ValueError(
                f"variable {name!r} declares both 'sample_dimension' and "
                "'instance_dimension'; a variable cannot act as both a "
                "contiguous ragged count variable and an indexed ragged "
                "index variable"
            )
        if sample_dim is not None:
            if var.ndim != 1:
                raise ValueError(
                    f"count variable {name!r} with 'sample_dimension' must be "
                    f"one-dimensional, got {var.ndim} dimensions"
                )
            if sample_dim in count_vars:
                raise ValueError(
                    f"variables {count_vars[sample_dim]!r} and {name!r} both "
                    f"declare 'sample_dimension = {sample_dim!r}'; only one "
                    "count variable is allowed per observation dimension"
                )
            count_vars[sample_dim] = name
        if instance_dim is not None:
            if var.ndim != 1:
                raise ValueError(
                    f"index variable {name!r} with 'instance_dimension' must "
                    f"be one-dimensional, got {var.ndim} dimensions"
                )
            obs_dim = var.dims[0]
            if obs_dim in index_vars:
                raise ValueError(
                    f"variables {index_vars[obs_dim]!r} and {name!r} both "
                    f"declare 'instance_dimension' along dimension "
                    f"{obs_dim!r}; only one index variable is allowed per "
                    "observation dimension"
                )
            index_vars[obs_dim] = name
    return count_vars, index_vars


def _detect_ragged_structure(ds: Dataset) -> _RaggedInfo | None:
    """Detect and validate the CF ragged structure of a dataset.

    Returns a dict with keys ``layout`` ("contiguous" or "indexed"),
    ``instance_dim``, ``obs_dim``, ``count_var`` and ``index_var`` (the latter
    two may be None), or None if no ragged structure is declared.
    """
    count_vars, index_vars = _ragged_declarations(ds)
    shared = set(count_vars) & set(index_vars)
    if shared:
        obs_dim = sorted(shared)[0]
        raise ValueError(
            f"variables {count_vars[obs_dim]!r} and {index_vars[obs_dim]!r} "
            f"declare conflicting ragged layouts for dimension {obs_dim!r}: "
            "a contiguous ragged count variable ('sample_dimension') and an "
            "indexed ragged index variable ('instance_dimension') cannot "
            "describe the same observation dimension"
        )
    if len(count_vars) + len(index_vars) > 1:
        names = sorted(set(count_vars.values()) | set(index_vars.values()))
        raise ValueError(
            f"variables {names!r} declare ragged structures for more than one "
            "observation dimension; only a single ragged structure per "
            "dataset is supported"
        )
    if count_vars:
        obs_dim, count_name = next(iter(count_vars.items()))
        instance_dim = ds[count_name].dims[0]
        if obs_dim not in ds.dims:
            raise ValueError(
                f"count variable {count_name!r} declares "
                f"'sample_dimension = {obs_dim!r}', but no such dimension "
                "exists in the dataset"
            )
        if obs_dim == instance_dim:
            raise ValueError(
                f"count variable {count_name!r} declares "
                f"'sample_dimension = {obs_dim!r}', which is also its own "
                "instance dimension"
            )
        return {
            "layout": "contiguous",
            "instance_dim": instance_dim,
            "obs_dim": obs_dim,
            "count_var": count_name,
            "index_var": None,
        }
    if index_vars:
        obs_dim, index_name = next(iter(index_vars.items()))
        instance_dim = ds[index_name].attrs["instance_dimension"]
        if instance_dim == obs_dim:
            raise ValueError(
                f"index variable {index_name!r} declares "
                f"'instance_dimension = {instance_dim!r}', which is also its "
                "own observation dimension"
            )
        return {
            "layout": "indexed",
            "instance_dim": instance_dim,
            "obs_dim": obs_dim,
            "count_var": None,
            "index_var": index_name,
        }
    return None


def _validate_counts(
    counts: np.ndarray, count_name: str, obs_dim: str, obs_size: int
) -> None:
    if not np.issubdtype(counts.dtype, np.integer):
        raise ValueError(
            f"count variable {count_name!r} must have an integer dtype, "
            f"got {counts.dtype}"
        )
    if (counts < 0).any():
        raise ValueError(f"count variable {count_name!r} contains negative counts")
    total = int(counts.sum())
    if total != obs_size:
        raise ValueError(
            f"count variable {count_name!r} sums to {total}, which does not "
            f"match the size of dimension {obs_dim!r} ({obs_size})"
        )


def _validate_index(
    index: np.ndarray, index_name: str, instance_dim: str, n_instances: int | None
) -> None:
    if not np.issubdtype(index.dtype, np.integer):
        raise ValueError(
            f"index variable {index_name!r} must have an integer dtype, "
            f"got {index.dtype}"
        )
    if (index < 0).any():
        raise ValueError(
            f"index variable {index_name!r} contains negative instance indices"
        )
    if n_instances is not None and index.size and int(index.max()) >= n_instances:
        raise ValueError(
            f"index variable {index_name!r} contains instance index "
            f"{int(index.max())}, which is out of bounds for dimension "
            f"{instance_dim!r} of size {n_instances}"
        )


def decode_ragged(ds: Dataset) -> Dataset:
    """Decode a CF ragged structure into a uniform per-instance representation.

    Observation-level variables keep the observation dimension and gain a
    coordinate holding the integer instance index of each observation, so
    instances can be selected uniformly regardless of the on-disk layout.
    Instance-level variables (including empty instances) are preserved.
    Ragged structure metadata is recorded in ``ds.encoding['ragged']`` so
    that :py:func:`xarray.encode_cf` can re-encode either layout.
    """
    info = _detect_ragged_structure(ds)
    if info is None:
        return ds
    instance_dim = info["instance_dim"]
    obs_dim = info["obs_dim"]
    if info["layout"] == "contiguous":
        count_name = info["count_var"]
        counts = np.asarray(ds[count_name].values)
        _validate_counts(counts, count_name, obs_dim, ds.sizes[obs_dim])
        index = np.repeat(np.arange(counts.size), counts)
        index_name = _free_name(ds, f"{instance_dim}_index")
        ds = ds.assign_coords({index_name: (obs_dim, index)})
    else:
        index_name = info["index_var"]
        index = np.asarray(ds[index_name].values)
        n_instances = ds.sizes.get(instance_dim)
        _validate_index(index, index_name, instance_dim, n_instances)
        if n_instances is None:
            # no instance-level variable defines the instance dimension yet
            n_instances = int(index.max()) + 1 if index.size else 0
            ds = ds.assign_coords({instance_dim: np.arange(n_instances)})
        ds = ds.set_coords(index_name)
    ds.encoding[RAGGED_ENCODING_KEY] = {**info, "index_var": index_name}
    return ds


def _obs_instance_index(ds: Dataset, info: _RaggedInfo) -> np.ndarray:
    """Return the integer instance index of each observation."""
    if info["layout"] == "contiguous":
        counts = np.asarray(ds[info["count_var"]].values)
        return np.repeat(np.arange(counts.size), counts)
    return np.asarray(ds[info["index_var"]].values)


def encode_ragged(
    ds: Dataset, layout: Literal["contiguous", "indexed"] | None = None
) -> Dataset:
    """Encode a dataset with a ragged structure into a CF ragged layout.

    The instance index of each observation is taken from the coordinate
    recorded by :py:func:`decode_ragged` (or derived from existing count or
    index variables). The structural count or index variable is regenerated
    for the chosen layout; when encoding to the contiguous layout the
    observations are reordered so that each instance is stored contiguously,
    preserving the observation order within each instance.
    """
    info = ds.encoding.get(RAGGED_ENCODING_KEY)
    if info is not None and info["index_var"] in ds.variables:
        index = np.asarray(ds[info["index_var"]].values)
    else:
        detected = _detect_ragged_structure(ds)
        if detected is None:
            if layout is not None:
                raise ValueError(
                    f"cannot encode ragged layout {layout!r}: no ragged "
                    "structure was found in the dataset"
                )
            return ds
        info = detected
        index = _obs_instance_index(ds, info)
    if layout is None:
        layout = info["layout"]
    if layout not in ("contiguous", "indexed"):
        raise ValueError(
            f"invalid ragged layout {layout!r}; must be 'contiguous' or 'indexed'"
        )
    instance_dim = info["instance_dim"]
    obs_dim = info["obs_dim"]
    n_instances = ds.sizes[instance_dim]
    counts = np.bincount(index, minlength=n_instances)

    structural_names = [
        name
        for name in (info.get("count_var"), info.get("index_var"))
        if name is not None and name in ds.variables
    ]
    structural_attrs = {name: dict(ds[name].attrs) for name in structural_names}
    for attrs in structural_attrs.values():
        attrs.pop("sample_dimension", None)
        attrs.pop("instance_dimension", None)
    ds = ds.drop_vars(structural_names)

    if layout == "contiguous":
        order = np.argsort(index, kind="stable")
        if not np.array_equal(order, np.arange(index.size)):
            ds = ds.isel({obs_dim: order})
        count_name = info.get("count_var") or _free_name(ds, f"{instance_dim}_count")
        count_attrs = structural_attrs.get(count_name, {})
        count_attrs["sample_dimension"] = obs_dim
        ds[count_name] = Variable((instance_dim,), counts, count_attrs)
    else:
        index_name = info.get("index_var") or _free_name(ds, f"{instance_dim}_index")
        index_attrs = structural_attrs.get(index_name, {})
        index_attrs["instance_dimension"] = instance_dim
        ds[index_name] = Variable((obs_dim,), index, index_attrs)
    return ds
