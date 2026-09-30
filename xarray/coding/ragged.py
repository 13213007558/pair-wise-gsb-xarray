"""Explicit support for CF ragged array representations of trajectory data.

CF conventions (chapter 9, "Discrete Sampling Geometries") describe two
ways to store a collection of trajectories (instances) with a varying
number of observations each in flat arrays:

- *contiguous ragged*: a count variable with a 'sample_dimension'
  attribute stores the number of observations per instance.  The
  observations of one instance occupy consecutive rows along the sample
  (observation) dimension.
- *indexed ragged*: an index variable with an 'instance_dimension'
  attribute stores the instance index of every observation.  Observations
  of different instances may be interleaved.

This module decodes both layouts into a single unified in-memory
representation:

- observation-level variables keep the observation dimension,
- instance-level variables keep the instance dimension,
- an integer coordinate on the observation dimension maps every
  observation to its instance, so instances can be selected with
  select_ragged_instances without padding to a dense
  (instance, max_obs) matrix.

The structural variables (count or index variable) are not treated as
ordinary data variables; the information needed to re-encode either
layout is stored in the "ragged" entry of the instance-id coordinate's
encoding.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np

from xarray.core.variable import Variable

if TYPE_CHECKING:
    from xarray.core.dataset import Dataset

    T_Variables = TypeVar("T_Variables", bound=Mapping[Any, Variable])

#: encoding key under which the ragged structure description is stored on
#: the instance-id coordinate of a decoded dataset
RAGGED_ENCODING_KEY = "ragged"

RaggedLayout = Literal["contiguous", "indexed"]

_RAGGED_LAYOUTS = ("contiguous", "indexed")


def _dim_sizes(variables: Mapping[Any, Variable]) -> dict[Hashable, int]:
    sizes: dict[Hashable, int] = {}
    for var in variables.values():
        for dim, size in zip(var.dims, var.shape, strict=True):
            sizes[dim] = size
    return sizes


def _find_ragged_declarations(
    variables: Mapping[Any, Variable],
) -> tuple[dict[Hashable, Hashable], dict[Hashable, Hashable]]:
    """Find ragged structure declarations.

    Returns
    -------
    count_decls : dict
        Maps count variable names to their 'sample_dimension'.
    index_decls : dict
        Maps index variable names to their 'instance_dimension'.
    """
    count_decls: dict[Hashable, Hashable] = {}
    index_decls: dict[Hashable, Hashable] = {}
    for name, var in variables.items():
        sample_dim = var.attrs.get("sample_dimension")
        instance_dim = var.attrs.get("instance_dimension")
        if sample_dim is not None and instance_dim is not None:
            raise ValueError(
                f"Variable {name!r} declares both a 'sample_dimension' and an "
                "'instance_dimension' attribute; a variable cannot be both a "
                "contiguous ragged count variable and an indexed ragged "
                "instance index variable."
            )
        if sample_dim is not None:
            count_decls[name] = sample_dim
        if instance_dim is not None:
            index_decls[name] = instance_dim
    return count_decls, index_decls


def _check_integer_dtype(var: Variable, name: Hashable, role: str) -> None:
    if var.dtype.kind not in "iu":
        raise ValueError(
            f"Ragged {role} variable {name!r} must have an integer dtype, "
            f"got dtype {var.dtype}."
        )


def _unique_id_name(instance_dim: Hashable, variables: Mapping[Any, Variable]) -> str:
    base = f"{instance_dim}_id"
    name = base
    while name in variables:
        name += "_"
    return name


def decode_ragged_variables(
    variables: T_Variables,
) -> tuple[dict[Hashable, Variable], set[Hashable]]:
    """Decode CF ragged array structures into the unified representation.

    Parameters
    ----------
    variables : mapping of Variable
        CF decoded variables, possibly containing ragged structure
        declarations ('sample_dimension' / 'instance_dimension'
        attributes).

    Returns
    -------
    new_variables : dict
        Variables in the unified ragged representation.  Count variables
        of contiguous ragged arrays are removed (they are fully described
        by the instance-id coordinate); index variables of indexed ragged
        arrays are kept as the instance-id coordinate.
    coord_names : set
        Names of the created instance-id coordinates.
    """
    count_decls, index_decls = _find_ragged_declarations(variables)
    if not count_decls and not index_decls:
        return dict(variables), set()

    sizes = _dim_sizes(variables)

    # group declarations by observation (sample) dimension
    structures: dict[Hashable, dict[str, Hashable]] = {}
    for name, sample_dim in count_decls.items():
        entry = structures.setdefault(sample_dim, {})
        if "count" in entry:
            raise ValueError(
                f"Variables {entry['count']!r} and {name!r} both declare a "
                f"'sample_dimension' for the same ragged structure along "
                f"dimension {sample_dim!r}."
            )
        entry["count"] = name
    for name in index_decls:
        var = variables[name]
        if var.ndim != 1:
            raise ValueError(
                f"Indexed ragged index variable {name!r} must be "
                f"one-dimensional, got dimensions {var.dims!r}."
            )
        entry = structures.setdefault(var.dims[0], {})
        if "index" in entry:
            raise ValueError(
                f"Variables {entry['index']!r} and {name!r} both declare an "
                f"'instance_dimension' for the same ragged structure along "
                f"dimension {var.dims[0]!r}."
            )
        entry["index"] = name

    for obs_dim, entry in structures.items():
        if "count" in entry and "index" in entry:
            raise ValueError(
                f"Variables {entry['count']!r} (with 'sample_dimension') and "
                f"{entry['index']!r} (with 'instance_dimension') declare "
                f"conflicting ragged layouts for the same observation "
                f"dimension {obs_dim!r}."
            )
        if obs_dim not in sizes:
            name = entry.get("count", entry.get("index"))
            raise ValueError(
                f"Ragged structure declared by variable {name!r} refers to "
                f"dimension {obs_dim!r}, which is not used by any variable."
            )

    new_vars = dict(variables)
    coord_names: set[Hashable] = set()
    for obs_dim, entry in structures.items():
        if "count" in entry:
            id_name, id_var = _decode_contiguous(
                entry["count"], variables[entry["count"]], obs_dim, sizes, variables
            )
        else:
            id_name, id_var = _decode_indexed(
                entry["index"], variables[entry["index"]], sizes
            )
        new_vars[id_name] = id_var
        coord_names.add(id_name)
        if "count" in entry:
            # the count variable is an encoding artifact: the per-instance
            # counts are fully described by the instance-id coordinate
            del new_vars[entry["count"]]
    return new_vars, coord_names


def _decode_contiguous(
    name: Hashable,
    var: Variable,
    obs_dim: Hashable,
    sizes: dict[Hashable, int],
    variables: Mapping[Any, Variable],
) -> tuple[Hashable, Variable]:
    if var.ndim != 1:
        raise ValueError(
            f"Contiguous ragged count variable {name!r} must be "
            f"one-dimensional, got dimensions {var.dims!r}."
        )
    _check_integer_dtype(var, name, "count")
    instance_dim = var.dims[0]
    counts = np.asarray(var.data)
    if (counts < 0).any():
        raise ValueError(
            f"Contiguous ragged count variable {name!r} contains negative "
            "counts."
        )
    total = int(counts.sum())
    if total != sizes[obs_dim]:
        raise ValueError(
            f"Contiguous ragged count variable {name!r} sums to {total}, "
            f"which does not match the size {sizes[obs_dim]} of the sample "
            f"dimension {obs_dim!r}."
        )
    n_instances = int(counts.size)
    ids = np.repeat(np.arange(n_instances), counts)
    spec = {
        "layout": "contiguous",
        "instance_dim": instance_dim,
        "obs_dim": obs_dim,
        "count_var": name,
        "index_var": None,
        "n_instances": n_instances,
        "count_attrs": {k: v for k, v in var.attrs.items() if k != "sample_dimension"},
        "count_dtype": var.dtype,
    }
    id_name = _unique_id_name(instance_dim, variables)
    id_var = Variable((obs_dim,), ids, encoding={RAGGED_ENCODING_KEY: spec})
    return id_name, id_var


def _decode_indexed(
    name: Hashable, var: Variable, sizes: dict[Hashable, int]
) -> tuple[Hashable, Variable]:
    _check_integer_dtype(var, name, "instance index")
    instance_dim = var.attrs["instance_dimension"]
    obs_dim = var.dims[0]
    ids = np.asarray(var.data)
    if ids.size and int(ids.min()) < 0:
        raise ValueError(
            f"Indexed ragged index variable {name!r} contains negative "
            "instance indices."
        )
    max_id = int(ids.max()) if ids.size else -1
    if instance_dim in sizes:
        n_instances = sizes[instance_dim]
        if max_id >= n_instances:
            raise ValueError(
                f"Indexed ragged index variable {name!r} refers to instance "
                f"index {max_id}, but the instance dimension "
                f"{instance_dim!r} has size {n_instances}."
            )
    else:
        n_instances = max_id + 1
    attrs = {k: v for k, v in var.attrs.items() if k != "instance_dimension"}
    spec = {
        "layout": "indexed",
        "instance_dim": instance_dim,
        "obs_dim": obs_dim,
        "count_var": None,
        "index_var": name,
        "n_instances": int(n_instances),
    }
    encoding = dict(var.encoding)
    encoding[RAGGED_ENCODING_KEY] = spec
    id_var = Variable(var.dims, var._data, attrs=attrs, encoding=encoding)
    return name, id_var


def _scrub_coordinates_attr(
    variables: dict[Hashable, Variable], removed_name: Hashable
) -> None:
    """Remove references to a dropped variable from 'coordinates' attributes."""
    removed = str(removed_name)
    for var in variables.values():
        for mapping in (var.attrs, var.encoding):
            coordinates = mapping.get("coordinates")
            if not isinstance(coordinates, str) or removed not in coordinates:
                continue
            remaining = [c for c in coordinates.split() if c != removed]
            if remaining:
                mapping["coordinates"] = " ".join(remaining)
            else:
                del mapping["coordinates"]


def encode_ragged_variables(
    variables: T_Variables, layout: RaggedLayout | None = None
) -> dict[Hashable, Variable]:
    """Encode the unified ragged representation into a CF ragged layout.

    Parameters
    ----------
    variables : mapping of Variable
        Variables in the unified ragged representation, i.e. containing
        instance-id coordinates carrying a "ragged" encoding entry.
    layout : {"contiguous", "indexed"}, optional
        Ragged layout to encode.  If None (default), every ragged
        structure is encoded in the layout it was decoded from.

    Returns
    -------
    new_variables : dict
        Variables with CF ragged structure variables (count or index
        variables) restored and instance-id coordinates removed or
        converted as required by the chosen layout.
    """
    specs = {
        name: var.encoding[RAGGED_ENCODING_KEY]
        for name, var in variables.items()
        if RAGGED_ENCODING_KEY in var.encoding
    }
    if not specs:
        return dict(variables)
    if layout is not None and layout not in _RAGGED_LAYOUTS:
        raise ValueError(
            f"Invalid ragged layout {layout!r}; expected one of {_RAGGED_LAYOUTS}."
        )

    sizes = _dim_sizes(variables)
    new_vars = dict(variables)
    for name, spec in specs.items():
        var = variables[name]
        effective_layout: RaggedLayout = layout or spec["layout"]
        instance_dim = spec["instance_dim"]
        obs_dim = spec["obs_dim"]
        n_instances = sizes.get(instance_dim, spec["n_instances"])
        ids = np.asarray(var.data)
        if ids.dtype.kind not in "iu":
            raise ValueError(
                f"Ragged instance-id variable {name!r} must have an integer "
                f"dtype, got dtype {ids.dtype}."
            )
        if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= n_instances):
            raise ValueError(
                f"Ragged instance-id variable {name!r} contains instance "
                f"indices outside the valid range [0, {n_instances})."
            )
        encoding = {k: v for k, v in var.encoding.items() if k != RAGGED_ENCODING_KEY}
        if effective_layout == "indexed":
            attrs = dict(var.attrs)
            attrs["instance_dimension"] = instance_dim
            new_vars[name] = Variable((obs_dim,), ids, attrs=attrs, encoding=encoding)
        else:
            # the contiguous layout requires the observations of every
            # instance to be grouped together; use a stable sort so that
            # the observation order within each instance is preserved
            order = np.argsort(ids, kind="stable")
            if not np.array_equal(order, np.arange(ids.size)):
                for other_name, other_var in new_vars.items():
                    if obs_dim in other_var.dims:
                        new_vars[other_name] = other_var.isel({obs_dim: order})
                ids = ids[order]
            counts = np.bincount(ids, minlength=n_instances)
            count_dtype = spec.get("count_dtype")
            if count_dtype is not None:
                counts = counts.astype(count_dtype)
            count_name = spec["count_var"] or f"{instance_dim}_count"
            if count_name in new_vars:
                raise ValueError(
                    f"Cannot encode ragged structure in contiguous layout: "
                    f"count variable name {count_name!r} is already used by "
                    "another variable."
                )
            count_attrs = dict(spec.get("count_attrs", {}))
            count_attrs["sample_dimension"] = obs_dim
            new_vars[count_name] = Variable((instance_dim,), counts, attrs=count_attrs)
            del new_vars[name]
            _scrub_coordinates_attr(new_vars, name)
    return new_vars


def ragged_specs(ds: Dataset) -> dict[Hashable, dict[str, Any]]:
    """Return the ragged structure descriptions of a decoded dataset."""
    return {
        name: var.encoding[RAGGED_ENCODING_KEY]
        for name, var in ds.variables.items()
        if RAGGED_ENCODING_KEY in var.encoding
    }


def select_ragged_instances(ds: Dataset, instances: int | Iterable[int]) -> Dataset:
    """Select trajectories (instances) of a decoded ragged dataset.

    Observations of the selected instances keep their original relative
    order along the observation dimension; no dense padding is involved.
    Empty instances (without any observation) are preserved.

    Parameters
    ----------
    ds : Dataset
        Dataset decoded with ragged support (see decode_cf).
    instances : int or iterable of int
        Positional instance indices to select, in the desired output
        order.

    Returns
    -------
    Dataset
        Dataset restricted to the selected instances, with instance-id
        coordinates remapped to the new instance positions.
    """
    specs = ragged_specs(ds)
    if not specs:
        raise ValueError(
            "Dataset does not contain a decoded ragged structure; "
            "decode it with decode_cf(..., decode_ragged=True) first."
        )
    if isinstance(instances, Iterable):
        positions = np.asarray(list(instances))
    else:
        positions = np.asarray([instances])
    if positions.dtype.kind not in "iu":
        raise ValueError(
            f"Instance indices must be integers, got dtype {positions.dtype}."
        )
    if len(np.unique(positions)) != len(positions):
        raise ValueError(f"Instance indices must be unique, got {positions!r}.")

    result = ds
    for name, spec in specs.items():
        instance_dim = spec["instance_dim"]
        obs_dim = spec["obs_dim"]
        n_instances = result.sizes.get(instance_dim, spec["n_instances"])
        if positions.size and (
            int(positions.min()) < 0 or int(positions.max()) >= n_instances
        ):
            raise ValueError(
                f"Instance indices {positions!r} are out of bounds for "
                f"ragged structure with {n_instances} instances "
                f"(instance-id variable {name!r})."
            )
        ids = np.asarray(result[name].data)
        obs_indexers = np.nonzero(np.isin(ids, positions))[0]
        remap = np.full(n_instances, -1, dtype=np.intp)
        remap[positions] = np.arange(len(positions))
        new_ids = remap[ids[obs_indexers]]
        indexers: dict[Hashable, Any] = {obs_dim: obs_indexers}
        if instance_dim in result.sizes:
            indexers[instance_dim] = positions
        result = result.isel(indexers)
        new_id_var = result[name].variable.copy(data=new_ids)
        result = result.assign_coords({name: new_id_var})
        new_spec = dict(result[name].encoding.get(RAGGED_ENCODING_KEY, spec))
        new_spec["n_instances"] = len(positions)
        result[name].encoding[RAGGED_ENCODING_KEY] = new_spec
    return result
