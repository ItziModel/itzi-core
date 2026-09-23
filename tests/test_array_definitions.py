"""
Copyright (C) 2026 Laurent G. Courty

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 2.1
of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.
"""

from collections import Counter
from dataclasses import FrozenInstanceError

import pytest

from itzi_core.array_definitions import ARRAY_DEFINITIONS, ArrayCategory


def test_array_definitions():
    # Empty names is acceptable in internal arrays
    csdms_names = [
        arr_def.csdms_name
        for arr_def in ARRAY_DEFINITIONS
        if ArrayCategory.INTERNAL not in arr_def.category
    ]
    assert "" not in csdms_names, "Found empty names in <csdms_name>."
    values_counts = Counter(csdms_names)
    duplicates = [item for item, count in values_counts.items() if count > 1]
    assert not duplicates, f"Found duplicates in <csdms_name>: {duplicates}"

    # All arrays must have a unique key and description
    for attr in ["key", "description"]:
        all_values = [getattr(arr_def, attr) for arr_def in ARRAY_DEFINITIONS]
        # No empty name
        if "" in all_values:
            assert False, f"Found empty names in <{attr}>."
        # Make sure there is no duplicates
        values_counts = Counter(all_values)
        duplicates = [item for item, count in values_counts.items() if count > 1]
        assert not duplicates, f"Found duplicates in <{attr}>: {duplicates}"


def test_maximum_arrays_are_internal_outputs():
    definitions = {arr_def.key: arr_def for arr_def in ARRAY_DEFINITIONS}
    for key in ("hmax", "vmax"):
        assert ArrayCategory.INTERNAL in definitions[key].category
        assert ArrayCategory.OUTPUT in definitions[key].category


def test_computed_from_arrays_are_defined():
    keys = {arr_def.key for arr_def in ARRAY_DEFINITIONS}
    for arr_def in ARRAY_DEFINITIONS:
        if arr_def.computes_from is not None:
            assert arr_def.computes_from in keys
            assert arr_def.computes_from != arr_def.key


def test_array_catalog_is_immutable() -> None:
    assert isinstance(ARRAY_DEFINITIONS, tuple)
    definition = ARRAY_DEFINITIONS[0]
    assert isinstance(definition.category, tuple)
    with pytest.raises(FrozenInstanceError):
        definition.key = "changed"  # ty: ignore[invalid-assignment]
