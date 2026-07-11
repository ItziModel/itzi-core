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

from itzi_core.array_definitions import ARRAY_DEFINITIONS, ArrayCategory


def test_array_definitions():
    # Empty names is acceptable in internal arrays
    for attr in ["csdms_name", "cf_name"]:
        all_values = [
            getattr(arr_def, attr)
            for arr_def in ARRAY_DEFINITIONS
            if ArrayCategory.INTERNAL not in arr_def.category
        ]
        print(all_values)
        # No empty name
        if "" in all_values and not attr == "cf_name":
            print(attr)
            assert False, f"Found empty names in <{attr}>."
        # Make sure there is no duplicates
        values_counts = Counter(all_values)
        duplicates = [item for item, count in values_counts.items() if count > 1]
        if 0 < len(duplicates):
            if attr == "cf_name" and len(duplicates) == 1 and "" in duplicates:
                continue
            assert False, f"Found duplicates in <{attr}>: {duplicates}"

    # All arrays must have a unique key and description
    for attr in ["key", "description"]:
        all_values = [getattr(arr_def, attr) for arr_def in ARRAY_DEFINITIONS]
        print(all_values)
        # No empty name
        if "" in all_values and not attr == "cf_name":
            print(attr)
            assert False, f"Found empty names in <{attr}>."
        # Make sure there is no duplicates
        values_counts = Counter(all_values)
        duplicates = [item for item, count in values_counts.items() if count > 1]
        if 0 < len(duplicates):
            if attr == "cf_name" and len(duplicates) == 1 and "" in duplicates:
                continue
            assert False, f"Found duplicates in <{attr}>: {duplicates}"
