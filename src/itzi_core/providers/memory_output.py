"""
Copyright (C) 2025-2026 Laurent G. Courty

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 2.1
of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.
"""

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np

from itzi_core.data_containers import DrainageNetworkAttributes, DrainageNetworkTopology
from itzi_core.providers.base import RasterOutputProvider, VectorOutputProvider


class MemoryRasterOutputProvider(RasterOutputProvider):
    """Save rasters in memory as numpy arrays."""

    def __init__(self, out_map_names: Mapping[str, str]) -> None:
        """Initialize output provider with simulation configuration."""
        # user-selected map names.
        self.out_map_names = out_map_names
        self.output_maps_dict: dict[str, list[tuple[datetime | timedelta, np.ndarray]]] = {
            key: [] for key in self.out_map_names
        }

    def write_arrays(
        self, array_dict: Mapping[str, np.ndarray], sim_time: datetime | timedelta
    ) -> None:
        for arr_key, arr in array_dict.items():
            if isinstance(arr, np.ndarray):
                self.output_maps_dict[arr_key].append((deepcopy(sim_time), arr.copy()))


class MemoryVectorOutputProvider(VectorOutputProvider):
    """Save drainage simulation outputs in memory."""

    def __init__(self) -> None:
        """Initialize output provider with simulation configuration."""
        self.drainage_topology: DrainageNetworkTopology | None = None
        self.drainage_attributes: list[tuple[datetime | timedelta, DrainageNetworkAttributes]] = []

    def write_topology(self, topology: DrainageNetworkTopology) -> None:
        """Save the fixed drainage-network topology."""
        if self.drainage_topology is not None:
            raise RuntimeError("Drainage topology has already been written.")
        self.drainage_topology = deepcopy(topology)

    def write_attributes(
        self, attributes: DrainageNetworkAttributes, sim_time: datetime | timedelta
    ) -> None:
        """Save drainage attributes for the current time step."""
        if self.drainage_topology is None:
            raise RuntimeError("Drainage attributes cannot be written before topology.")
        self.drainage_attributes.append((deepcopy(sim_time), deepcopy(attributes)))
