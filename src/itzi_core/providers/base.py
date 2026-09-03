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

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime, timedelta

    import numpy as np

    from itzi_core.data_containers import (
        DrainageNetworkAttributes,
        DrainageNetworkTopology,
        MassBalanceData,
    )
    from itzi_core.providers.domain_data import DomainData


class RasterInputProvider(ABC):
    """Abstract base class for handling raster simulation inputs."""

    def get_origin(self) -> tuple[float, float]:
        """Return the coordinates of the NW corner
        as a tuple (N, W)"""
        domain_data: DomainData = self.get_domain_data()
        return (domain_data.north, domain_data.west)

    @abstractmethod
    def get_domain_data(self) -> DomainData:
        """Return a DomainData object."""

    @abstractmethod
    def get_array(
        self, map_key: str, current_time: datetime
    ) -> tuple[np.ndarray | None, datetime, datetime]:
        """Take a given map key and current time
        return a numpy array associated with its half-open validity window `[start, end)`.

        If no array is active at `current_time`, return `None` plus the half-open interval
        for which that "no data" result remains valid.
        """


class RasterOutputProvider(ABC):
    """Abstract base class for handling raster simulation outputs."""

    @abstractmethod
    def write_arrays(
        self, array_dict: Mapping[str, np.ndarray], sim_time: datetime | timedelta
    ) -> None:
        """Write all arrays for the current time step."""

    def finalize(self) -> None:
        """Flush and close provider resources."""


class VectorOutputProvider(ABC):
    """Abstract base class for drainage simulation outputs."""

    @abstractmethod
    def write_topology(self, topology: DrainageNetworkTopology) -> None:
        """Write the fixed drainage-network topology once."""

    @abstractmethod
    def write_attributes(
        self,
        attributes: DrainageNetworkAttributes,
        sim_time: datetime | timedelta,
    ) -> None:
        """Write drainage attributes for the current time step."""

    def finalize(self) -> None:
        """Flush and close provider resources."""


class MassBalanceOutputProvider(ABC):
    """Abstract base class for mass-balance outputs."""

    @abstractmethod
    def log(self, report_data: MassBalanceData) -> None:
        """Persist a completed mass-balance report."""

    def finalize(self) -> None:
        """Finalize outputs and cleanup."""
