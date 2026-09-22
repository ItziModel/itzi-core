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

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from itzi_core import (
    ARRAY_DEFINITIONS,
    ArrayCategory,
    ArrayDefinition,
    DomainData,
    HotstartRunConfig,
    InfiltrationModelType,
    InputWindow,
    Simulation,
    SimulationBuilder,
    SimulationConfig,
    SurfaceFlowParameters,
    TemporalType,
    estimate_surface_flow_timestep,
)
from itzi_core.providers import (
    CSVMassBalanceOutputProvider,
    DrainageLinkAttributes,
    DrainageLinkTopology,
    DrainageNetworkAttributes,
    DrainageNetworkTopology,
    DrainageNodeAttributes,
    DrainageNodeTopology,
    MassBalanceData,
    MassBalanceOutputProvider,
    RasterInputProvider,
    RasterOutputProvider,
    VectorOutputProvider,
)
from itzi_core.providers import (
    DomainData as ProviderDomainData,
)


def test_application_api_exports() -> None:
    assert ARRAY_DEFINITIONS
    assert all(isinstance(definition, ArrayDefinition) for definition in ARRAY_DEFINITIONS)
    assert ArrayCategory.INPUT.value == "INPUT"
    assert DomainData
    assert HotstartRunConfig
    assert InfiltrationModelType.NULL.value == "null"
    assert InputWindow
    assert Simulation
    assert SimulationBuilder
    assert SimulationConfig
    assert SurfaceFlowParameters
    assert TemporalType.RELATIVE.value == "relative"
    assert estimate_surface_flow_timestep(0.5, 2.0, 10.0, 0.4) == 0.5


def test_provider_api_exports() -> None:
    assert CSVMassBalanceOutputProvider
    assert ProviderDomainData is DomainData
    assert DrainageLinkAttributes
    assert DrainageLinkTopology
    assert DrainageNetworkAttributes
    assert DrainageNetworkTopology
    assert DrainageNodeAttributes
    assert DrainageNodeTopology
    assert MassBalanceData
    assert MassBalanceOutputProvider
    assert RasterInputProvider
    assert RasterOutputProvider
    assert VectorOutputProvider


def test_input_window_is_an_immutable_snapshot() -> None:
    window = InputWindow(
        origin=(10.0, 20.0),
        start=datetime(2000, 1, 1, tzinfo=UTC),
        end=datetime(2000, 1, 2, tzinfo=UTC),
    )

    with pytest.raises(ValidationError):
        window.start = datetime(2001, 1, 1, tzinfo=UTC)  # ty: ignore[invalid-assignment]
