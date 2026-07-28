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

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import ValidationError

from itzi_core.const import TemporalType
from itzi_core.data_containers import MassBalanceData, SimulationConfig, SurfaceFlowParameters
from itzi_core.providers.base import MassBalanceOutputProvider
from itzi_core.providers.memory_output import (
    MemoryRasterOutputProvider,
    MemoryVectorOutputProvider,
)
from itzi_core.simulation_builder import SimulationBuilder

if TYPE_CHECKING:
    from itzi_core.simulation import Simulation


class RecordingMassBalanceProvider(MassBalanceOutputProvider):
    def __init__(self) -> None:
        self.reports: list[MassBalanceData] = []
        self.finalize_calls = 0

    def log(self, report_data: MassBalanceData) -> None:
        self.reports.append(report_data)

    def finalize(self) -> None:
        self.finalize_calls += 1


class FalseyRecordingMassBalanceProvider(RecordingMassBalanceProvider):
    def __bool__(self) -> bool:
        return False


def _build_simulation(
    domain_5by5,
    helpers,
    *,
    provider: MassBalanceOutputProvider | None = None,
) -> Simulation:
    start_time = datetime(2000, 1, 1, tzinfo=UTC)
    sim_config = SimulationConfig(
        start_time=start_time,
        end_time=start_time + timedelta(seconds=1),
        record_step=timedelta(seconds=1),
        temporal_type=TemporalType.RELATIVE,
        input_map_names=helpers.make_input_map_names(
            dem="dem",
            friction="friction",
            water_depth="water_depth",
        ),
        output_map_names=helpers.make_output_map_names("provider_test", []),
        surface_flow_parameters=SurfaceFlowParameters(),
    )
    builder = (
        SimulationBuilder(sim_config, domain_5by5.arr_mask, np.float32)
        .with_domain_data(domain_5by5.domain_data)
        .with_raster_output_provider(MemoryRasterOutputProvider(sim_config.output_map_names))
        .with_vector_output_provider(MemoryVectorOutputProvider())
    )
    if provider is not None:
        assert builder.with_mass_balance_output_provider(provider) is builder

    simulation = builder.build()
    simulation.set_array("dem", domain_5by5.arr_dem_flat)
    simulation.set_array("friction", domain_5by5.arr_n)
    simulation.set_array("water_depth", domain_5by5.arr_start_h)
    return simulation


def test_custom_provider_receives_simulation_mass_balance(domain_5by5, helpers) -> None:
    provider = RecordingMassBalanceProvider()
    simulation = _build_simulation(domain_5by5, helpers, provider=provider)

    simulation.initialize()
    simulation.update()

    assert len(provider.reports) == 2
    assert all(isinstance(report, MassBalanceData) for report in provider.reports)
    assert provider.reports[-1].simulation_time == timedelta(seconds=1)


def test_falsey_injected_provider_is_used(domain_5by5, helpers) -> None:
    provider = FalseyRecordingMassBalanceProvider()

    simulation = _build_simulation(domain_5by5, helpers, provider=provider)
    simulation.initialize()

    assert simulation.report.mass_balance_output_provider is provider
    assert len(provider.reports) == 1


def test_mass_balance_output_is_disabled_without_configuration(domain_5by5, helpers) -> None:
    simulation = _build_simulation(domain_5by5, helpers)

    simulation.initialize()

    assert simulation.report.mass_balance_output_provider is None


def test_simulation_config_rejects_removed_stats_file() -> None:
    assert "stats_file" not in SimulationConfig.model_fields
    start_time = datetime(2000, 1, 1, tzinfo=UTC)

    with pytest.raises(ValidationError, match="stats_file"):
        SimulationConfig(
            start_time=start_time,
            end_time=start_time + timedelta(seconds=1),
            record_step=timedelta(seconds=1),
            temporal_type=TemporalType.RELATIVE,
            input_map_names={},
            output_map_names={},
            surface_flow_parameters=SurfaceFlowParameters(),
            stats_file="removed.csv",
        )


def test_finalize_calls_mass_balance_provider_once(domain_5by5, helpers) -> None:
    provider = FalseyRecordingMassBalanceProvider()
    simulation = _build_simulation(domain_5by5, helpers, provider=provider)
    simulation.initialize()

    simulation.finalize()

    assert provider.finalize_calls == 1
