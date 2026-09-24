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

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from itzi_core.const import TemporalType
from itzi_core.data_containers import SimulationConfig, SurfaceFlowParameters
from itzi_core.domain_data import DomainData
from itzi_core.providers.memory_input import MemoryRasterInputProvider
from itzi_core.simulation_builder import SimulationBuilder


def _simulation_config(helpers) -> SimulationConfig:
    start_time = datetime(2000, 1, 1, tzinfo=UTC)
    return SimulationConfig(
        start_time=start_time,
        end_time=start_time + timedelta(seconds=1),
        record_step=timedelta(seconds=1),
        temporal_type=TemporalType.RELATIVE,
        input_map_names=helpers.make_input_map_names(
            ground_elevation="dem",
            friction="friction",
            water_depth="water_depth",
        ),
        output_map_names=helpers.make_output_map_names("builder_test", []),
        surface_flow_parameters=SurfaceFlowParameters(),
    )


def _builder(domain_5by5, config: SimulationConfig) -> SimulationBuilder:
    return SimulationBuilder(config, domain_5by5.arr_mask).with_domain_data(
        domain_5by5.domain_data
    )


def _input_provider(
    domain_data: DomainData, config: SimulationConfig
) -> MemoryRasterInputProvider:
    return MemoryRasterInputProvider(
        {
            "domain_data": domain_data,
            "simulation_start_time": config.start_time,
            "simulation_end_time": config.end_time,
        }
    )


def test_input_provider_cannot_overwrite_domain_data(domain_5by5, helpers) -> None:
    config = _simulation_config(helpers)
    builder = _builder(domain_5by5, config)
    input_provider = _input_provider(domain_5by5.domain_data, config)

    with pytest.raises(ValueError, match="DomainData is already set"):
        builder.with_input_provider(input_provider)


def test_domain_data_cannot_overwrite_input_provider_domain_data(domain_5by5, helpers) -> None:
    config = _simulation_config(helpers)
    input_provider = _input_provider(domain_5by5.domain_data, config)
    builder = SimulationBuilder(config, domain_5by5.arr_mask).with_input_provider(input_provider)

    with pytest.raises(ValueError, match="DomainData is already set"):
        builder.with_domain_data(domain_5by5.domain_data)


def test_configured_raster_outputs_require_a_provider(domain_5by5, helpers) -> None:
    config = _simulation_config(helpers).model_copy(
        update={"output_map_names": {"water_depth": "depth"}}
    )

    with pytest.raises(ValueError, match="raster output provider"):
        _builder(domain_5by5, config).build()


def test_configured_drainage_output_requires_a_vector_provider(domain_5by5, helpers) -> None:
    config = _simulation_config(helpers).model_copy(update={"drainage_output": "drainage"})

    with pytest.raises(ValueError, match="vector output provider"):
        _builder(domain_5by5, config).build()


def test_drainage_without_configured_output_does_not_require_a_vector_provider(
    domain_5by5, helpers, monkeypatch
) -> None:
    config = _simulation_config(helpers).model_copy(update={"swmm_inp": "drainage.inp"})
    builder = _builder(domain_5by5, config)
    monkeypatch.setattr(builder, "_create_drainage_simulation", lambda domain_data: ((), None))

    simulation = builder.build()

    assert simulation.report.vector_provider is None


def test_simulation_config_rejects_removed_stats_file() -> None:
    assert "stats_file" not in SimulationConfig.model_fields
    start_time = datetime(2000, 1, 1, tzinfo=UTC)

    with pytest.raises(ValidationError, match="stats_file"):
        SimulationConfig.model_validate(
            {
                "start_time": start_time,
                "end_time": start_time + timedelta(seconds=1),
                "record_step": timedelta(seconds=1),
                "temporal_type": TemporalType.RELATIVE,
                "input_map_names": {},
                "output_map_names": {},
                "surface_flow_parameters": SurfaceFlowParameters(),
                "stats_file": "removed.csv",
            }
        )
