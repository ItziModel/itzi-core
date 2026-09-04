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

import numpy as np
import pytest

from itzi_core.const import TemporalType
from itzi_core.data_containers import ContinuityData, SimulationData
from itzi_core.providers.memory_output import (
    MemoryRasterOutputProvider,
    MemoryVectorOutputProvider,
)
from itzi_core.report import Report
from tests.fixtures_vector_output import create_dummy_drainage_network

CONTINUITY_DATA = ContinuityData(
    new_domain_vol=0.0,
    volume_change=0.0,
    created_volume=0.0,
    created_volume_ratio=0.0,
)


def test_get_output_arrays_returns_a_fresh_selection() -> None:
    start_time = datetime(2000, 1, 1, tzinfo=UTC)
    out_map_names = {"water_depth": "depth", "hmax": "depth_max"}
    raster_provider = MemoryRasterOutputProvider(out_map_names)
    report = Report(
        start_time=start_time,
        temporal_type=TemporalType.ABSOLUTE,
        raster_output_provider=raster_provider,
        vector_output_provider=MemoryVectorOutputProvider(),
        mass_balance_output_provider=None,
        out_map_names=out_map_names,
        dt=timedelta(seconds=1),
    )
    data = SimulationData(
        sim_time=start_time,
        time_step=1.0,
        time_steps_counter=0,
        continuity_data=CONTINUITY_DATA,
        raw_arrays={
            "water_depth": np.array([[1.0]], dtype=np.float32),
            "hmax": np.array([[2.0]], dtype=np.float32),
        },
        accumulation_arrays={},
        cell_dx=1.0,
        cell_dy=1.0,
        drainage_network_attributes=None,
    )

    report.step(data)
    del out_map_names["water_depth"]
    report.step(data)

    assert len(raster_provider.output_maps_dict["water_depth"]) == 1
    assert len(raster_provider.output_maps_dict["hmax"]) == 2


def test_maxima_are_selected_independently_of_base_arrays() -> None:
    start_time = datetime(2000, 1, 1, tzinfo=UTC)
    out_map_names = {"hmax": "depth_max", "vmax": "speed_max"}
    raster_provider = MemoryRasterOutputProvider(out_map_names)
    report = Report(
        start_time=start_time,
        temporal_type=TemporalType.ABSOLUTE,
        raster_output_provider=raster_provider,
        vector_output_provider=MemoryVectorOutputProvider(),
        mass_balance_output_provider=None,
        out_map_names=out_map_names,
        dt=timedelta(seconds=1),
    )
    data = SimulationData(
        sim_time=start_time,
        time_step=1.0,
        time_steps_counter=0,
        continuity_data=CONTINUITY_DATA,
        raw_arrays={
            "water_depth": np.array([[1.0]], dtype=np.float32),
            "hmax": np.array([[2.0]], dtype=np.float32),
            "v": np.array([[3.0]], dtype=np.float32),
            "vmax": np.array([[4.0]], dtype=np.float32),
        },
        accumulation_arrays={},
        cell_dx=1.0,
        cell_dy=1.0,
        drainage_network_attributes=None,
    )

    report.step(data)

    assert set(raster_provider.output_maps_dict) == {"hmax", "vmax"}
    np.testing.assert_array_equal(
        raster_provider.output_maps_dict["hmax"][0][1], data.raw_arrays["hmax"]
    )
    np.testing.assert_array_equal(
        raster_provider.output_maps_dict["vmax"][0][1], data.raw_arrays["vmax"]
    )


def test_drainage_topology_is_written_before_attributes() -> None:
    start_time = datetime(2000, 1, 1, tzinfo=UTC)
    raster_provider = MemoryRasterOutputProvider({})
    vector_provider = MemoryVectorOutputProvider()
    report = Report(
        start_time=start_time,
        temporal_type=TemporalType.RELATIVE,
        raster_output_provider=raster_provider,
        vector_output_provider=vector_provider,
        mass_balance_output_provider=None,
        out_map_names={},
        dt=timedelta(seconds=60),
    )
    drainage_network = create_dummy_drainage_network()
    data = SimulationData(
        sim_time=start_time,
        time_step=60.0,
        time_steps_counter=1,
        continuity_data=CONTINUITY_DATA,
        raw_arrays={},
        accumulation_arrays={},
        cell_dx=1.0,
        cell_dy=1.0,
        drainage_network_attributes=drainage_network.attributes,
    )

    with pytest.raises(RuntimeError, match="before topology"):
        report.step(data)
    assert not raster_provider.output_maps_dict

    report.start(drainage_network.topology)
    report.start(drainage_network.topology)
    report.step(data)
    report.step(data.model_copy(update={"sim_time": start_time + timedelta(seconds=60)}))

    assert vector_provider.drainage_topology == drainage_network.topology
    assert vector_provider.drainage_attributes == [
        (timedelta(0), drainage_network.attributes),
        (timedelta(seconds=60), drainage_network.attributes),
    ]
