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

from datetime import datetime, timedelta
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import pytest

from itzi_core.const import InfiltrationModelType, TemporalType
from itzi_core.data_containers import SimulationConfig, SurfaceFlowParameters
from itzi_core.providers.memory_output import (
    MemoryRasterOutputProvider,
    MemoryVectorOutputProvider,
)
from itzi_core.simulation_builder import SimulationBuilder

if TYPE_CHECKING:
    from itzi_core.simulation import Simulation


@pytest.fixture(scope="module")
def sim_5by5_max_values(domain_5by5, helpers) -> Simulation:
    """Run a 5x5 simulation for 2s with 1s record step.

    Outputs: water_depth, hmax, v, vmax
    Used for testing that max values are correctly computed.
    """
    # Build SimulationConfig
    sim_config = SimulationConfig(
        start_time=datetime(2000, 1, 1, 0, 0, 0),
        end_time=datetime(2000, 1, 1, 0, 0, 2),  # 2 seconds
        record_step=timedelta(seconds=1),
        temporal_type=TemporalType.RELATIVE,
        input_map_names=helpers.make_input_map_names(
            dem="z",
            friction="n",
            water_depth="start_h",
        ),
        output_map_names=helpers.make_output_map_names(
            "out_5by5_max_values",
            ["water_depth", "hmax", "v", "vmax"],
        ),
        # Same values as 5by5_max_values.ini
        surface_flow_parameters=SurfaceFlowParameters(hmin=0.000001, dtmax=1, cfl=0.8),
        infiltration_model=InfiltrationModelType.NULL,
    )

    # Create output provider
    raster_output = MemoryRasterOutputProvider(sim_config.output_map_names)

    # Build simulation
    simulation = (
        SimulationBuilder(sim_config, domain_5by5.arr_mask, np.float32)
        .with_domain_data(domain_5by5.domain_data)
        .with_raster_output_provider(raster_output)
        .with_vector_output_provider(MemoryVectorOutputProvider())
        .build()
    )

    # Set input arrays
    simulation.set_array("dem", domain_5by5.arr_dem_flat)
    simulation.set_array("friction", domain_5by5.arr_n)
    simulation.set_array("water_depth", domain_5by5.arr_start_h)

    # Run simulation
    simulation.initialize()
    while simulation.sim_time < simulation.end_time:
        simulation.update()
    simulation.finalize()

    return simulation


class TestMaxValues:
    """Test that the maximum values of h and v are properly calculated.

    The simulation tracks hmax and vmax internally and reports them as
    cumulative maximum arrays.
    """

    def test_water_depth_max(self, sim_5by5_max_values):
        """Reported hmax is nondecreasing and ends at the internal maximum."""
        output_dict = sim_5by5_max_values.report.raster_provider.output_maps_dict

        h_max_arrays = [arr for _, arr in output_dict["hmax"]]
        h_arrays = [arr for _, arr in output_dict["water_depth"]]
        h_max_internal = sim_5by5_max_values.get_array("hmax")

        assert len(h_max_arrays) == 3
        np.testing.assert_allclose(h_max_arrays[0], h_arrays[0])
        for previous, current in pairwise(h_max_arrays):
            assert np.all(current >= previous)
        np.testing.assert_allclose(h_max_arrays[-1], h_max_internal)

    def test_velocity_max(self, sim_5by5_max_values):
        """Reported vmax is nondecreasing and ends at the internal maximum."""
        output_dict = sim_5by5_max_values.report.raster_provider.output_maps_dict

        v_max_arrays = [arr for _, arr in output_dict["vmax"]]
        v_arrays = [arr for _, arr in output_dict["v"]]
        v_max_internal = sim_5by5_max_values.get_array("vmax")

        assert len(v_max_arrays) == 3
        np.testing.assert_allclose(v_max_arrays[0], v_arrays[0])
        for previous, current in pairwise(v_max_arrays):
            assert np.all(current >= previous)
        np.testing.assert_allclose(v_max_arrays[-1], v_max_internal)


def test_set_array_synchronizes_maxima(sim_5by5_max_values: Simulation) -> None:
    simulation = sim_5by5_max_values

    larger_depth = simulation.get_array("hmax").copy() + 1.0
    simulation.set_array("water_depth", larger_depth)
    np.testing.assert_allclose(simulation.get_array("hmax"), larger_depth)

    simulation.set_array("water_depth", np.zeros_like(larger_depth))
    np.testing.assert_allclose(simulation.get_array("hmax"), larger_depth)

    larger_wse = simulation.get_array("dem") + larger_depth + 1.0
    simulation.set_array("water_surface_elevation", larger_wse)
    np.testing.assert_allclose(simulation.get_array("hmax"), larger_depth + 1.0)

    larger_speed = simulation.get_array("vmax").copy() + 1.0
    simulation.set_array("v", larger_speed)
    np.testing.assert_allclose(simulation.get_array("vmax"), larger_speed)
