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

import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from itzi_core.const import InfiltrationModelType, TemporalType
from itzi_core.data_containers import SimulationConfig, SurfaceFlowParameters
from itzi_core.providers.csv_mass_balance_output import CSVMassBalanceOutputProvider
from itzi_core.providers.domain_data import DomainData
from itzi_core.providers.memory_output import (
    MemoryRasterOutputProvider,
    MemoryVectorOutputProvider,
)
from itzi_core.simulation_builder import SimulationBuilder

RAIN_RATE = 0.02
SIMULATION_DURATION = timedelta(milliseconds=100)


def _run_steep_plane(max_slope: float, stats_file: Path):
    rows = cols = 9
    cell_size = 1.0
    start_time = datetime(2000, 1, 1)
    end_time = start_time + SIMULATION_DURATION
    domain_data = DomainData(
        north=rows * cell_size,
        south=0,
        east=cols * cell_size,
        west=0,
        rows=rows,
        cols=cols,
        crs_wkt="",
    )
    config = SimulationConfig(
        start_time=start_time,
        end_time=end_time,
        record_step=SIMULATION_DURATION,
        temporal_type=TemporalType.RELATIVE,
        input_map_names={},
        output_map_names={},
        surface_flow_parameters=SurfaceFlowParameters(
            hmin=0.005,
            cfl=0.5,
            dtmax=0.05,
            slope_threshold=0.8,
            max_slope=max_slope,
        ),
        dtinf=0.05,
        infiltration_model=InfiltrationModelType.NULL,
    )
    array_mask = np.zeros((rows, cols), dtype=np.bool_)
    raster_output = MemoryRasterOutputProvider({"out_map_names": config.output_map_names})
    simulation = (
        SimulationBuilder(config, array_mask, np.float32)
        .with_domain_data(domain_data)
        .with_raster_output_provider(raster_output)
        .with_vector_output_provider(MemoryVectorOutputProvider({}))
        .with_mass_balance_output_provider(CSVMassBalanceOutputProvider(file_name=str(stats_file)))
        .build()
    )

    # The one-metre drop between columns is steeper than slope_threshold.
    arr_dem = np.tile(-np.arange(cols, dtype=np.float32), (rows, 1))
    simulation.set_array("dem", arr_dem)
    simulation.set_array("friction", np.full((rows, cols), 0.05, dtype=np.float32))
    simulation.set_array("rain", np.full((rows, cols), RAIN_RATE, dtype=np.float32))
    simulation.set_array("bctype", np.zeros((rows, cols), dtype=np.float32))

    simulation.initialize()
    while simulation.sim_time < simulation.end_time:
        simulation.update()
    simulation.finalize()
    return simulation


def _last_rainfall_volume(stats_file: Path) -> float:
    with stats_file.open(newline="") as file:
        return float(list(csv.DictReader(file))[-1]["rainfall_volume"])


def test_rain_on_steep_plane_uses_capped_downhill_flow(tmp_path):
    """Rainfall activates the high-slope GMS branch without creating invalid state."""
    low_cap = _run_steep_plane(max_slope=0.2, stats_file=tmp_path / "low_cap.csv")
    high_cap = _run_steep_plane(max_slope=0.8, stats_file=tmp_path / "high_cap.csv")
    expected_rainfall_volume = RAIN_RATE * 9 * 9 * SIMULATION_DURATION.total_seconds()

    for simulation, stats_file in [
        (low_cap, tmp_path / "low_cap.csv"),
        (high_cap, tmp_path / "high_cap.csv"),
    ]:
        water_depth = simulation.get_array("water_depth")
        eastward_flow = simulation.get_array("qe")
        southward_flow = simulation.get_array("qs")

        assert simulation.sim_time == simulation.end_time
        assert np.all(np.isfinite(water_depth))
        assert np.all(np.isfinite(eastward_flow))
        assert np.all(np.isfinite(southward_flow))
        assert np.min(water_depth) >= 0
        assert np.sum(water_depth) == pytest.approx(expected_rainfall_volume, rel=1e-5)
        assert _last_rainfall_volume(stats_file) == pytest.approx(expected_rainfall_volume)

    # Both runs solve the same first wet face, so the GMS flux differs only by sqrt(max_slope).
    center = (4, 4)
    low_flux = low_cap.get_array("qe")[center]
    high_flux = high_cap.get_array("qe")[center]
    assert low_flux > 0
    assert high_flux / low_flux == pytest.approx(np.sqrt(0.8 / 0.2), rel=1e-5)
