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
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from itzi_core import DomainData
from itzi_core.const import InfiltrationModelType, TemporalType
from itzi_core.data_containers import SimulationConfig, SurfaceFlowParameters
from itzi_core.providers.csv_mass_balance_output import CSVMassBalanceOutputProvider
from itzi_core.providers.memory_output import (
    MemoryRasterOutputProvider,
    MemoryVectorOutputProvider,
)
from itzi_core.simulation_builder import SimulationBuilder

RAIN_RATE = 0.02
SIMULATION_DURATION = timedelta(milliseconds=100)
REGIME_SWITCH_DEPTH = 0.015
REGIME_SWITCH_SLOPE = 0.1
REGIME_SWITCH_DT = 0.1


def _run_steep_plane(max_slope: float, stats_file: Path):
    rows = cols = 9
    cell_size = 1.0
    start_time = datetime(2000, 1, 1, tzinfo=UTC)
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
            slope_threshold=max_slope,
            max_slope=max_slope,
        ),
        dtinf=0.05,
        infiltration_model=InfiltrationModelType.NULL,
    )
    array_mask = np.zeros((rows, cols), dtype=np.bool_)
    raster_output = MemoryRasterOutputProvider(config.output_map_names)
    simulation = (
        SimulationBuilder(config, array_mask, np.float32)
        .with_domain_data(domain_data)
        .with_raster_output_provider(raster_output)
        .with_vector_output_provider(MemoryVectorOutputProvider())
        .with_mass_balance_output_provider(CSVMassBalanceOutputProvider(file_name=str(stats_file)))
        .build()
    )

    # The one-metre drop between columns is steeper than slope_threshold.
    arr_dem = np.tile(-np.arange(cols, dtype=np.float32), (rows, 1))
    simulation.set_array("ground_elevation", arr_dem)
    simulation.set_array("friction", np.full((rows, cols), 0.05, dtype=np.float32))
    simulation.set_array("rainfall_rate", np.full((rows, cols), RAIN_RATE, dtype=np.float32))
    simulation.set_array("boundary_type", np.zeros((rows, cols), dtype=np.float32))

    simulation.initialize()
    while simulation.sim_time < simulation.end_time:
        simulation.update()
    simulation.finalize()
    return simulation


def _last_rainfall_volume(stats_file: Path) -> float:
    with stats_file.open(newline="") as file:
        return float(list(csv.DictReader(file))[-1]["rainfall_volume"])


def _regime_switch_parameters(**overrides) -> SurfaceFlowParameters:
    return SurfaceFlowParameters.model_validate(
        {
            "hmin": 0.005,
            "cfl": 0.1,
            "dtmax": REGIME_SWITCH_DT,
            "theta": 0.7,
            "slope_threshold": 0.2,
            "max_slope": 0.2,
        }
        | overrides
    )


def _build_regime_switch_simulation(
    flow_params: SurfaceFlowParameters,
    hotstart_bytes: bytes | None = None,
    initial_depth: float = REGIME_SWITCH_DEPTH,
    initial_slope: float = REGIME_SWITCH_SLOPE,
):
    rows = cols = 9
    cell_size = 1.0
    start_time = datetime(2000, 1, 1, tzinfo=UTC)
    config = SimulationConfig(
        start_time=start_time,
        end_time=start_time + timedelta(seconds=2),
        record_step=timedelta(seconds=2),
        temporal_type=TemporalType.RELATIVE,
        input_map_names={},
        output_map_names={},
        surface_flow_parameters=flow_params,
        dtinf=1.0,
        infiltration_model=InfiltrationModelType.NULL,
    )
    domain_data = DomainData(
        north=rows * cell_size,
        south=0,
        east=cols * cell_size,
        west=0,
        rows=rows,
        cols=cols,
        crs_wkt="",
    )
    raster_output = MemoryRasterOutputProvider(config.output_map_names)
    builder = (
        SimulationBuilder(config, np.zeros((rows, cols), dtype=np.bool_), np.float32)
        .with_domain_data(domain_data)
        .with_raster_output_provider(raster_output)
        .with_vector_output_provider(MemoryVectorOutputProvider())
    )
    if hotstart_bytes is not None:
        builder.with_hotstart(hotstart_bytes)

    simulation = builder.build()
    if hotstart_bytes is None:
        simulation.set_array(
            "ground_elevation",
            np.tile(-initial_slope * np.arange(cols, dtype=np.float32), (rows, 1)),
        )
        simulation.set_array("friction", np.full((rows, cols), 0.05, dtype=np.float32))
        simulation.set_array("water_depth", np.full((rows, cols), initial_depth, dtype=np.float32))
        simulation.set_array("boundary_type", np.zeros((rows, cols), dtype=np.uint8))
        simulation.initialize()

    return simulation


def _uses_almeida_at_center_east_face(simulation, flow_params: SurfaceFlowParameters) -> bool:
    row = col = 4
    elevation = simulation.get_array("ground_elevation")
    depth = simulation.get_array("water_depth")
    wse = elevation + depth
    flow_depth = max(wse[row, col], wse[row, col + 1]) - max(
        elevation[row, col], elevation[row, col + 1]
    )
    slope = (wse[row, col] - wse[row, col + 1]) / simulation.surface_flow.dx
    return flow_depth > flow_params.hmin and abs(slope) < flow_params.slope_threshold


def _assert_hotstart_regime_switch_matches_continuous(
    checkpoint_params: SurfaceFlowParameters,
    resumed_params: SurfaceFlowParameters,
    initial_depth: float = REGIME_SWITCH_DEPTH,
    initial_slope: float = REGIME_SWITCH_SLOPE,
) -> None:
    continuous = _build_regime_switch_simulation(
        checkpoint_params,
        initial_depth=initial_depth,
        initial_slope=initial_slope,
    )

    # Establish a non-zero discharge history before serialising the state.
    for _ in range(2):
        continuous.update()

    assert continuous.get_array("discharge_east")[4, 4] > 0
    assert _uses_almeida_at_center_east_face(
        continuous, checkpoint_params
    ) != _uses_almeida_at_center_east_face(continuous, resumed_params)

    checkpoint_error = continuous.get_array("error_depth_accum").copy()
    hotstart_bytes = continuous.create_hotstart().getvalue()
    control = _build_regime_switch_simulation(checkpoint_params, hotstart_bytes)
    resumed = _build_regime_switch_simulation(resumed_params, hotstart_bytes)

    # The continuous path is the reference for a configuration change at this state.
    for attribute, checkpoint_value, resumed_value in [
        ("min_flow_depth", checkpoint_params.hmin, resumed_params.hmin),
        ("slope_threshold", checkpoint_params.slope_threshold, resumed_params.slope_threshold),
        ("max_slope", checkpoint_params.max_slope, resumed_params.max_slope),
    ]:
        if checkpoint_value != resumed_value:
            setattr(continuous.surface_flow, attribute, resumed_value)

    for step in range(1, 11):
        continuous.update()
        control.update()
        resumed.update()

        if step not in {1, 10}:
            continue

        for array_key in [
            "water_depth",
            "flow_depth_east",
            "flow_depth_south",
            "discharge_east",
            "discharge_south",
            "error_depth_accum",
        ]:
            np.testing.assert_allclose(
                resumed.get_array(array_key),
                continuous.get_array(array_key),
                err_msg=f"Hotstart switch mismatch for {array_key} at step {step}",
            )

        for simulation in [continuous, resumed]:
            for array_key in ["water_depth", "discharge_east", "discharge_south"]:
                assert np.all(np.isfinite(simulation.get_array(array_key)))

        np.testing.assert_allclose(continuous.get_array("error_depth_accum"), checkpoint_error)
        np.testing.assert_allclose(resumed.get_array("error_depth_accum"), checkpoint_error)

        if step == 1:
            assert not np.allclose(
                continuous.get_array("discharge_east"), control.get_array("discharge_east")
            )


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
        eastward_flow = simulation.get_array("discharge_east")
        southward_flow = simulation.get_array("discharge_south")

        assert simulation.sim_time == simulation.end_time
        assert np.all(np.isfinite(water_depth))
        assert np.all(np.isfinite(eastward_flow))
        assert np.all(np.isfinite(southward_flow))
        assert np.min(water_depth) >= 0
        assert np.sum(water_depth) == pytest.approx(expected_rainfall_volume, rel=1e-5)
        assert _last_rainfall_volume(stats_file) == pytest.approx(expected_rainfall_volume)

    # Both runs solve the same first wet face, so the GMS flux differs only by sqrt(max_slope).
    center = (4, 4)
    low_flux = low_cap.get_array("discharge_east")[center]
    high_flux = high_cap.get_array("discharge_east")[center]
    assert low_flux > 0
    assert high_flux / low_flux == pytest.approx(np.sqrt(0.8 / 0.2), rel=1e-5)


@pytest.mark.parametrize(
    ("parameter", "checkpoint_value", "resumed_value"),
    [
        ("hmin", 0.005, 0.025),
        ("hmin", 0.025, 0.005),
        ("slope_threshold", 0.2, 0.05),
        ("slope_threshold", 0.05, 0.2),
    ],
)
def test_hotstart_regime_switch_matches_continuous_switch(
    parameter: str,
    checkpoint_value: float,
    resumed_value: float,
):
    """Switching between Almeida and GMS must not add a hotstart transient."""
    checkpoint_params = _regime_switch_parameters(**{parameter: checkpoint_value})
    resumed_params = _regime_switch_parameters(
        **(checkpoint_params.model_dump() | {parameter: resumed_value})
    )

    _assert_hotstart_regime_switch_matches_continuous(checkpoint_params, resumed_params)


def test_hotstart_shallow_face_switch_to_almeida_matches_continuous_switch():
    """A near-dry GMS face may resume with Almeida without a restart transient."""
    checkpoint_params = _regime_switch_parameters(hmin=0.005)
    resumed_params = _regime_switch_parameters(hmin=0.0005)

    _assert_hotstart_regime_switch_matches_continuous(
        checkpoint_params,
        resumed_params,
        initial_depth=0.001,
    )


def test_hotstart_switch_above_prior_gms_slope_cap_matches_continuous_switch():
    """A capped GMS face may resume with the uncapped Almeida update."""
    checkpoint_params = _regime_switch_parameters(slope_threshold=0.8, max_slope=0.8)
    resumed_params = _regime_switch_parameters(slope_threshold=0.95, max_slope=0.95)

    _assert_hotstart_regime_switch_matches_continuous(
        checkpoint_params,
        resumed_params,
        initial_slope=0.9,
    )
