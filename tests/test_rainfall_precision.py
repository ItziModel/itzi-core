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

import numpy as np
import pytest
from itzi_core.compute.hydrology import apply_hydrology, infiltration_user

from itzi_core.compute import rastermetrics
from itzi_core.const import InfiltrationModelType, TemporalType
from itzi_core.data_containers import MassBalanceData, SimulationConfig, SurfaceFlowParameters
from itzi_core.providers.base import MassBalanceOutputProvider
from itzi_core.providers.domain_data import DomainData
from itzi_core.providers.memory_output import (
    MemoryRasterOutputProvider,
    MemoryVectorOutputProvider,
)
from itzi_core.simulation_builder import SimulationBuilder

RAIN_RATE = 10.0 / (1000 * 3600)


class CaptureMassBalanceOutputProvider(MassBalanceOutputProvider):
    def __init__(self) -> None:
        self.reports: list[MassBalanceData] = []

    def log(self, report_data: MassBalanceData) -> None:
        self.reports.append(report_data)


def test_float32_rain_rate_representation_error_is_negligible():
    """Storing 10 mm/h as float32 should not cause a meaningful daily error."""
    stored_rate = np.float32(RAIN_RATE)
    daily_depth = float(stored_rate) * 24 * 3600
    exact_daily_depth = 0.01 * 24

    assert daily_depth == pytest.approx(exact_daily_depth, abs=2e-8)


@pytest.mark.parametrize(
    ("dtype", "expected_outcome"),
    [
        pytest.param(np.float32, "xfail", marks=pytest.mark.xfail(strict=True)),
        pytest.param(np.float64, "pass"),
    ],
)
def test_repeated_rainfall_accumulation_matches_analytical_depth(dtype, expected_outcome):
    """Accumulating many short rainfall intervals should conserve their total depth."""
    del expected_outcome
    steps = 100_000
    dt = 1.0
    rate = np.full((1, 1), RAIN_RATE, dtype=dtype)
    accumulated = np.zeros_like(rate)

    for _ in range(steps):
        rastermetrics.accumulate_rate_to_total(accumulated, rate, dt)

    expected_depth = RAIN_RATE * steps * dt
    assert float(accumulated[0, 0]) == pytest.approx(expected_depth, rel=1e-9)


@pytest.mark.parametrize(
    ("dtype", "shape"),
    [
        pytest.param(np.float32, (1000, 1000)),
        pytest.param(np.float32, (1001, 1000)),
        pytest.param(np.float64, (1000, 1000)),
        pytest.param(np.float64, (1001, 1000)),
    ],
)
def test_total_volume_matches_float64_reference(dtype, shape):
    """Mass-balance reductions should be accurate on real-world grid sizes."""
    depths = np.full(shape, 0.01, dtype=dtype)
    expected_volume = float(np.sum(depths, dtype=np.float64))

    volume = float(rastermetrics.calculate_total_volume(depths, 1.0))

    assert volume == pytest.approx(expected_volume, rel=1e-10)


def _run_rain_only_simulation(
    dtype: type[np.floating],
    *,
    initial_depth: float = 1.0,
    timestep: float = 0.01,
    duration_seconds: float = 1.0,
) -> MassBalanceData:
    start_time = datetime(2000, 1, 1, tzinfo=UTC)
    duration = timedelta(seconds=duration_seconds)
    shape = (1, 1)
    domain_data = DomainData(
        north=1,
        south=0,
        east=1,
        west=0,
        rows=shape[0],
        cols=shape[1],
        crs_wkt="",
    )
    config = SimulationConfig(
        start_time=start_time,
        end_time=start_time + duration,
        record_step=duration,
        temporal_type=TemporalType.RELATIVE,
        input_map_names={},
        output_map_names={},
        surface_flow_parameters=SurfaceFlowParameters(dtmax=timestep),
        dtinf=timestep,
        infiltration_model=InfiltrationModelType.NULL,
    )
    mass_balance_output = CaptureMassBalanceOutputProvider()
    simulation = (
        SimulationBuilder(config, np.zeros(shape, dtype=np.bool_), dtype)
        .with_domain_data(domain_data)
        .with_raster_output_provider(MemoryRasterOutputProvider(config.output_map_names))
        .with_vector_output_provider(MemoryVectorOutputProvider())
        .with_mass_balance_output_provider(mass_balance_output)
        .build()
    )
    simulation.set_array("dem", np.zeros(shape, dtype=dtype))
    simulation.set_array("friction", np.full(shape, 0.03, dtype=dtype))
    simulation.set_array("water_depth", np.full(shape, initial_depth, dtype=dtype))
    simulation.set_array("rain", np.full(shape, RAIN_RATE, dtype=dtype))

    simulation.initialize()
    while simulation.sim_time < simulation.end_time:
        simulation.update()
    simulation.finalize()

    return mass_balance_output.reports[-1]


def _run_combined_removal_simulation(
    infiltration_model: InfiltrationModelType,
) -> tuple[MassBalanceData, dict[str, list], float]:
    start_time = datetime(2000, 1, 1, tzinfo=UTC)
    duration = timedelta(seconds=1000)
    shape = (1, 1)
    domain_data = DomainData(
        north=1,
        south=0,
        east=1,
        west=0,
        rows=1,
        cols=1,
        crs_wkt="",
    )
    output_map_names = {
        "mean_infiltration": "mean_infiltration",
        "mean_losses": "mean_losses",
    }
    config = SimulationConfig(
        start_time=start_time,
        end_time=start_time + duration,
        record_step=duration,
        temporal_type=TemporalType.RELATIVE,
        input_map_names={},
        output_map_names=output_map_names,
        surface_flow_parameters=SurfaceFlowParameters(dtmax=1000.0),
        dtinf=1000.0,
        infiltration_model=infiltration_model,
    )
    mass_balance_output = CaptureMassBalanceOutputProvider()
    raster_output = MemoryRasterOutputProvider(output_map_names)
    dtype = np.float32
    simulation = (
        SimulationBuilder(config, np.zeros(shape, dtype=np.bool_), dtype)
        .with_domain_data(domain_data)
        .with_raster_output_provider(raster_output)
        .with_vector_output_provider(MemoryVectorOutputProvider())
        .with_mass_balance_output_provider(mass_balance_output)
        .build()
    )
    simulation.set_array("dem", np.zeros(shape, dtype=dtype))
    simulation.set_array("friction", np.full(shape, 0.03, dtype=dtype))
    simulation.set_array("water_depth", np.full(shape, 0.5, dtype=dtype))
    simulation.set_array("losses", np.full(shape, 0.00025, dtype=dtype))
    if infiltration_model == InfiltrationModelType.CONSTANT:
        simulation.set_array("infiltration", np.full(shape, 0.00075, dtype=dtype))
    else:
        simulation.set_array("effective_porosity", np.full(shape, 0.3, dtype=dtype))
        simulation.set_array("soil_water_content", np.full(shape, 0.3, dtype=dtype))
        simulation.set_array("hydraulic_conductivity", np.full(shape, 0.00075, dtype=dtype))
        simulation.set_array("capillary_pressure", np.full(shape, 0.5, dtype=dtype))
        simulation.set_array("total_infiltration", np.full(shape, 0.1, dtype=dtype))

    simulation.initialize()
    while simulation.sim_time < simulation.end_time:
        simulation.update()
    simulation.finalize()

    total_infiltration = float(simulation.get_array("total_infiltration")[0, 0])
    return mass_balance_output.reports[-1], raster_output.output_maps_dict, total_infiltration


@pytest.mark.parametrize(
    ("dtype", "initial_depth", "timestep"),
    [
        pytest.param(
            np.float32,
            1.0,
            0.01,
            marks=pytest.mark.xfail(
                strict=True,
                reason="sub-ULP rainfall increments are lost when added to one metre of depth",
            ),
        ),
        pytest.param(
            np.float32,
            0.1,
            1.0,
            marks=pytest.mark.xfail(
                strict=True,
                reason="rainfall is quantized when added to existing float32 depth",
            ),
        ),
        pytest.param(
            np.float32,
            1.0,
            1.0,
            marks=pytest.mark.xfail(
                strict=True,
                reason="rainfall is quantized when added to existing float32 depth",
            ),
        ),
        pytest.param(np.float32, 0.0, 0.01),
        pytest.param(np.float64, 1.0, 0.01),
    ],
)
def test_rain_only_simulation_closes_mass_balance(dtype, initial_depth, timestep):
    """A closed, uniform, rain-only domain should have exact analytical closure."""
    report = _run_rain_only_simulation(
        dtype,
        initial_depth=initial_depth,
        timestep=timestep,
    )
    expected_volume = RAIN_RATE

    assert report.rainfall_volume == pytest.approx(expected_volume, rel=1e-9)
    assert report.volume_change == pytest.approx(expected_volume, rel=1e-9)
    assert report.volume_change == pytest.approx(report.rainfall_volume, rel=1e-9)


def test_closure_error_detects_rainfall_residual_without_changing_percent_error():
    """The reported error should expose a rainfall-versus-volume discrepancy."""
    report = _run_rain_only_simulation(np.float32, timestep=1.0)
    residual = report.volume_change - report.rainfall_volume
    normalizer = max(abs(report.volume_change), abs(report.rainfall_volume))
    expected_error = abs(residual) / normalizer

    assert report.volume_error == 0.0
    assert report.percent_error == 0.0
    assert report.closure_residual == pytest.approx(residual)
    assert report.closure_residual != 0.0
    assert report.closure_error == pytest.approx(expected_error)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    ("water_depth", "expected_infiltration", "expected_losses", "expected_effective"),
    [
        (0.0, 0.0, 0.0, 0.2),
        (0.5, 0.375, 0.125, -0.3),
        (1.0, 0.75, 0.25, -0.8),
        (2.0, 0.75, 0.25, -0.8),
    ],
)
def test_combined_infiltration_and_losses_use_proportional_applied_rates(
    dtype, water_depth, expected_infiltration, expected_losses, expected_effective
):
    """Removal arrays should contain rates applied using start-of-event water."""
    dt = 1.0
    depth = np.full((1, 1), water_depth, dtype=dtype)
    infiltration = np.zeros((1, 1), dtype=dtype)
    losses = np.zeros((1, 1), dtype=dtype)
    effective_precipitation = np.zeros((1, 1), dtype=dtype)

    infiltration_user(depth, np.full((1, 1), 0.75, dtype=dtype), infiltration, dt)
    infiltration_user(depth, np.full((1, 1), 0.25, dtype=dtype), losses, dt)
    apply_hydrology(
        np.full((1, 1), 0.2, dtype=dtype),
        infiltration,
        losses,
        depth,
        effective_precipitation,
        dt,
    )

    assert infiltration[0, 0] == pytest.approx(expected_infiltration)
    assert losses[0, 0] == pytest.approx(expected_losses)
    assert effective_precipitation[0, 0] == pytest.approx(expected_effective)
    assert infiltration[0, 0] + losses[0, 0] <= water_depth / dt + np.finfo(dtype).eps
    assert effective_precipitation[0, 0] == pytest.approx(0.2 - infiltration[0, 0] - losses[0, 0])


def test_combined_removal_cap_handles_zero_candidates():
    zeros = np.zeros((1, 1), dtype=np.float64)
    effective_precipitation = np.zeros_like(zeros)

    apply_hydrology(zeros, zeros.copy(), zeros.copy(), zeros, effective_precipitation, 1.0)

    assert effective_precipitation[0, 0] == 0.0


@pytest.mark.parametrize(
    "infiltration_model",
    [InfiltrationModelType.CONSTANT, InfiltrationModelType.GREEN_AMPT],
)
def test_combined_removal_reports_and_maps_use_applied_rates(infiltration_model):
    report, output_maps, total_infiltration = _run_combined_removal_simulation(infiltration_model)

    assert report.infiltration_volume == pytest.approx(-0.375)
    assert report.losses_volume == pytest.approx(-0.125)
    assert report.volume_change == pytest.approx(-0.5)
    assert report.closure_residual == pytest.approx(0.0, abs=1e-6)
    assert report.closure_error < 0.001
    assert output_maps["mean_infiltration"][-1][1][0, 0] == pytest.approx(1350.0)
    assert output_maps["mean_losses"][-1][1][0, 0] == pytest.approx(450.0)
    if infiltration_model == InfiltrationModelType.GREEN_AMPT:
        assert total_infiltration == pytest.approx(0.475)
