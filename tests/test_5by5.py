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
from typing import TYPE_CHECKING

import numpy as np
import pytest

from itzi_core.const import InfiltrationModelType, TemporalType
from itzi_core.data_containers import SimulationConfig, SurfaceFlowParameters
from itzi_core.providers.domain_data import DomainData
from itzi_core.providers.memory_output import (
    MemoryRasterOutputProvider,
    MemoryVectorOutputProvider,
)
from itzi_core.simulation_builder import SimulationBuilder

if TYPE_CHECKING:
    from itzi_core.simulation import Simulation


def _build_diagnostic_simulation(
    domain_5by5,
    helpers,
    output_keys: list[str],
    *,
    end_seconds: float,
    record_seconds: float,
    dtmax: float,
    initial_depth: np.ndarray | None = None,
) -> Simulation:
    start_time = datetime(2000, 1, 1)
    sim_config = SimulationConfig(
        start_time=start_time,
        end_time=start_time + timedelta(seconds=end_seconds),
        record_step=timedelta(seconds=record_seconds),
        temporal_type=TemporalType.RELATIVE,
        input_map_names=helpers.make_input_map_names(
            dem="z",
            friction="n",
            water_depth="start_h",
        ),
        output_map_names=helpers.make_output_map_names("diagnostics", output_keys),
        surface_flow_parameters=SurfaceFlowParameters(hmin=0.0001, dtmax=dtmax, cfl=0.2),
        infiltration_model=InfiltrationModelType.NULL,
    )
    raster_output = MemoryRasterOutputProvider(sim_config.output_map_names)
    simulation = (
        SimulationBuilder(sim_config, domain_5by5.arr_mask, np.float32)
        .with_domain_data(domain_5by5.domain_data)
        .with_raster_output_provider(raster_output)
        .with_vector_output_provider(MemoryVectorOutputProvider())
        .build()
    )
    simulation.set_array("dem", domain_5by5.arr_dem_flat.copy())
    simulation.set_array("friction", domain_5by5.arr_n.copy())
    simulation.set_array(
        "water_depth",
        np.zeros_like(domain_5by5.arr_start_h) if initial_depth is None else initial_depth.copy(),
    )
    return simulation


def _run_diagnostic_regression(domain_5by5, helpers, *, force_all: bool):
    simulation = _build_diagnostic_simulation(
        domain_5by5,
        helpers,
        ["water_depth", "v", "vmax", "vdir", "froude"],
        end_seconds=1.0,
        record_seconds=0.4,
        dtmax=0.3,
        initial_depth=domain_5by5.arr_start_h,
    )
    if force_all:
        scheduled_step = simulation.surface_flow.step

        def always_compute_step(*, compute_vdir: bool, compute_froude: bool):
            return scheduled_step(compute_vdir=True, compute_froude=True)

        simulation.surface_flow.step = always_compute_step

    simulation.initialize()
    while simulation.sim_time < simulation.end_time:
        simulation.update()

    output_maps = simulation.report.raster_provider.output_maps_dict
    result = {
        "outputs": {
            key: [(time, array.copy()) for time, array in output_maps[key]]
            for key in ("vdir", "froude")
        },
        "water_depth": simulation.get_array("water_depth").copy(),
        "v": simulation.get_array("v").copy(),
        "vmax": simulation.get_array("vmax").copy(),
        "steps": simulation.time_steps_counters["since_start"],
    }
    simulation.finalize()
    return result


def _run_center_pulse_simulation(
    domain_5by5,
    helpers,
    *,
    dx: float,
    dy: float,
    duration_s: float = 1.0,
) -> np.ndarray:
    rows, cols = domain_5by5.domain_data.shape
    domain_data = DomainData(
        north=rows * dy,
        south=0.0,
        east=cols * dx,
        west=0.0,
        rows=rows,
        cols=cols,
        crs_wkt="",
    )

    initial_volume = float(np.sum(domain_5by5.arr_start_h) * domain_5by5.domain_data.cell_area)
    arr_start_h = np.zeros(domain_data.shape, dtype=np.float32)
    arr_start_h[2, 2] = initial_volume / domain_data.cell_area

    sim_config = SimulationConfig(
        start_time=datetime(2000, 1, 1, 0, 0, 0),
        end_time=datetime(2000, 1, 1, 0, 0, 0) + timedelta(seconds=duration_s),
        record_step=timedelta(seconds=duration_s),
        temporal_type=TemporalType.RELATIVE,
        input_map_names=helpers.make_input_map_names(
            dem="z",
            friction="n",
            water_depth="start_h",
        ),
        output_map_names=helpers.make_output_map_names(
            f"out_5by5_rect_{int(dx)}x{int(dy)}",
            ["water_depth"],
        ),
        surface_flow_parameters=SurfaceFlowParameters(hmin=0.0001, dtmax=0.3, cfl=0.2),
        infiltration_model=InfiltrationModelType.NULL,
    )

    raster_output = MemoryRasterOutputProvider(sim_config.output_map_names)
    simulation = (
        SimulationBuilder(sim_config, domain_5by5.arr_mask, np.float32)
        .with_domain_data(domain_data)
        .with_raster_output_provider(raster_output)
        .with_vector_output_provider(MemoryVectorOutputProvider())
        .build()
    )

    simulation.set_array("dem", domain_5by5.arr_dem_flat.copy())
    simulation.set_array("friction", domain_5by5.arr_n.copy())
    simulation.set_array("water_depth", arr_start_h.copy())

    simulation.initialize()
    while simulation.sim_time < simulation.end_time:
        simulation.update()
    final_depth = simulation.get_array("water_depth").copy()
    simulation.finalize()
    return final_depth


@pytest.mark.parametrize(
    ("diagnostic_keys", "report_flags"),
    [
        ([], (False, False)),
        (["vdir"], (True, False)),
        (["froude"], (False, True)),
        (["vdir", "froude"], (True, True)),
    ],
    ids=["neither", "vdir", "froude", "both"],
)
def test_scheduler_computes_only_requested_report_diagnostics(
    domain_5by5,
    helpers,
    diagnostic_keys: list[str],
    report_flags: tuple[bool, bool],
):
    """Non-report steps retain diagnostics, including before an off-cadence final report."""
    simulation = _build_diagnostic_simulation(
        domain_5by5,
        helpers,
        ["water_depth", *diagnostic_keys],
        end_seconds=10.0,
        record_seconds=4.0,
        dtmax=3.0,
    )
    scheduled_step = simulation.surface_flow.step
    calls = []

    def tracked_step(*, compute_vdir: bool, compute_froude: bool):
        step_end = simulation.sim_time + simulation.dt
        vdir_before = simulation.get_array("vdir").copy()
        froude_before = simulation.get_array("froude").copy()
        result = scheduled_step(
            compute_vdir=compute_vdir,
            compute_froude=compute_froude,
        )
        if not compute_vdir:
            np.testing.assert_array_equal(simulation.get_array("vdir"), vdir_before)
        if not compute_froude:
            np.testing.assert_array_equal(simulation.get_array("froude"), froude_before)
        calls.append(
            (
                step_end - simulation.start_time,
                (compute_vdir, compute_froude),
            )
        )
        return result

    simulation.surface_flow.step = tracked_step
    simulation.initialize()
    simulation.get_array("vdir").fill(-123.0)
    simulation.get_array("froude").fill(-456.0)
    while simulation.sim_time < simulation.end_time:
        simulation.update()
    simulation.finalize()

    assert calls == [
        (timedelta(seconds=3), (False, False)),
        (timedelta(seconds=4), report_flags),
        (timedelta(seconds=7), (False, False)),
        (timedelta(seconds=8), report_flags),
        (timedelta(seconds=10), report_flags),
    ]

    output_maps = simulation.report.raster_provider.output_maps_dict
    expected_report_times = [
        timedelta(seconds=0),
        timedelta(seconds=4),
        timedelta(seconds=8),
        timedelta(seconds=10),
    ]
    assert [time for time, _ in output_maps["water_depth"]] == expected_report_times
    for key in ("vdir", "froude"):
        expected_times = expected_report_times if key in diagnostic_keys else []
        assert [time for time, _ in output_maps[key]] == expected_times


def test_lazy_diagnostic_reports_match_always_compute_reference(domain_5by5, helpers):
    optimized = _run_diagnostic_regression(domain_5by5, helpers, force_all=False)
    reference = _run_diagnostic_regression(domain_5by5, helpers, force_all=True)

    assert optimized["steps"] == reference["steps"]
    for key in ("water_depth", "v", "vmax"):
        np.testing.assert_allclose(optimized[key], reference[key], rtol=1e-6, atol=1e-7)
    for key in ("vdir", "froude"):
        optimized_outputs = optimized["outputs"][key]
        reference_outputs = reference["outputs"][key]
        assert [time for time, _ in optimized_outputs] == [time for time, _ in reference_outputs]
        for (_, optimized_array), (_, reference_array) in zip(
            optimized_outputs, reference_outputs, strict=True
        ):
            np.testing.assert_allclose(optimized_array, reference_array, rtol=1e-6, atol=1e-7)


@pytest.fixture(scope="module")
def sim_5by5(domain_5by5, helpers) -> Simulation:
    """Run a 5x5 simulation for 60s with 30s record step.

    Outputs: water_depth, water_surface_elevation, froude, v, vdir, qx, qy, created_volume
    """
    # Build SimulationConfig
    sim_config = SimulationConfig(
        start_time=datetime(2000, 1, 1, 0, 0, 0),
        end_time=datetime(2000, 1, 1, 0, 1, 0),  # 60 seconds
        record_step=timedelta(seconds=30),
        temporal_type=TemporalType.RELATIVE,
        input_map_names=helpers.make_input_map_names(
            dem="z",
            friction="n",
            water_depth="start_h",
        ),
        output_map_names=helpers.make_output_map_names(
            "out_5by5",
            [
                "water_depth",
                "water_surface_elevation",
                "froude",
                "v",
                "vdir",
                "qx",
                "qy",
                "created_volume",
            ],
        ),
        # Same values as original 5by5.ini
        surface_flow_parameters=SurfaceFlowParameters(hmin=0.0001, dtmax=0.3, cfl=0.2),
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


class TestNumberOfOutput:
    """Test that the correct number of output maps are produced.
    60 seconds test with 30s record steps.
    Memory provider outputs: initial (t=0) + 2 record steps (t=30, t=60) = 3 outputs.
    Note: Unlike GRASS provider, memory provider does not generate separate max maps."""

    def test_water_depth_count(self, sim_5by5):
        """water_depth should have 3 outputs (initial + 2 record steps)."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict
        assert len(output_dict["water_depth"]) == 3

    def test_water_surface_elevation_count(self, sim_5by5):
        """water_surface_elevation should have 3 outputs."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict
        assert len(output_dict["water_surface_elevation"]) == 3

    def test_froude_count(self, sim_5by5):
        """froude should have 3 outputs."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict
        assert len(output_dict["froude"]) == 3

    def test_v_count(self, sim_5by5):
        """v (velocity) should have 3 outputs (initial + 2 record steps)."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict
        assert len(output_dict["v"]) == 3

    def test_vdir_count(self, sim_5by5):
        """vdir (velocity direction) should have 3 outputs."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict
        assert len(output_dict["vdir"]) == 3

    def test_qx_count(self, sim_5by5):
        """qx should have 3 outputs."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict
        assert len(output_dict["qx"]) == 3

    def test_qy_count(self, sim_5by5):
        """qy should have 3 outputs."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict
        assert len(output_dict["qy"]) == 3

    def test_created_volume_count(self, sim_5by5):
        """created_volume should have 3 outputs."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict
        assert len(output_dict["created_volume"]) == 3


class TestFlowSymmetry:
    """Test that water depths at 4 symmetric control points around center are equal.

    On a 5x5 grid, the center is [2, 2]. The 4 symmetric neighbours are:
    - [1, 2] (above center)
    - [3, 2] (below center)
    - [2, 1] (left of center)
    - [2, 3] (right of center)

    After a dam-break, flow should spread symmetrically in all directions.
    """

    def test_flow_symmetry(self, sim_5by5):
        """Water depths at symmetric points should be equal after dam-break."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict

        # Get the last water_depth output (index -1 or 2)
        # The GRASS test uses water_depth_0002 which is the last time step
        _, h_array = output_dict["water_depth"][-1]

        # Sample at the 4 symmetric control points around center
        # Center is at [2, 2], neighbours are at [1,2], [3,2], [2,1], [2,3]
        h_above = h_array[1, 2]  # row 1, col 2
        h_below = h_array[3, 2]  # row 3, col 2
        h_left = h_array[2, 1]  # row 2, col 1
        h_right = h_array[2, 3]  # row 2, col 3

        # All 4 values should be approximately equal due to symmetry
        values = [h_above, h_below, h_left, h_right]
        assert np.allclose(values[:-1], values[1:]), (
            f"Symmetric points should have equal depths: "
            f"above={h_above:.6f}, below={h_below:.6f}, "
            f"left={h_left:.6f}, right={h_right:.6f}"
        )


@pytest.mark.parametrize(
    ("dx", "dy", "wetter_axis"),
    [
        (20.0, 10.0, "y"),
        (10.0, 20.0, "x"),
    ],
)
def test_rectangular_cells_preserve_axis_symmetry_and_bias_early_spreading(
    domain_5by5, helpers, dx: float, dy: float, wetter_axis: str
):
    """Early-time spreading should stay symmetric within each axis and favor the shorter cell size."""
    h_array = _run_center_pulse_simulation(domain_5by5, helpers, dx=dx, dy=dy)

    h_above = h_array[1, 2]
    h_below = h_array[3, 2]
    h_left = h_array[2, 1]
    h_right = h_array[2, 3]

    assert np.isclose(h_above, h_below)
    assert np.isclose(h_left, h_right)

    if wetter_axis == "y":
        assert min(h_above, h_below) > max(h_left, h_right)
    else:
        assert min(h_left, h_right) > max(h_above, h_below)


def test_swapping_dx_and_dy_transposes_the_early_time_solution(domain_5by5, helpers):
    """Swapping the cell sizes should swap the x/y spreading pattern."""
    h_dx20_dy10 = _run_center_pulse_simulation(domain_5by5, helpers, dx=20.0, dy=10.0)
    h_dx10_dy20 = _run_center_pulse_simulation(domain_5by5, helpers, dx=10.0, dy=20.0)

    assert np.allclose(h_dx20_dy10, h_dx10_dy20.T)
