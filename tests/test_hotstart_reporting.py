"""Hotstart reporting intervals survive clipped deadlines and cadence changes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest

from itzi_core import HotstartData, Simulation, SimulationBuilder, SimulationConfig
from itzi_core.const import InfiltrationModelType, TemporalType
from itzi_core.data_containers import SurfaceFlowParameters
from itzi_core.providers.memory_output import (
    MemoryMassBalanceOutputProvider,
    MemoryRasterOutputProvider,
)

if TYPE_CHECKING:
    from tests.conftest import Domain5by5Data


@pytest.fixture(params=[TemporalType.RELATIVE, TemporalType.ABSOLUTE])
def sim_config(request: pytest.FixtureRequest) -> SimulationConfig:
    start_time = datetime(2000, 1, 1, tzinfo=UTC)
    return SimulationConfig(
        start_time=start_time,
        end_time=start_time + timedelta(seconds=25),
        record_step=timedelta(seconds=10),
        temporal_type=request.param,
        dtinf=1.0,
        infiltration_model=InfiltrationModelType.CONSTANT,
        input_map_names={
            "ground_elevation": "dem",
            "friction": "n",
            "water_depth": "depth",
            "rainfall_rate": "rain",
            "infiltration": "infiltration",
            "losses": "losses",
            "inflow": "inflow",
        },
        output_map_names={
            key: key
            for key in (
                "water_depth",
                "mean_rainfall",
                "mean_infiltration",
                "mean_losses",
                "mean_inflow",
            )
        },
        surface_flow_parameters=SurfaceFlowParameters(dtmax=1.0),
    )


def build_simulation(
    config: SimulationConfig,
    domain: Domain5by5Data,
    hotstart: HotstartData | None = None,
) -> Simulation:
    builder = (
        SimulationBuilder(config, domain.arr_mask)
        .with_domain_data(domain.domain_data)
        .with_raster_output_provider(MemoryRasterOutputProvider())
        .with_mass_balance_output_provider(MemoryMassBalanceOutputProvider())
    )
    if hotstart is not None:
        return builder.with_hotstart(hotstart).build()

    simulation = builder.build()
    simulation.set_array("ground_elevation", domain.arr_dem_flat)
    simulation.set_array("friction", domain.arr_n)
    simulation.set_array("boundary_type", domain.arr_bctype)
    simulation.set_array("water_depth", np.full(domain.domain_data.shape, 0.1, dtype=np.float32))
    simulation.set_array("rainfall_rate", domain.arr_rain)
    simulation.set_array("infiltration", domain.arr_inf)
    simulation.set_array("losses", domain.arr_loss)
    simulation.set_array(
        "inflow", np.full(domain.domain_data.shape, 1.0 / 3_600_000, dtype=np.float32)
    )
    return simulation.initialize()


def assert_report_intervals(simulation: Simulation, previous_report_time: datetime) -> None:
    raster_output = simulation.report.raster_provider
    mass_balance_output = simulation.report.mass_balance_output_provider
    assert isinstance(raster_output, MemoryRasterOutputProvider)
    assert isinstance(mass_balance_output, MemoryMassBalanceOutputProvider)
    assert mass_balance_output.reports

    expected_means = {
        "mean_rainfall": 10.0,
        "mean_infiltration": 2.0,
        "mean_losses": 1.5,
        "mean_inflow": 1.0 / 3_600_000,
    }
    for key, expected_mean in expected_means.items():
        records = raster_output.output_maps_dict[key]
        assert len(records) == len(mass_balance_output.reports)
        for _, array in records:
            np.testing.assert_allclose(array, expected_mean, rtol=1e-6)

    area = simulation.domain_data.cells * simulation.domain_data.cell_area
    for report in mass_balance_output.reports:
        report_time = report.simulation_time
        if isinstance(report_time, timedelta):
            report_time = simulation.start_time + report_time
        interval_s = (report_time - previous_report_time).total_seconds()
        assert report.timesteps == interval_s
        assert report.average_timestep == pytest.approx(1.0)
        for column, rate in (
            ("rainfall_volume", 10.0),
            ("infiltration_volume", -2.0),
            ("losses_volume", -1.5),
            ("inflow_volume", 1.0),
        ):
            assert getattr(report, column) == pytest.approx(
                rate / 3_600_000 * interval_s * area, rel=1e-6
            )
        previous_report_time = report_time


@pytest.mark.parametrize(
    "case",
    [
        pytest.param((25, 23, 10, 25), id="clipped-final-interval"),
        pytest.param((25, 20, 10, 25), id="report-boundary"),
        pytest.param((25, 0, 10, 25), id="initial-report"),
        pytest.param((5, 3, 10, 5), id="before-first-scheduled-report"),
        pytest.param((25, 23, 1, 25), id="changed-cadence"),
        pytest.param((25, 23, 10, 40), id="extended-end"),
    ],
)
def test_resume_preserves_reporting_interval(
    domain_5by5: Domain5by5Data,
    sim_config: SimulationConfig,
    case: tuple[int, int, int, int],
) -> None:
    duration_s, capture_s, resumed_record_s, resumed_end_s = case
    config = sim_config.model_copy(
        update={"end_time": sim_config.start_time + timedelta(seconds=duration_s)}
    )
    reference = build_simulation(config, domain_5by5)
    if capture_s:
        reference.update_until(timedelta(seconds=capture_s))
    previous_report_time = reference.report.last_step
    hotstart = reference.capture_hotstart()
    resumed_config = config.model_copy(
        update={
            "record_step": timedelta(seconds=resumed_record_s),
            "end_time": config.start_time + timedelta(seconds=resumed_end_s),
        }
    )
    resumed = build_simulation(resumed_config, domain_5by5, hotstart)
    assert resumed.report.last_step == previous_report_time

    resumed.update_until(resumed.end_time - resumed.start_time)
    assert_report_intervals(resumed, previous_report_time)

    if resumed_config == config:
        reference.update_until(reference.end_time - reference.start_time)
        np.testing.assert_array_equal(
            resumed.get_array("water_depth"), reference.get_array("water_depth")
        )
        resumed_output = resumed.report.mass_balance_output_provider
        reference_output = reference.report.mass_balance_output_provider
        assert isinstance(resumed_output, MemoryMassBalanceOutputProvider)
        assert isinstance(reference_output, MemoryMassBalanceOutputProvider)
        assert resumed_output.reports[-1] == reference_output.reports[-1]
    reference.finalize()
    resumed.finalize()


def test_repeated_resume_preserves_changed_cadence_interval(
    domain_5by5: Domain5by5Data, sim_config: SimulationConfig
) -> None:
    source = build_simulation(sim_config, domain_5by5)
    source.update_until(timedelta(seconds=12))
    resumed_config = sim_config.model_copy(update={"record_step": timedelta(seconds=7)})
    first = build_simulation(resumed_config, domain_5by5, source.capture_hotstart())

    # Re-capture before the first resumed report; the accumulators still begin at 10 s.
    immediate = build_simulation(resumed_config, domain_5by5, first.capture_hotstart())
    assert immediate.report.last_step == sim_config.start_time + timedelta(seconds=10)
    immediate.update_until(timedelta(seconds=25))
    assert_report_intervals(immediate, sim_config.start_time + timedelta(seconds=10))

    first.update_until(timedelta(seconds=23))
    assert first.report.last_step == sim_config.start_time + timedelta(seconds=19)
    assert first.schedule.deadline("record") == first.end_time
    second = build_simulation(resumed_config, domain_5by5, first.capture_hotstart())
    assert second.report.last_step == first.report.last_step

    second.update_until(timedelta(seconds=25))
    assert_report_intervals(second, sim_config.start_time + timedelta(seconds=19))
    first.update_until(timedelta(seconds=25))
    np.testing.assert_array_equal(second.get_array("water_depth"), first.get_array("water_depth"))
    first_output = first.report.mass_balance_output_provider
    second_output = second.report.mass_balance_output_provider
    assert isinstance(first_output, MemoryMassBalanceOutputProvider)
    assert isinstance(second_output, MemoryMassBalanceOutputProvider)
    assert second_output.reports[-1] == first_output.reports[-1]
    for simulation in (source, first, immediate, second):
        simulation.finalize()
