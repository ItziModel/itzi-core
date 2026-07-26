"""
Hotstart integration tests for provider-backed timed memory inputs.
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
from itzi_core.itzi_error import HotstartError, NullError
from itzi_core.providers.memory_input import MemoryRasterInputProvider, TimedRasterSlice
from itzi_core.providers.memory_output import (
    MemoryRasterOutputProvider,
    MemoryVectorOutputProvider,
)
from itzi_core.simulation_builder import SimulationBuilder

if TYPE_CHECKING:
    from itzi_core.simulation import Simulation


TIME_SLICE_SECONDS = (0, 10, 20, 30)
RAIN_MM_PER_HOUR = {
    0: 0.0,
    10: 180.0,
    20: 360.0,
    30: 540.0,
}


def _run_to_end(simulation: "Simulation", *, skip_initialize: bool = False) -> None:
    if not skip_initialize:
        simulation.initialize()
    while simulation.sim_time < simulation.end_time:
        simulation.update()
    simulation.finalize()


def _assert_final_state_matches(resumed: "Simulation", reference: "Simulation") -> None:
    for key in ["water_depth", "qe", "qs"]:
        np.testing.assert_allclose(
            resumed.raster_domain.get_array(key),
            reference.raster_domain.get_array(key),
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"Final {key} mismatch",
        )


def _make_expected_rain_array(shape: tuple[int, int], rain_mm_per_hour: float) -> np.ndarray:
    return np.full(shape, rain_mm_per_hour / (1000 * 3600), dtype=np.float32)


def _make_hotstart_timed_rain_slices(
    shape: tuple[int, int],
    start_time: datetime,
) -> tuple[list[TimedRasterSlice], dict[int, np.ndarray]]:
    timed_slices = [
        TimedRasterSlice(
            start_time=start_time,
            end_time=start_time + timedelta(seconds=10),
            array=np.full(shape, RAIN_MM_PER_HOUR[0], dtype=np.float32),
        ),
        TimedRasterSlice(
            start_time=start_time + timedelta(seconds=10),
            end_time=start_time + timedelta(seconds=20),
            array=np.full(shape, RAIN_MM_PER_HOUR[10], dtype=np.float32),
        ),
        TimedRasterSlice(
            start_time=start_time + timedelta(seconds=20),
            end_time=start_time + timedelta(seconds=30),
            array=np.full(shape, RAIN_MM_PER_HOUR[20], dtype=np.float32),
        ),
        TimedRasterSlice(
            start_time=start_time + timedelta(seconds=30),
            end_time=start_time + timedelta(seconds=40),
            array=np.full(shape, RAIN_MM_PER_HOUR[30], dtype=np.float32),
        ),
    ]
    expected_rain_arrays = {
        seconds: _make_expected_rain_array(shape, rain_mm_per_hour)
        for seconds, rain_mm_per_hour in RAIN_MM_PER_HOUR.items()
    }
    return timed_slices, expected_rain_arrays


def _make_boundary_timed_rain_slices(
    shape: tuple[int, int],
    start_time: datetime,
) -> tuple[list[TimedRasterSlice], dict[int, np.ndarray]]:
    timed_slices = [
        TimedRasterSlice(
            start_time=start_time,
            end_time=start_time + timedelta(seconds=10),
            array=np.zeros(shape, dtype=np.float32),
        ),
        TimedRasterSlice(
            start_time=start_time + timedelta(seconds=10),
            end_time=start_time + timedelta(seconds=20),
            array=np.full(shape, 360.0, dtype=np.float32),
        ),
    ]
    expected_rain_arrays = {
        0: _make_expected_rain_array(shape, 0.0),
        10: _make_expected_rain_array(shape, 360.0),
    }
    return timed_slices, expected_rain_arrays


def _make_static_arrays(
    domain_5by5,
    *,
    dem: np.ndarray | None = None,
    water_depth: np.ndarray | None = None,
    rain: np.ndarray | None = None,
    include_water_depth: bool = True,
) -> dict[str, np.ndarray]:
    static_arrays = {
        "dem": domain_5by5.arr_dem_flat.copy() if dem is None else dem.copy(),
        "friction": domain_5by5.arr_n.copy(),
    }
    if include_water_depth:
        static_arrays["water_depth"] = (
            domain_5by5.arr_start_h.copy() if water_depth is None else water_depth.copy()
        )
    if rain is not None:
        static_arrays["rain"] = rain.copy()
    return static_arrays


def _make_simulation_config(
    start_time: datetime,
    end_time: datetime,
    *,
    temporal_type: TemporalType,
    record_step: timedelta = timedelta(seconds=10),
    surface_flow_parameters: SurfaceFlowParameters | None = None,
) -> SimulationConfig:
    if surface_flow_parameters is None:
        surface_flow_parameters = SurfaceFlowParameters(hmin=0.0001, dtmax=0.3, cfl=0.2)

    return SimulationConfig(
        start_time=start_time,
        end_time=end_time,
        record_step=record_step,
        temporal_type=temporal_type,
        input_map_names={
            "dem": "dem",
            "friction": "friction",
            "water_depth": "water_depth",
            "rain": "rain",
        },
        output_map_names={"water_depth": "out_hotstart_timed_inputs_water_depth"},
        surface_flow_parameters=surface_flow_parameters,
        infiltration_model=InfiltrationModelType.NULL,
    )


def _build_provider_simulation(
    sim_config: SimulationConfig,
    domain_5by5,
    *,
    static_arrays: dict[str, np.ndarray],
    timed_arrays: dict[str, list[TimedRasterSlice]] | None = None,
    hotstart_bytes: bytes | None = None,
) -> "Simulation":
    input_provider = MemoryRasterInputProvider(
        {
            "domain_data": domain_5by5.domain_data,
            "simulation_start_time": sim_config.start_time,
            "simulation_end_time": sim_config.end_time,
            "static_arrays": static_arrays,
            "timed_arrays": timed_arrays or {},
        }
    )

    builder = (
        SimulationBuilder(sim_config, domain_5by5.arr_mask, np.float32)
        .with_input_provider(input_provider)
        .with_raster_output_provider(
            MemoryRasterOutputProvider({"out_map_names": sim_config.output_map_names})
        )
        .with_vector_output_provider(MemoryVectorOutputProvider({}))
    )
    if hotstart_bytes is not None:
        builder.with_hotstart(hotstart_bytes)
    return builder.build()


def _run_reference_with_hotstart_checkpoint(
    sim_config: SimulationConfig,
    domain_5by5,
    *,
    static_arrays: dict[str, np.ndarray],
    timed_arrays: dict[str, list[TimedRasterSlice]],
    split_target_time: datetime,
) -> tuple[dict[str, datetime | np.ndarray], bytes, "Simulation"]:
    simulation = _build_provider_simulation(
        sim_config,
        domain_5by5,
        static_arrays=static_arrays,
        timed_arrays=timed_arrays,
    )
    simulation.initialize()

    while simulation.sim_time < split_target_time:
        simulation.update()

    checkpoint = {
        "sim_time": simulation.sim_time,
        "rain": simulation.raster_domain.get_array("rain").copy(),
        "water_depth": simulation.raster_domain.get_array("water_depth").copy(),
    }
    hotstart_bytes = simulation.create_hotstart().getvalue()

    while simulation.sim_time < simulation.end_time:
        simulation.update()
    simulation.finalize()

    return checkpoint, hotstart_bytes, simulation


def _assert_resume_with_timed_memory_inputs(
    domain_5by5,
    *,
    temporal_type: TemporalType,
) -> None:
    start_time = datetime(2000, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(seconds=25)
    split_target_time = start_time + timedelta(seconds=12)
    timed_rain_slices, expected_rain_arrays = _make_hotstart_timed_rain_slices(
        domain_5by5.domain_data.shape,
        start_time,
    )
    static_arrays = _make_static_arrays(domain_5by5)
    sim_config = _make_simulation_config(
        start_time,
        end_time,
        temporal_type=temporal_type,
    )
    checkpoint, hotstart_bytes, reference = _run_reference_with_hotstart_checkpoint(
        sim_config,
        domain_5by5,
        static_arrays=static_arrays,
        timed_arrays={"rain": timed_rain_slices},
        split_target_time=split_target_time,
    )

    saved_sim_time = checkpoint["sim_time"]
    second_slice_start = start_time + timedelta(seconds=10)
    second_slice_end = start_time + timedelta(seconds=20)
    assert second_slice_start <= saved_sim_time < second_slice_end

    resumed = _build_provider_simulation(
        sim_config,
        domain_5by5,
        static_arrays=static_arrays,
        timed_arrays={"rain": timed_rain_slices},
        hotstart_bytes=hotstart_bytes,
    )

    assert resumed.sim_time == saved_sim_time
    np.testing.assert_allclose(resumed.raster_domain.get_array("rain"), checkpoint["rain"])
    np.testing.assert_allclose(resumed.raster_domain.get_array("rain"), expected_rain_arrays[10])

    assert resumed.timed_arrays is not None
    rain_timed_array = resumed.timed_arrays["rain"]
    assert second_slice_start <= resumed.sim_time < second_slice_end
    assert rain_timed_array.arr_start == second_slice_start
    assert rain_timed_array.arr_end == second_slice_end
    np.testing.assert_allclose(resumed.raster_domain.get_array("rain"), expected_rain_arrays[10])
    assert not np.allclose(resumed.raster_domain.get_array("rain"), expected_rain_arrays[0])

    resumed.update()
    assert second_slice_start <= resumed.sim_time < second_slice_end
    np.testing.assert_allclose(resumed.raster_domain.get_array("rain"), expected_rain_arrays[10])

    _run_to_end(resumed, skip_initialize=True)
    _assert_final_state_matches(resumed, reference)


def test_resume_with_relative_time_memory_inputs(domain_5by5) -> None:
    _assert_resume_with_timed_memory_inputs(
        domain_5by5,
        temporal_type=TemporalType.RELATIVE,
    )


def test_resume_with_absolute_time_memory_inputs(domain_5by5) -> None:
    _assert_resume_with_timed_memory_inputs(
        domain_5by5,
        temporal_type=TemporalType.ABSOLUTE,
    )


def test_resume_rejects_changed_input_map_names(domain_5by5) -> None:
    start_time = datetime(2000, 1, 1)
    end_time = start_time + timedelta(seconds=25)
    timed_rain_slices, _ = _make_hotstart_timed_rain_slices(
        domain_5by5.domain_data.shape, start_time
    )
    sim_config = _make_simulation_config(
        start_time,
        end_time,
        temporal_type=TemporalType.RELATIVE,
    )
    _, hotstart_bytes, _ = _run_reference_with_hotstart_checkpoint(
        sim_config,
        domain_5by5,
        static_arrays=_make_static_arrays(domain_5by5),
        timed_arrays={"rain": timed_rain_slices},
        split_target_time=start_time + timedelta(seconds=12),
    )
    changed_config = sim_config.model_copy(
        update={"input_map_names": {**sim_config.input_map_names, "rain": "other_rain"}}
    )

    with pytest.raises(HotstartError, match="input map names mismatch"):
        _build_provider_simulation(
            changed_config,
            domain_5by5,
            static_arrays=_make_static_arrays(domain_5by5),
            timed_arrays={"rain": timed_rain_slices},
            hotstart_bytes=hotstart_bytes,
        )


def test_resume_rejects_conflicting_timed_input_boundary(domain_5by5) -> None:
    start_time = datetime(2000, 1, 1)
    end_time = start_time + timedelta(seconds=25)
    timed_rain_slices, _ = _make_hotstart_timed_rain_slices(
        domain_5by5.domain_data.shape, start_time
    )
    sim_config = _make_simulation_config(
        start_time,
        end_time,
        temporal_type=TemporalType.RELATIVE,
    )
    _, hotstart_bytes, _ = _run_reference_with_hotstart_checkpoint(
        sim_config,
        domain_5by5,
        static_arrays=_make_static_arrays(domain_5by5),
        timed_arrays={"rain": timed_rain_slices},
        split_target_time=start_time + timedelta(seconds=12),
    )
    changed_slices = [
        timed_rain_slices[0],
        TimedRasterSlice(
            start_time=start_time + timedelta(seconds=10),
            end_time=start_time + timedelta(seconds=22),
            array=np.full(domain_5by5.domain_data.shape, RAIN_MM_PER_HOUR[10], dtype=np.float32),
        ),
        TimedRasterSlice(
            start_time=start_time + timedelta(seconds=22),
            end_time=start_time + timedelta(seconds=30),
            array=np.full(domain_5by5.domain_data.shape, RAIN_MM_PER_HOUR[20], dtype=np.float32),
        ),
    ]

    with pytest.raises(HotstartError, match="boundary conflicts"):
        _build_provider_simulation(
            sim_config,
            domain_5by5,
            static_arrays=_make_static_arrays(domain_5by5),
            timed_arrays={"rain": changed_slices},
            hotstart_bytes=hotstart_bytes,
        )


def test_resume_rejects_pending_timed_input_without_provider(domain_5by5) -> None:
    start_time = datetime(2000, 1, 1)
    end_time = start_time + timedelta(seconds=25)
    timed_rain_slices, _ = _make_hotstart_timed_rain_slices(
        domain_5by5.domain_data.shape, start_time
    )
    sim_config = _make_simulation_config(
        start_time,
        end_time,
        temporal_type=TemporalType.RELATIVE,
    )
    _, hotstart_bytes, _ = _run_reference_with_hotstart_checkpoint(
        sim_config,
        domain_5by5,
        static_arrays=_make_static_arrays(domain_5by5),
        timed_arrays={"rain": timed_rain_slices},
        split_target_time=start_time + timedelta(seconds=12),
    )
    builder = (
        SimulationBuilder(sim_config, domain_5by5.arr_mask, np.float32)
        .with_domain_data(domain_5by5.domain_data)
        .with_raster_output_provider(
            MemoryRasterOutputProvider({"out_map_names": sim_config.output_map_names})
        )
        .with_vector_output_provider(MemoryVectorOutputProvider({}))
        .with_hotstart(hotstart_bytes)
    )

    with pytest.raises(HotstartError, match="pending timed-input deadline"):
        builder.build()


@pytest.mark.parametrize("source_key", ["water_depth", "water_surface_elevation"])
@pytest.mark.parametrize("split_seconds", [10, 12], ids=["input_boundary", "mid_interval"])
def test_hotstart_priming_preserves_evolved_water_depth(
    domain_5by5,
    source_key: str,
    split_seconds: int,
) -> None:
    start_time = datetime(2000, 1, 1)
    end_time = start_time + timedelta(seconds=25)
    source_depth = domain_5by5.arr_start_h.copy()
    timed_rain_slices, _ = _make_hotstart_timed_rain_slices(
        domain_5by5.domain_data.shape, start_time
    )
    dem = (
        domain_5by5.arr_dem_high.copy()
        if source_key == "water_surface_elevation"
        else domain_5by5.arr_dem_flat.copy()
    )
    source_array = dem + source_depth if source_key == "water_surface_elevation" else source_depth
    sim_config = _make_simulation_config(
        start_time,
        end_time,
        temporal_type=TemporalType.RELATIVE,
    ).model_copy(
        update={
            "input_map_names": {
                "dem": "dem",
                "friction": "friction",
                "rain": "rain",
                source_key: source_key,
            }
        }
    )
    static_arrays = _make_static_arrays(
        domain_5by5,
        dem=dem,
        include_water_depth=False,
    )
    timed_arrays = {
        "rain": timed_rain_slices,
        source_key: [
            TimedRasterSlice(
                start_time=start_time,
                end_time=end_time,
                array=source_array,
            )
        ],
    }
    checkpoint, hotstart_bytes, reference = _run_reference_with_hotstart_checkpoint(
        sim_config,
        domain_5by5,
        static_arrays=static_arrays,
        timed_arrays=timed_arrays,
        split_target_time=start_time + timedelta(seconds=split_seconds),
    )
    archived_depth = checkpoint["water_depth"]

    if split_seconds == 10:
        assert checkpoint["sim_time"] == start_time + timedelta(seconds=10)
    else:
        assert (
            start_time + timedelta(seconds=10)
            < checkpoint["sim_time"]
            < (start_time + timedelta(seconds=20))
        )
    assert not np.allclose(archived_depth, source_depth, rtol=0, atol=1e-4)

    resumed = _build_provider_simulation(
        sim_config,
        domain_5by5,
        static_arrays=static_arrays,
        timed_arrays=timed_arrays,
        hotstart_bytes=hotstart_bytes,
    )

    assert resumed.timed_arrays is not None
    source_cache = resumed.timed_arrays[source_key]
    assert source_cache.arr_start == start_time
    assert source_cache.arr_end == end_time
    np.testing.assert_array_equal(resumed.raster_domain.get_array("water_depth"), archived_depth)

    _run_to_end(resumed, skip_initialize=True)
    _assert_final_state_matches(resumed, reference)


@pytest.mark.parametrize(
    ("target_seconds", "expected_source_seconds", "expected_window"),
    [
        (9, 0, (0, 10)),
        (10, 10, (10, 20)),
        (12, 10, (10, 20)),
    ],
)
def test_timed_memory_rain_switches_cleanly_around_boundary(
    domain_5by5,
    target_seconds: int,
    expected_source_seconds: int,
    expected_window: tuple[int, int],
) -> None:
    start_time = datetime(2000, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(seconds=20)
    timed_rain_slices, expected_rain_arrays = _make_boundary_timed_rain_slices(
        domain_5by5.domain_data.shape,
        start_time,
    )
    sim_config = _make_simulation_config(
        start_time,
        end_time,
        temporal_type=TemporalType.RELATIVE,
        record_step=timedelta(seconds=20),
        surface_flow_parameters=SurfaceFlowParameters(hmin=0.0001, dtmax=20.0, cfl=0.2),
    )
    simulation = _build_provider_simulation(
        sim_config,
        domain_5by5,
        static_arrays=_make_static_arrays(
            domain_5by5,
            water_depth=np.zeros(domain_5by5.domain_data.shape, dtype=np.float32),
        ),
        timed_arrays={"rain": timed_rain_slices},
    )

    simulation.initialize()
    simulation.update_until(timedelta(seconds=target_seconds))

    assert simulation.timed_arrays is not None
    rain_timed_array = simulation.timed_arrays["rain"]
    np.testing.assert_allclose(
        simulation.raster_domain.get_array("rain"),
        expected_rain_arrays[expected_source_seconds],
    )
    expected_start = start_time + timedelta(seconds=expected_window[0])
    expected_end = start_time + timedelta(seconds=expected_window[1])
    assert rain_timed_array.arr_start == expected_start
    assert rain_timed_array.arr_end == expected_end


def test_timed_memory_rain_is_applied_before_a_step_crosses_its_boundary(domain_5by5) -> None:
    start_time = datetime(2000, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(seconds=20)
    timed_rain_slices, _ = _make_boundary_timed_rain_slices(
        domain_5by5.domain_data.shape,
        start_time,
    )
    sim_config = _make_simulation_config(
        start_time,
        end_time,
        temporal_type=TemporalType.RELATIVE,
        record_step=timedelta(seconds=15),
        surface_flow_parameters=SurfaceFlowParameters(hmin=0.0001, dtmax=20.0, cfl=0.2),
    )
    simulation = _build_provider_simulation(
        sim_config,
        domain_5by5,
        static_arrays=_make_static_arrays(
            domain_5by5,
            water_depth=np.zeros(domain_5by5.domain_data.shape, dtype=np.float32),
        ),
        timed_arrays={"rain": timed_rain_slices},
    )

    simulation.initialize()
    simulation.update()

    assert simulation.sim_time == start_time + timedelta(seconds=10)
    domain_volume = float(
        np.sum(simulation.raster_domain.get_array("water_depth"))
        * simulation.raster_domain.cell_area
    )
    assert domain_volume == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(
        simulation.raster_domain.get_array("rain"),
        np.full(domain_5by5.domain_data.shape, 360.0 / (1000 * 3600), dtype=np.float32),
    )


def test_update_rejects_a_call_at_the_configured_end(domain_5by5) -> None:
    start_time = datetime(2000, 1, 1)
    end_time = start_time + timedelta(seconds=10)
    sim_config = _make_simulation_config(
        start_time,
        end_time,
        temporal_type=TemporalType.RELATIVE,
    )
    simulation = _build_provider_simulation(
        sim_config,
        domain_5by5,
        static_arrays=_make_static_arrays(domain_5by5),
    )

    simulation.initialize()
    simulation.update_until(end_time - start_time)

    with pytest.raises(ValueError, match="configured end time"):
        simulation.update()


def test_update_until_rejects_a_target_after_the_configured_end(domain_5by5) -> None:
    start_time = datetime(2000, 1, 1)
    end_time = start_time + timedelta(seconds=20)
    sim_config = _make_simulation_config(
        start_time,
        end_time,
        temporal_type=TemporalType.RELATIVE,
        surface_flow_parameters=SurfaceFlowParameters(hmin=0.0001, dtmax=20.0, cfl=0.2),
    )
    simulation = _build_provider_simulation(
        sim_config,
        domain_5by5,
        static_arrays=_make_static_arrays(
            domain_5by5,
            water_depth=np.zeros(domain_5by5.domain_data.shape, dtype=np.float32),
        ),
    )
    simulation.initialize()

    with pytest.raises(ValueError, match="must not exceed the configured simulation end time"):
        simulation.update_until(timedelta(seconds=21))

    assert simulation.sim_time == start_time


def test_update_until_clears_the_temporary_limit_after_an_exception(
    domain_5by5,
    monkeypatch,
) -> None:
    start_time = datetime(2000, 1, 1)
    end_time = start_time + timedelta(seconds=20)
    sim_config = _make_simulation_config(
        start_time,
        end_time,
        temporal_type=TemporalType.RELATIVE,
        surface_flow_parameters=SurfaceFlowParameters(hmin=0.0001, dtmax=20.0, cfl=0.2),
    )
    simulation = _build_provider_simulation(
        sim_config,
        domain_5by5,
        static_arrays=_make_static_arrays(
            domain_5by5,
            water_depth=np.zeros(domain_5by5.domain_data.shape, dtype=np.float32),
        ),
    )
    simulation.initialize()

    def fail_update() -> None:
        raise RuntimeError("expected update failure")

    with monkeypatch.context() as patch:
        patch.setattr(simulation, "update", fail_update)
        with pytest.raises(RuntimeError, match="expected update failure"):
            simulation.update_until(timedelta(seconds=5))

    simulation.update()
    assert simulation.sim_time == start_time + timedelta(seconds=10)


def test_non_cadence_aligned_end_writes_one_final_report(domain_5by5) -> None:
    start_time = datetime(2000, 1, 1)
    end_time = start_time + timedelta(seconds=25)
    sim_config = _make_simulation_config(
        start_time,
        end_time,
        temporal_type=TemporalType.RELATIVE,
        surface_flow_parameters=SurfaceFlowParameters(hmin=0.0001, dtmax=20.0, cfl=0.2),
    )
    simulation = _build_provider_simulation(
        sim_config,
        domain_5by5,
        static_arrays=_make_static_arrays(
            domain_5by5,
            water_depth=np.zeros(domain_5by5.domain_data.shape, dtype=np.float32),
        ),
    )

    simulation.initialize()
    simulation.update_until(end_time - start_time)
    simulation.finalize()

    output_times = [
        sim_time
        for sim_time, _ in simulation.report.raster_provider.output_maps_dict["water_depth"]
    ]
    assert output_times == [
        timedelta(seconds=0),
        timedelta(seconds=10),
        timedelta(seconds=20),
        timedelta(seconds=25),
    ]
    assert output_times.count(timedelta(seconds=25)) == 1
    assert simulation.next_ts["record"] == start_time + timedelta(seconds=35)


def test_timed_rain_and_inflow_close_old_rates_at_the_input_boundary(domain_5by5) -> None:
    start_time = datetime(2000, 1, 1)
    input_boundary = start_time + timedelta(seconds=10)
    end_time = start_time + timedelta(seconds=20)
    shape = domain_5by5.domain_data.shape
    old_rain_mm_per_hour = 360.0
    new_rain_mm_per_hour = 720.0
    old_rain_rate = old_rain_mm_per_hour / (1000.0 * 3600.0)
    new_rain_rate = new_rain_mm_per_hour / (1000.0 * 3600.0)
    old_inflow_rate = 0.1
    new_inflow_rate = 0.2

    def two_slices(first_value: float, second_value: float) -> list[TimedRasterSlice]:
        return [
            TimedRasterSlice(
                start_time=start_time,
                end_time=input_boundary,
                array=np.full(shape, first_value, dtype=np.float32),
            ),
            TimedRasterSlice(
                start_time=input_boundary,
                end_time=end_time,
                array=np.full(shape, second_value, dtype=np.float32),
            ),
        ]

    sim_config = _make_simulation_config(
        start_time,
        end_time,
        temporal_type=TemporalType.RELATIVE,
        record_step=timedelta(seconds=20),
        surface_flow_parameters=SurfaceFlowParameters(hmin=0.0001, dtmax=20.0, cfl=0.2),
    ).model_copy(
        update={
            "input_map_names": {
                "dem": "dem",
                "friction": "friction",
                "water_depth": "water_depth",
                "rain": "rain",
                "inflow": "inflow",
            }
        }
    )
    simulation = _build_provider_simulation(
        sim_config,
        domain_5by5,
        static_arrays=_make_static_arrays(
            domain_5by5,
            water_depth=np.zeros(shape, dtype=np.float32),
        ),
        timed_arrays={
            "rain": two_slices(old_rain_mm_per_hour, new_rain_mm_per_hour),
            "inflow": two_slices(old_inflow_rate, new_inflow_rate),
        },
    )

    simulation.initialize()
    simulation.update()

    assert simulation.sim_time == input_boundary
    np.testing.assert_allclose(
        simulation.raster_domain.get_array("rain"),
        np.full(shape, new_rain_rate, dtype=np.float32),
    )
    np.testing.assert_allclose(
        simulation.raster_domain.get_array("inflow"),
        np.full(shape, new_inflow_rate, dtype=np.float32),
    )
    np.testing.assert_allclose(
        simulation.raster_domain.get_array("rainfall_accum"),
        np.full(shape, old_rain_rate * 10.0, dtype=np.float32),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        simulation.raster_domain.get_array("inflow_accum"),
        np.full(shape, old_inflow_rate * 10.0, dtype=np.float32),
        rtol=1e-6,
    )
    assert simulation.accum_update_time["rainfall_accum"] == input_boundary
    assert simulation.accum_update_time["inflow_accum"] == input_boundary


def test_build_fails_when_dem_input_has_only_nan_cells(domain_5by5) -> None:
    start_time = datetime(2000, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(seconds=25)
    sim_config = _make_simulation_config(
        start_time,
        end_time,
        temporal_type=TemporalType.ABSOLUTE,
    )

    with pytest.raises(NullError, match=r"input map <dem> contains only NULL/NaN cells"):
        _build_provider_simulation(
            sim_config,
            domain_5by5,
            static_arrays=_make_static_arrays(
                domain_5by5,
                dem=np.full(domain_5by5.domain_data.shape, np.nan, dtype=np.float32),
            ),
        )


def test_build_raises_when_non_dem_input_has_only_nan_cells(domain_5by5) -> None:
    start_time = datetime(2000, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(seconds=25)
    sim_config = _make_simulation_config(
        start_time,
        end_time,
        temporal_type=TemporalType.ABSOLUTE,
    )

    with pytest.raises(RuntimeWarning, match=r"input map <rain> contains only NULL/NaN cells"):
        _build_provider_simulation(
            sim_config,
            domain_5by5,
            static_arrays=_make_static_arrays(
                domain_5by5,
                rain=np.full(domain_5by5.domain_data.shape, np.nan, dtype=np.float32),
            ),
        )
