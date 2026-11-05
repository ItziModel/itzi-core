"""Structured hotstart capture, validation, and restoration tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import ValidationError

from itzi_core import (
    HotstartData,
    RasterHotstartState,
    Simulation,
    SimulationBuilder,
    SimulationConfig,
    SurfaceFlowParameters,
)
from itzi_core.const import InfiltrationModelType, TemporalType
from itzi_core.itzi_error import HotstartError
from itzi_core.rasterdomain import RasterDomain

if TYPE_CHECKING:
    from tests.conftest import Domain5by5Data


@pytest.fixture
def sim_config() -> SimulationConfig:
    start = datetime(2000, 1, 1, tzinfo=UTC)
    return SimulationConfig(
        start_time=start,
        end_time=start + timedelta(seconds=60),
        record_step=timedelta(seconds=30),
        temporal_type=TemporalType.RELATIVE,
        input_map_names={"ground_elevation": "dem", "friction": "n", "water_depth": "depth"},
        output_map_names={},
        surface_flow_parameters=SurfaceFlowParameters(hmin=0.0001, dtmax=0.3, cfl=0.2),
    )


@pytest.fixture
def simulation(domain_5by5: Domain5by5Data, sim_config: SimulationConfig) -> Simulation:
    simulation = (
        SimulationBuilder(sim_config, domain_5by5.arr_mask.copy())
        .with_domain_data(domain_5by5.domain_data)
        .build()
    )
    simulation.set_array("ground_elevation", domain_5by5.arr_dem_flat.copy())
    simulation.set_array("friction", domain_5by5.arr_n.copy())
    simulation.set_array("water_depth", domain_5by5.arr_start_h.copy())
    return simulation


@pytest.fixture
def hotstart(simulation: Simulation) -> HotstartData:
    simulation.initialize().update()
    return simulation.capture_hotstart()


def resume_builder(
    domain_5by5: Domain5by5Data, config: SimulationConfig, hotstart: HotstartData
) -> SimulationBuilder:
    return (
        SimulationBuilder(config, domain_5by5.arr_mask)
        .with_domain_data(domain_5by5.domain_data)
        .with_hotstart(hotstart)
    )


def test_capture_requires_initialized_state(simulation: Simulation) -> None:
    with pytest.raises(RuntimeError, match="initialize"):
        simulation.capture_hotstart()


def test_capture_owns_arrays_and_mutable_metadata(simulation: Simulation) -> None:
    simulation.initialize().update()
    hotstart = simulation.capture_hotstart()
    arrays = {key: arr.copy() for key, arr in hotstart.raster_state.padded_arrays.items()}
    metadata = hotstart.metadata.model_dump_json()
    mask = hotstart.raster_state.mask.copy()

    for key, array in hotstart.raster_state.padded_arrays.items():
        assert not np.shares_memory(array, simulation.raster_domain.get_padded(key))
    assert not np.shares_memory(hotstart.raster_state.mask, simulation.raster_domain.mask)

    simulation.update()
    simulation.sim_config.input_map_names["friction"] = "changed"
    simulation.raster_domain.mask[:] = True

    assert hotstart.metadata.model_dump_json() == metadata
    np.testing.assert_array_equal(hotstart.raster_state.mask, mask)
    for key, array in arrays.items():
        np.testing.assert_array_equal(hotstart.raster_state.padded_arrays[key], array)


def test_caller_can_reconstruct_checkpoint_from_metadata_and_arrays(
    domain_5by5: Domain5by5Data, sim_config: SimulationConfig, hotstart: HotstartData
) -> None:
    loaded = HotstartData(
        metadata=json.loads(hotstart.metadata.model_dump_json()),
        raster_state=RasterHotstartState(
            mask=hotstart.raster_state.mask.copy(),
            padded_arrays={
                key: array.copy() for key, array in hotstart.raster_state.padded_arrays.items()
            },
        ),
        swmm_hotstart_bytes=hotstart.swmm_hotstart_bytes,
    )
    resumed = resume_builder(domain_5by5, sim_config, loaded).build()

    assert resumed.sim_time == hotstart.metadata.simulation_state.sim_time
    assert loaded.metadata.state_version == 2
    assert resumed.report.last_step == loaded.metadata.simulation_state.last_report_time
    assert loaded.metadata.resume_config.input_sources == sim_config.input_map_names
    for key, array in loaded.raster_state.padded_arrays.items():
        np.testing.assert_array_equal(resumed.raster_domain.get_padded(key), array)


def test_one_checkpoint_can_resume_independent_simulations(
    domain_5by5: Domain5by5Data, sim_config: SimulationConfig, hotstart: HotstartData
) -> None:
    first = resume_builder(domain_5by5, sim_config, hotstart).build()
    second = resume_builder(domain_5by5, sim_config, hotstart).build()
    original_arrays = {key: arr.copy() for key, arr in hotstart.raster_state.padded_arrays.items()}
    original_metadata = hotstart.metadata.model_dump_json()
    first.update()
    first.time_steps_counters["since_start"] = 999

    assert second.sim_time == hotstart.metadata.simulation_state.sim_time
    assert hotstart.metadata.model_dump_json() == original_metadata
    for key, array in original_arrays.items():
        np.testing.assert_array_equal(second.raster_domain.get_padded(key), array)
        np.testing.assert_array_equal(hotstart.raster_state.padded_arrays[key], array)
    assert isinstance(second.capture_hotstart(), HotstartData)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_capture_and_resume_preserve_solver_padding_and_dtypes(
    domain_5by5: Domain5by5Data,
    sim_config: SimulationConfig,
    dtype: type[np.float32 | np.float64],
) -> None:
    source = (
        SimulationBuilder(sim_config, domain_5by5.arr_mask, dtype)
        .with_domain_data(domain_5by5.domain_data)
        .build()
        .initialize()
    )
    # A solver boundary value cannot be recovered by padding the interior raster.
    source.raster_domain.get_padded("discharge_east")[0, :] = 12.5
    source.raster_domain.get_padded("discharge_south")[:, -1] = -3.25
    hotstart = source.capture_hotstart()
    resumed = (
        SimulationBuilder(sim_config, domain_5by5.arr_mask, dtype)
        .with_domain_data(domain_5by5.domain_data)
        .with_hotstart(hotstart)
        .build()
    )

    for key, array in hotstart.raster_state.padded_arrays.items():
        original = source.raster_domain.get_padded(key)
        restored = resumed.raster_domain.get_padded(key)
        assert array.dtype == original.dtype == restored.dtype
        np.testing.assert_array_equal(array, original)
        np.testing.assert_array_equal(restored, original)
    assert hotstart.raster_state.padded_arrays["boundary_type"].dtype == np.uint8


@pytest.mark.parametrize("source", [b"old archive", "checkpoint.zip"])
def test_builder_requires_structured_data(
    domain_5by5: Domain5by5Data, sim_config: SimulationConfig, source: bytes | str
) -> None:
    builder = SimulationBuilder(sim_config, domain_5by5.arr_mask)
    with pytest.raises(TypeError, match="HotstartData"):
        builder.with_hotstart(source)


@pytest.mark.parametrize(
    ("damage", "expected_error"),
    [
        ("mask_shape", "Mask shape mismatch"),
        ("mask_dtype", "Mask dtype mismatch"),
        ("mask_content", "Mask content mismatch"),
        ("missing_array", "missing required arrays"),
        ("extra_array", "unexpected arrays"),
        ("array_shape", "shape mismatch"),
        ("array_dtype", "dtype mismatch"),
        ("boundary_code", "Invalid values for 'boundary_type'"),
    ],
)
def test_raster_restoration_validates_before_mutating(
    simulation: Simulation, damage: str, expected_error: str
) -> None:
    state = simulation.raster_domain.export_state()
    if damage == "mask_shape":
        state = state.model_copy(update={"mask": np.zeros((3, 3), dtype=np.bool_)})
    elif damage == "mask_dtype":
        state = state.model_copy(update={"mask": state.mask.astype(np.uint8)})
    elif damage == "mask_content":
        state.mask[0, 0] = not state.mask[0, 0]
    elif damage == "missing_array":
        del state.padded_arrays["water_depth"]
    elif damage == "extra_array":
        state.padded_arrays["unexpected"] = np.zeros((3, 3))
    elif damage == "array_shape":
        state.padded_arrays["water_depth"] = np.zeros((3, 3), dtype=np.float32)
    elif damage == "array_dtype":
        state.padded_arrays["water_depth"] = state.padded_arrays["water_depth"].astype(np.float64)
    else:
        state.padded_arrays["boundary_type"][2, 2] = 255

    original = simulation.raster_domain.export_state()
    with pytest.raises(HotstartError, match=expected_error):
        simulation.raster_domain.restore_state(state)

    np.testing.assert_array_equal(simulation.raster_domain.mask, original.mask)
    for key, array in original.padded_arrays.items():
        np.testing.assert_array_equal(simulation.raster_domain.get_padded(key), array)


def test_masked_boundary_normalization_preserves_caller_state() -> None:
    mask = np.zeros((3, 3), dtype=np.bool_)
    mask[0, 0] = True
    domain = RasterDomain(np.float32, mask, (1.0, 1.0))
    state = domain.export_state()
    state.padded_arrays["boundary_type"][1, 1] = 255
    original = state.padded_arrays["boundary_type"].copy()
    for array in state.padded_arrays.values():
        array.flags.writeable = False

    domain.restore_state(state)

    np.testing.assert_array_equal(state.padded_arrays["boundary_type"], original)
    assert domain.get_array("boundary_type")[0, 0] == domain.fill_values["boundary_type"]
    for key in domain.k_all:
        assert np.shares_memory(domain.get_array(key), domain.get_padded(key))


@pytest.mark.parametrize(
    "case",
    [
        ("north", 100.0, "Domain north mismatch"),
        ("south", 25.0, "Domain south mismatch"),
        ("east", 100.0, "Domain east mismatch"),
        ("west", 25.0, "Domain west mismatch"),
        ("rows", 999, "Domain rows mismatch"),
        ("cols", 999, "Domain cols mismatch"),
        ("crs_wkt", "EPSG:4326", "Domain CRS mismatch"),
    ],
)
def test_builder_rejects_domain_mismatch(
    domain_5by5: Domain5by5Data,
    sim_config: SimulationConfig,
    hotstart: HotstartData,
    case: tuple[str, float | int | str, str],
) -> None:
    field, value, error = case
    domain = domain_5by5.domain_data.model_copy(update={field: value})
    builder = (
        SimulationBuilder(sim_config, domain_5by5.arr_mask)
        .with_domain_data(domain)
        .with_hotstart(hotstart)
    )
    with pytest.raises(HotstartError, match=error):
        builder.build()


@pytest.mark.parametrize(
    "case",
    [
        ("start_time", datetime(2000, 1, 1, 0, 0, 5, tzinfo=UTC), "start_time"),
        ("end_time", datetime(2000, 1, 1, tzinfo=UTC), "end_time"),
        ("record_step", timedelta(0), "record_step"),
        ("record_step", timedelta(seconds=-1), "record_step"),
        ("infiltration_model", InfiltrationModelType.GREEN_AMPT, "infiltration"),
    ],
)
def test_builder_rejects_incompatible_config(
    domain_5by5: Domain5by5Data,
    sim_config: SimulationConfig,
    hotstart: HotstartData,
    case: tuple[str, datetime | timedelta | InfiltrationModelType, str],
) -> None:
    field, value, error = case
    config = sim_config.model_copy(update={field: value})
    with pytest.raises(HotstartError, match=error):
        resume_builder(domain_5by5, config, hotstart).build()


@pytest.mark.parametrize(
    "case",
    [
        ("cfl", 0.15, "cfl"),
        ("theta", 0.9, "theta"),
        ("dtmax", 0.2, "dtmax"),
        ("slope_threshold", 1e-5, "slope_threshold"),
        ("max_slope", 5.0, "max_slope"),
        ("hmin", 0.0002, "min_flow_depth"),
    ],
)
def test_builder_allows_surface_flow_tuning(
    domain_5by5: Domain5by5Data,
    sim_config: SimulationConfig,
    hotstart: HotstartData,
    case: tuple[str, float, str],
) -> None:
    parameter, value, attribute = case
    params = sim_config.surface_flow_parameters.model_copy(update={parameter: value})
    config = sim_config.model_copy(update={"surface_flow_parameters": params})
    resumed = resume_builder(domain_5by5, config, hotstart).build()
    assert getattr(resumed.surface_flow, attribute) == value


def test_builder_allows_mass_balance_threshold_change(
    domain_5by5: Domain5by5Data, sim_config: SimulationConfig, hotstart: HotstartData
) -> None:
    params = sim_config.surface_flow_parameters.model_copy(update={"max_error": 0.5})
    config = sim_config.model_copy(update={"surface_flow_parameters": params})
    resumed = resume_builder(domain_5by5, config, hotstart).build()
    assert resumed.mass_balance_error_threshold == 0.5


def test_builder_rejects_gravity_change(
    domain_5by5: Domain5by5Data, sim_config: SimulationConfig, hotstart: HotstartData
) -> None:
    params = sim_config.surface_flow_parameters.model_copy(update={"g": 9.9})
    config = sim_config.model_copy(update={"surface_flow_parameters": params})
    with pytest.raises(HotstartError, match="g: current"):
        resume_builder(domain_5by5, config, hotstart).build()


@pytest.mark.parametrize("has_drainage", [True, False])
def test_builder_rejects_drainage_mismatch(
    domain_5by5: Domain5by5Data,
    sim_config: SimulationConfig,
    hotstart: HotstartData,
    has_drainage: bool,
) -> None:
    if has_drainage:
        hotstart = hotstart.model_copy(update={"swmm_hotstart_bytes": b"SWMM state"})
        config = sim_config
    else:
        config = sim_config.model_copy(update={"swmm_inp": "fake.inp"})
    with pytest.raises(HotstartError, match="drainage state"):
        resume_builder(domain_5by5, config, hotstart).build()


@pytest.mark.parametrize("has_payload", [True, False])
def test_builder_rejects_inconsistent_swmm_elapsed_time(
    domain_5by5: Domain5by5Data,
    sim_config: SimulationConfig,
    hotstart: HotstartData,
    has_payload: bool,
) -> None:
    state = hotstart.metadata.simulation_state.model_copy(
        update={"swmm_elapsed_time": None if has_payload else 0.0}
    )
    hotstart = hotstart.model_copy(
        update={
            "metadata": hotstart.metadata.model_copy(update={"simulation_state": state}),
            "swmm_hotstart_bytes": b"SWMM state" if has_payload else None,
        }
    )
    config = sim_config.model_copy(update={"swmm_inp": "fake.inp" if has_payload else None})
    with pytest.raises(HotstartError, match="SWMM payload and elapsed time"):
        resume_builder(domain_5by5, config, hotstart).build()


@pytest.mark.parametrize(
    "damage",
    [
        "version",
        "old_version",
        "schedule",
        "counters",
        "accumulations",
        "dt",
        "report_before_start",
        "report_after_checkpoint",
    ],
)
def test_builder_revalidates_modified_checkpoints_before_restoration(
    domain_5by5: Domain5by5Data,
    simulation: Simulation,
    hotstart: HotstartData,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    metadata = hotstart.metadata
    state = metadata.simulation_state
    if damage == "version":
        metadata = metadata.model_copy(update={"state_version": 999})
    elif damage == "old_version":
        metadata = metadata.model_copy(update={"state_version": 1})
    elif damage == "schedule":
        state = state.model_copy(update={"next_ts": {}})
    elif damage == "counters":
        state = state.model_copy(update={"time_steps_counters": {}})
    elif damage == "accumulations":
        state = state.model_copy(update={"accum_update_time": {}})
    elif damage == "report_before_start":
        state = state.model_copy(
            update={"last_report_time": simulation.start_time - timedelta(seconds=1)}
        )
    elif damage == "report_after_checkpoint":
        state = state.model_copy(
            update={"last_report_time": state.sim_time + timedelta(seconds=1)}
        )
    else:
        state = state.model_copy(update={"dt": float("nan")})
    metadata = metadata.model_copy(update={"simulation_state": state})
    hotstart = hotstart.model_copy(update={"metadata": metadata})

    def unexpected_restore(*args: object) -> None:
        pytest.fail("Invalid checkpoint reached raster restoration")

    monkeypatch.setattr(RasterDomain, "restore_state", unexpected_restore)
    with pytest.raises(HotstartError):
        resume_builder(domain_5by5, simulation.sim_config, hotstart).build()


def test_raster_state_requires_numpy_arrays(hotstart: HotstartData) -> None:
    with pytest.raises(ValidationError):
        RasterHotstartState(mask=hotstart.raster_state.mask, padded_arrays={"water_depth": [[1]]})
