"""Tests for validation in configuration data containers."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from itzi_core.const import TemporalType
from itzi_core.data_containers import SimulationConfig, SurfaceFlowParameters


def _simulation_config_data() -> dict:
    start_time = datetime(2000, 1, 1, tzinfo=UTC)
    return {
        "start_time": start_time,
        "end_time": start_time + timedelta(seconds=1),
        "record_step": timedelta(seconds=1),
        "temporal_type": TemporalType.RELATIVE,
        "input_map_names": {},
        "output_map_names": {},
        "surface_flow_parameters": SurfaceFlowParameters(),
    }


@pytest.mark.parametrize(
    "parameters",
    [
        {"slope_threshold": 0.9},
        {"max_slope": 0.7},
        {"slope_threshold": 0.9, "max_slope": 0.8},
    ],
)
def test_surface_flow_parameters_require_max_slope_to_cover_slope_threshold(parameters):
    with pytest.raises(ValidationError, match="max_slope must be greater than or equal"):
        SurfaceFlowParameters(**parameters)


def test_surface_flow_parameters_report_invalid_slope_threshold():
    with pytest.raises(ValidationError, match="slope_threshold"):
        SurfaceFlowParameters(slope_threshold=-0.1)


def test_simulation_config_removes_null_map_entries():
    config = SimulationConfig.model_validate(
        _simulation_config_data()
        | {
            "input_map_names": {"ground_elevation": "dem", "rainfall_rate": None},
            "output_map_names": {"water_depth": "depth", "flow_speed": None},
        }
    )

    assert config.input_map_names == {"ground_elevation": "dem"}
    assert config.output_map_names == {"water_depth": "depth"}


def test_simulation_config_rejects_unknown_input_map_key():
    with pytest.raises(
        ValidationError, match="input_map_names contain invalid input keys: unknown"
    ):
        SimulationConfig.model_validate(
            _simulation_config_data() | {"input_map_names": {"unknown": "source"}}
        )


def test_simulation_config_rejects_output_key_in_input_map():
    with pytest.raises(
        ValidationError, match="input_map_names contain invalid input keys: flow_rate_x"
    ):
        SimulationConfig.model_validate(
            _simulation_config_data() | {"input_map_names": {"flow_rate_x": "source"}}
        )


def test_simulation_config_rejects_input_key_in_output_map():
    with pytest.raises(
        ValidationError, match="output_map_names contain invalid output keys: friction"
    ):
        SimulationConfig.model_validate(
            _simulation_config_data() | {"output_map_names": {"friction": "result"}}
        )


def test_simulation_config_allows_empty_and_sparse_map_names():
    empty_config = SimulationConfig.model_validate(_simulation_config_data())
    sparse_config = SimulationConfig.model_validate(
        _simulation_config_data()
        | {
            "input_map_names": {"ground_elevation": "dem"},
            "output_map_names": {"flow_rate_x": "flux_east"},
        }
    )

    assert empty_config.input_map_names == {}
    assert empty_config.output_map_names == {}
    assert sparse_config.input_map_names == {"ground_elevation": "dem"}
    assert sparse_config.output_map_names == {"flow_rate_x": "flux_east"}
