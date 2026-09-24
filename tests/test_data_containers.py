"""Tests for validation in configuration data containers."""

import pytest
from pydantic import ValidationError

from itzi_core.data_containers import SurfaceFlowParameters


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
