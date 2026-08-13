"""
Copyright (C) 2025 Laurent G. Courty

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 2.1
of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.
"""

import numpy as np
import pytest

from itzi_core.compute import rastermetrics


def test_calculate_total_volume():
    """Test calculate_total_volume with known inputs."""
    # Create a test depth array (3x3 grid)
    depth_array = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    cell_surface_area = 10.0  # m²

    # Calculate expected result manually
    total_depth = np.sum(depth_array)
    expected_volume = total_depth * cell_surface_area

    # Call the function and assert result
    result = rastermetrics.calculate_total_volume(depth_array, cell_surface_area)
    assert np.isclose(result, expected_volume)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    ("interior_shape", "padded"),
    [
        ((999, 1001), False),
        ((1000, 1000), False),
        ((1001, 1000), False),
        ((999, 1001), True),
        ((1001, 1000), True),
    ],
)
def test_calculate_total_volume_uses_float64_reduction(dtype, interior_shape, padded):
    """Reductions should stay accurate around the serial/parallel threshold."""
    if padded:
        shape = (interior_shape[0] + 2, interior_shape[1] + 2)
        depths = np.full(shape, 10_000.0, dtype=dtype)
        reduced_depths = depths[1:-1, 1:-1]
        reduced_depths.fill(0.01)
    else:
        depths = np.full(interior_shape, 0.01, dtype=dtype)
        reduced_depths = depths

    reduced_depths[0, 0] = 0.123456
    reduced_depths[interior_shape[0] // 2, interior_shape[1] // 2] = 0.987654
    reduced_depths[-1, -1] = np.nan
    cell_surface_area = 1.23456789012345
    expected = float(np.nansum(reduced_depths, dtype=np.float64)) * cell_surface_area

    result = rastermetrics.calculate_total_volume(depths, cell_surface_area, padded=padded)

    assert result == pytest.approx(expected, rel=1e-10)


def test_calculate_total_volume_preserves_cell_area_precision():
    cell_surface_area = 1.23456789012345
    depths = np.ones((1, 1), dtype=np.float32)

    result = rastermetrics.calculate_total_volume(depths, cell_surface_area)

    assert result == cell_surface_area


def test_calculate_total_volume_all_nan_returns_zero():
    depths = np.full((3, 4), np.nan, dtype=np.float32)

    assert rastermetrics.calculate_total_volume(depths, 2.0) == 0.0


def test_calculate_wse():
    """Test calculate_wse with known inputs."""
    # Create test arrays (3x3 grid)
    h_array = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    dem_array = np.array(
        [[10.0, 10.1, 10.2], [10.3, 10.4, 10.5], [10.6, 10.7, 10.8]], dtype=np.float32
    )

    # Calculate expected result manually
    expected_wse = h_array + dem_array

    # Call the function and assert result
    result = rastermetrics.calculate_wse(h_array, dem_array)
    assert np.allclose(result, expected_wse)


def test_calculate_flux():
    """Test calculate_flux with known inputs."""
    # Create test array (3x3 grid)
    flow_array = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    cell_size = 10.0  # m

    # Calculate expected result manually
    expected_flux = flow_array * cell_size

    # Call the function and assert result
    result = rastermetrics.calculate_flux(flow_array, cell_size)
    assert np.allclose(result, expected_flux)


def test_calculate_average_rate_from_total():
    """Test calculate_average_rate_from_total with various inputs."""
    # Create test array (3x3 grid)
    total_volume_array = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32
    )
    interval_seconds = 60.0  # 1 minute

    # Test case 1: No conversion (conversion_factor = 1.0)
    expected_rate1 = total_volume_array / interval_seconds
    result1 = rastermetrics.calculate_average_rate_from_total(
        total_volume_array, interval_seconds, 1.0
    )
    assert np.allclose(result1, expected_rate1)

    # Test case 2: Conversion from m/s to mm/h (factor = 1000 * 3600)
    conversion_factor = 1000 * 3600  # Convert to mm/h
    expected_rate2 = (total_volume_array / interval_seconds) * conversion_factor
    result2 = rastermetrics.calculate_average_rate_from_total(
        total_volume_array, interval_seconds, conversion_factor
    )
    assert np.allclose(result2, expected_rate2)


def test_accumulate_rate_to_total():
    """Test accumulate_rate_to_total with various inputs."""
    # Create test arrays (3x3 grid)
    accum_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    rate_array = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    time_delta_seconds = 60.0  # 1 minute

    # Store original accum_array for comparison (shallow copy is sufficient for numeric arrays)
    original_accum_array = accum_array.copy()

    # Calculate expected result manually
    expected_accumulation = rate_array * time_delta_seconds
    expected_result = original_accum_array + expected_accumulation

    # Call the function (should modify accum_array in-place)
    rastermetrics.accumulate_rate_to_total(
        accum_array, rate_array, time_delta_seconds, padded=False
    )

    # Assert that accum_array was modified in-place to the expected result
    assert np.allclose(accum_array, expected_result)

    # Make sure the original array has not changed
    assert not np.allclose(accum_array, original_accum_array)

    # Test case 2: Zero time delta
    accum_array2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rate_array2 = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    original_accum_array2 = accum_array2.copy()

    # With zero time delta, accum_array should remain unchanged
    rastermetrics.accumulate_rate_to_total(accum_array2, rate_array2, 0.0)
    assert np.allclose(accum_array2, original_accum_array2)
