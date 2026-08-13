"""
Tests for flow computation optimizations
Testing mathematical calculations: velocity magnitude and direction

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

from math import atan2, copysign, pi, sqrt

import numpy as np
import pytest

from itzi_core.compute.partial_inertia_h import solve_h
from itzi_core.compute.partial_inertia_q import solve_q


def _solve_q_at_face(
    *,
    axis: str,
    depth: float,
    slope: float,
    hmin: float,
    slope_threshold: float,
    max_slope: float,
    dtype: type[np.floating] = np.float64,
    n0: float = 0.05,
    n1: float | None = None,
) -> float:
    """Return the new discharge at one interior face of a padded domain."""
    shape = (5, 5)
    target = (2, 2)
    neighbor = (2, 3) if axis == "east" else (3, 2)

    arr_z = np.zeros(shape, dtype=dtype)
    arr_z[neighbor] = -slope
    arr_n = np.full(shape, n0, dtype=dtype)
    if n1 is not None:
        arr_n[neighbor] = n1
    arr_h = np.full(shape, depth, dtype=dtype)
    arr_qe = np.zeros(shape, dtype=dtype)
    arr_qs = np.zeros(shape, dtype=dtype)
    arr_hfe = np.zeros(shape, dtype=dtype)
    arr_hfs = np.zeros(shape, dtype=dtype)
    arr_bctype = np.zeros(shape, dtype=np.uint8)
    arr_qe_new = np.zeros(shape, dtype=dtype)
    arr_qs_new = np.zeros(shape, dtype=dtype)

    solve_q(
        arr_z=arr_z,
        arr_n=arr_n,
        arr_h=arr_h,
        arr_qe=arr_qe,
        arr_qs=arr_qs,
        arr_hfe=arr_hfe,
        arr_hfs=arr_hfs,
        arr_bctype=arr_bctype,
        arr_qe_new=arr_qe_new,
        arr_qs_new=arr_qs_new,
        dt=0.1,
        dx=1.0,
        dy=1.0,
        g=9.81,
        theta=0.7,
        hf_min=hmin,
        slope_threshold=slope_threshold,
        max_slope=max_slope,
    )
    flow_array = arr_qe_new if axis == "east" else arr_qs_new
    return float(flow_array[target])


def _gms_discharge(depth: float, n: float, slope: float, max_slope: float) -> float:
    return copysign(depth ** (5.0 / 3.0) / n * sqrt(min(abs(slope), max_slope)), slope)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    ("depth", "uses_gms"),
    [
        (0.04, True),
        (0.05, True),
        (0.06, False),
    ],
)
def test_solve_q_selects_routing_formula_at_depth_threshold(dtype, depth, uses_gms):
    """Shallow faces use GMS; only depths strictly above hmin use Almeida."""
    slope = 0.1
    discharge = _solve_q_at_face(
        axis="east",
        depth=depth,
        slope=slope,
        hmin=0.05,
        slope_threshold=0.8,
        max_slope=1.0,
        dtype=dtype,
    )

    expected = (
        _gms_discharge(depth, n=0.05, slope=slope, max_slope=1.0)
        if uses_gms
        else 9.81 * depth * 0.1 * slope
    )
    assert discharge == pytest.approx(expected, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    ("slope", "uses_gms"),
    [
        (0.79, False),
        (0.8, True),
        (0.81, True),
    ],
)
def test_solve_q_selects_routing_formula_at_slope_threshold(dtype, slope, uses_gms):
    """Only slopes strictly below slope_threshold use the Almeida formula."""
    depth = 0.1
    discharge = _solve_q_at_face(
        axis="east",
        depth=depth,
        slope=slope,
        hmin=0.05,
        slope_threshold=0.8,
        max_slope=2.0,
        dtype=dtype,
    )

    expected = (
        _gms_discharge(depth, n=0.05, slope=slope, max_slope=2.0)
        if uses_gms
        else 9.81 * depth * 0.1 * slope
    )
    assert discharge == pytest.approx(expected, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    ("axis", "slope"), [("east", 10.0), ("east", -10.0), ("south", 10.0), ("south", -10.0)]
)
def test_solve_q_gms_caps_slope_and_preserves_direction(dtype, axis, slope):
    """GMS applies the slope cap and the averaged Manning coefficient on both axes."""
    depth = 0.1
    max_slope = 0.2
    discharge = _solve_q_at_face(
        axis=axis,
        depth=depth,
        slope=slope,
        hmin=0.05,
        slope_threshold=0.8,
        max_slope=max_slope,
        dtype=dtype,
        n0=0.04,
        n1=0.06,
    )

    expected = _gms_discharge(depth, n=0.05, slope=slope, max_slope=max_slope)
    assert discharge == pytest.approx(expected, rel=1e-5, abs=1e-8)


def test_velocity_direction_calculation():
    """Test velocity direction calculation"""
    # Test cases: (vx, vy, expected_direction_degrees)
    test_cases = [
        (1.0, 0.0, 0.0),  # East
        (0.0, 1.0, 270.0),  # North (note: -vy in atan2)
        (-1.0, 0.0, 180.0),  # West
        (0.0, -1.0, 90.0),  # South
        (1.0, 1.0, 315.0),  # Northeast
        (-1.0, 1.0, 225.0),  # Northwest
        (1.0, -1.0, 45.0),  # Southeast
        (-1.0, -1.0, 135.0),  # Southwest
    ]
    for vx, vy, expected_deg in test_cases:
        # Calculate direction as in solve_h
        vdir = atan2(-vy, vx) * 180.0 / pi
        vdir = vdir + 360.0 * (vdir < 0)
        assert vdir == pytest.approx(expected_deg)


def test_vectorizable_velocity_calculation():
    """Test that vectorizable velocity calculation gives correct results"""
    eps = 1e-12
    # Test cases: (flow, flow_depth, expected_velocity)
    test_cases = [
        (2.0, 1.0, 2.0),  # Normal case
        (0.0, 1.0, 0.0),  # No flow
        (2.0, 0.0, 0.0),  # No depth (should be zero after masking)
        (2.0, -0.1, 0.0),  # Negative depth (should be zero after masking)
        (5.0, 2.5, 2.0),  # Another normal case
    ]
    for q, hf, expected_v in test_cases:
        # Original conditional approach
        if hf <= 0.0:
            v_original = 0.0
        else:
            v_original = q / hf

        # Optimized branchless approach
        v_optimized = q / max(hf, eps) * (hf > 0)
        assert v_original == pytest.approx(v_optimized)
        assert v_optimized == pytest.approx(expected_v)


def test_solve_h_uses_dx_and_dy_separately_in_flow_divergence():
    """The water-depth update must use the x and y cell sizes independently."""
    shape = (5, 5)
    dtype = np.float64

    arr_ext = np.zeros(shape, dtype=dtype)
    arr_qe = np.zeros(shape, dtype=dtype)
    arr_qs = np.zeros(shape, dtype=dtype)
    arr_bct = np.zeros(shape, dtype=np.uint8)
    arr_bcv = np.zeros(shape, dtype=dtype)
    arr_h = np.zeros(shape, dtype=dtype)
    arr_hmax = np.zeros(shape, dtype=dtype)
    arr_hfix = np.zeros(shape, dtype=dtype)
    arr_herr = np.zeros(shape, dtype=dtype)
    arr_hfe = np.zeros(shape, dtype=dtype)
    arr_hfs = np.zeros(shape, dtype=dtype)
    arr_v = np.zeros(shape, dtype=dtype)
    arr_vdir = np.zeros(shape, dtype=dtype)
    arr_vmax = np.zeros(shape, dtype=dtype)
    arr_fr = np.zeros(shape, dtype=dtype)

    center = (2, 2)
    arr_h[center] = 1.0
    arr_hmax[center] = 1.0

    qw = 7.0
    qe = 1.0
    qn = 6.0
    qs = 2.0
    dx = 4.0
    dy = 2.0
    dt = 0.5
    g = 9.81

    arr_qe[2, 1] = qw
    arr_qe[2, 2] = qe
    arr_qs[1, 2] = qn
    arr_qs[2, 2] = qs

    solve_h(
        arr_ext=arr_ext,
        arr_qe=arr_qe,
        arr_qs=arr_qs,
        arr_bct=arr_bct,
        arr_bcv=arr_bcv,
        arr_h=arr_h,
        arr_hmax=arr_hmax,
        arr_hfix=arr_hfix,
        arr_herr=arr_herr,
        arr_hfe=arr_hfe,
        arr_hfs=arr_hfs,
        arr_v=arr_v,
        arr_vdir=arr_vdir,
        arr_vmax=arr_vmax,
        arr_fr=arr_fr,
        dx=dx,
        dy=dy,
        dt=dt,
        g=g,
    )

    expected_h = 1.0 + dt * (((qw - qe) / dx) + ((qn - qs) / dy))
    swapped_dims_h = 1.0 + dt * (((qw - qe) / dy) + ((qn - qs) / dx))

    assert arr_h[center] == pytest.approx(expected_h)
    assert arr_hmax[center] == pytest.approx(expected_h)
    assert not np.isclose(arr_h[center], swapped_dims_h)


class TestWaterDepthFunction:
    """Integration tests for flow functions with optimized calculations"""

    def setup_method(self):
        """Set up test arrays"""
        self.shape = (5, 5)
        self.dtype = np.float64

        # Create test arrays
        self.arr_ext = np.zeros(self.shape, dtype=self.dtype)
        self.arr_qe = np.ones(self.shape, dtype=self.dtype) * 0.5
        self.arr_qs = np.ones(self.shape, dtype=self.dtype) * 0.3
        self.arr_bct = np.zeros(self.shape, dtype=np.uint8)
        self.arr_bcv = np.zeros(self.shape, dtype=self.dtype)
        self.arr_h = np.ones(self.shape, dtype=self.dtype) * 0.1
        self.arr_hmax = np.ones(self.shape, dtype=self.dtype) * 0.1
        self.arr_hfix = np.zeros(self.shape, dtype=self.dtype)
        self.arr_herr = np.zeros(self.shape, dtype=self.dtype)
        self.arr_hfe = np.ones(self.shape, dtype=self.dtype) * 0.05
        self.arr_hfs = np.ones(self.shape, dtype=self.dtype) * 0.05
        self.arr_v = np.zeros(self.shape, dtype=self.dtype)
        self.arr_vdir = np.zeros(self.shape, dtype=self.dtype)
        self.arr_vmax = np.zeros(self.shape, dtype=self.dtype)
        self.arr_fr = np.zeros(self.shape, dtype=self.dtype)

        # Parameters
        self.dx = 1.0
        self.dy = 1.0
        self.dt = 0.1
        self.g = 9.81

    def test_solve_h_velocity_calculations(self):
        """Test that solve_h produces reasonable velocity calculations"""
        # Run solve_h
        solve_h(
            arr_ext=self.arr_ext,
            arr_qe=self.arr_qe,
            arr_qs=self.arr_qs,
            arr_bct=self.arr_bct,
            arr_bcv=self.arr_bcv,
            arr_h=self.arr_h,
            arr_hmax=self.arr_hmax,
            arr_hfix=self.arr_hfix,
            arr_herr=self.arr_herr,
            arr_hfe=self.arr_hfe,
            arr_hfs=self.arr_hfs,
            arr_v=self.arr_v,
            arr_vdir=self.arr_vdir,
            arr_vmax=self.arr_vmax,
            arr_fr=self.arr_fr,
            dx=self.dx,
            dy=self.dy,
            dt=self.dt,
            g=self.g,
        )

        # Check that velocities are reasonable
        # Interior cells should have non-zero velocities
        interior_v = self.arr_v[1:-1, 1:-1]
        assert np.all(interior_v >= 0), "Velocities should be non-negative"
        assert np.any(interior_v > 0), "Some interior velocities should be positive"

        # Check that Froude numbers are reasonable
        interior_fr = self.arr_fr[1:-1, 1:-1]
        assert np.all(interior_fr >= 0), "Froude numbers should be non-negative"
        assert np.all(interior_fr < 15), "Froude numbers should be reasonable"

        # Check that velocity directions are in valid range [0, 360)
        interior_vdir = self.arr_vdir[1:-1, 1:-1]
        assert np.all(interior_vdir >= 0), "Velocity directions should be >= 0"
        assert np.all(interior_vdir < 360), "Velocity directions should be < 360"

    def test_solve_h_with_zero_flow_depths(self):
        """Test solve_h behavior with zero flow depths"""
        # Set some flow depths to zero
        self.arr_hfe[2, 2] = 0.0
        self.arr_hfs[2, 2] = 0.0

        # Run solve_h
        solve_h(
            arr_ext=self.arr_ext,
            arr_qe=self.arr_qe,
            arr_qs=self.arr_qs,
            arr_bct=self.arr_bct,
            arr_bcv=self.arr_bcv,
            arr_h=self.arr_h,
            arr_hmax=self.arr_hmax,
            arr_hfix=self.arr_hfix,
            arr_herr=self.arr_herr,
            arr_hfe=self.arr_hfe,
            arr_hfs=self.arr_hfs,
            arr_v=self.arr_v,
            arr_vdir=self.arr_vdir,
            arr_vmax=self.arr_vmax,
            arr_fr=self.arr_fr,
            dx=self.dx,
            dy=self.dy,
            dt=self.dt,
            g=self.g,
        )

        # Should not crash and should produce finite results
        assert np.all(np.isfinite(self.arr_v)), "All velocities should be finite"
        assert np.all(np.isfinite(self.arr_fr)), "All Froude numbers should be finite"
        assert np.all(np.isfinite(self.arr_vdir)), "All velocity directions should be finite"


class TestFixedWaterLevel:
    """Test if the boundary condition type 4 (fixed water level) is properly applied"""

    def setup_method(self):
        """Set up test arrays"""
        self.shape = (5, 5)
        self.dtype = np.float64

        # Create test arrays
        self.arr_ext = np.zeros(self.shape, dtype=self.dtype)
        self.arr_qe = np.zeros(self.shape, dtype=self.dtype)
        self.arr_qs = np.zeros(self.shape, dtype=self.dtype)
        self.arr_herr = np.zeros(self.shape, dtype=self.dtype)
        self.arr_hfe = np.ones(self.shape, dtype=self.dtype) * 0.05
        self.arr_hfs = np.ones(self.shape, dtype=self.dtype) * 0.05
        self.arr_v = np.zeros(self.shape, dtype=self.dtype)
        self.arr_vdir = np.zeros(self.shape, dtype=self.dtype)
        self.arr_vmax = np.zeros(self.shape, dtype=self.dtype)
        self.arr_fr = np.zeros(self.shape, dtype=self.dtype)

        # Parameters
        self.dx = 1.0
        self.dy = 1.0
        self.dt = 0.1
        self.g = 9.81

        # fixed boundary on center cell
        bct_values = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 4, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        self.arr_bct = np.array(bct_values, dtype=np.uint8)
        assert self.shape == self.arr_bct.shape

    def test_adding_water(self):
        bcv_values = [
            [0, 0, 0.0, 0, 0],
            [0, 0, 0.0, 0, 0],
            [0, 0, 1.5, 0, 0],
            [0, 0, 0.0, 0, 0],
            [0, 0, 0.0, 0, 0],
        ]
        arr_bcv = np.array(bcv_values, dtype=self.dtype)
        arr_h = np.ones(self.shape, dtype=self.dtype)
        arr_hmax = np.ones(self.shape, dtype=self.dtype)
        arr_hfix = np.zeros(self.shape, dtype=self.dtype)
        assert arr_bcv.shape == self.shape

        solve_h(
            arr_ext=self.arr_ext,
            arr_qe=self.arr_qe,
            arr_qs=self.arr_qs,
            arr_bct=self.arr_bct,
            arr_bcv=arr_bcv,
            arr_h=arr_h,
            arr_hmax=arr_hmax,
            arr_hfix=arr_hfix,
            arr_herr=self.arr_herr,
            arr_hfe=self.arr_hfe,
            arr_hfs=self.arr_hfs,
            arr_v=self.arr_v,
            arr_vdir=self.arr_vdir,
            arr_vmax=self.arr_vmax,
            arr_fr=self.arr_fr,
            dx=self.dx,
            dy=self.dy,
            dt=self.dt,
            g=self.g,
        )
        assert np.max(arr_hmax) == pytest.approx(1.5)
        assert np.sum(arr_hfix) == pytest.approx(0.5)
        assert np.max(arr_h) == pytest.approx(1.5)
        assert np.min(arr_h) == pytest.approx(1.0)

    def test_removing_water(self):
        bcv_values = [
            [0, 0, 0.0, 0, 0],
            [0, 0, 0.0, 0, 0],
            [0, 0, 0.5, 0, 0],
            [0, 0, 0.0, 0, 0],
            [0, 0, 0.0, 0, 0],
        ]
        arr_bcv = np.array(bcv_values, dtype=self.dtype)
        arr_h = np.ones(self.shape, dtype=self.dtype)
        arr_hmax = np.ones(self.shape, dtype=self.dtype)
        arr_hfix = np.zeros(self.shape, dtype=self.dtype)
        assert arr_bcv.shape == self.shape

        solve_h(
            arr_ext=self.arr_ext,
            arr_qe=self.arr_qe,
            arr_qs=self.arr_qs,
            arr_bct=self.arr_bct,
            arr_bcv=arr_bcv,
            arr_h=arr_h,
            arr_hmax=arr_hmax,
            arr_hfix=arr_hfix,
            arr_herr=self.arr_herr,
            arr_hfe=self.arr_hfe,
            arr_hfs=self.arr_hfs,
            arr_v=self.arr_v,
            arr_vdir=self.arr_vdir,
            arr_vmax=self.arr_vmax,
            arr_fr=self.arr_fr,
            dx=self.dx,
            dy=self.dy,
            dt=self.dt,
            g=self.g,
        )
        assert np.max(arr_hmax) == pytest.approx(1.0)
        assert np.sum(arr_hfix) == pytest.approx(-0.5)
        assert np.max(arr_h) == pytest.approx(1.0)
        assert np.min(arr_h) == pytest.approx(0.5)
