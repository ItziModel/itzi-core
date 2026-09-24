"""
Copyright (C) 2015-2026 Laurent G. Courty

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

import math
from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np

from itzi_core.compute.partial_inertia_h import solve_h
from itzi_core.compute.partial_inertia_q import accumulate_boundary_fluxes, solve_q
from itzi_core.itzi_error import DtError, NullError

if TYPE_CHECKING:
    from itzi_core.data_containers import SurfaceFlowParameters
    from itzi_core.rasterdomain import RasterDomain


def estimate_surface_flow_timestep(
    cfl: float,
    min_cell_size: float,
    gravity: float,
    max_depth: float,
) -> float:
    """Estimate the surface-flow timestep in seconds."""
    return cfl * (min_cell_size / math.sqrt(gravity * max_depth))


class SurfaceFlowSimulation:
    """Surface flow simulation on staggered raster grid
    Accessed through step() methods
    By convention the flow is:
     - calculated at the East and South faces of each cell
     - positive from West to East and from North to South
    """

    def __init__(
        self,
        domain: RasterDomain,
        flow_params: SurfaceFlowParameters,
    ):
        self.dom = domain
        self.dtmax = flow_params.dtmax
        self.cfl = flow_params.cfl
        self.g = flow_params.g
        self.theta = flow_params.theta
        self.min_flow_depth = flow_params.hmin
        self.slope_threshold = flow_params.slope_threshold
        self.max_slope = flow_params.max_slope
        self.dx = domain.dx
        self.dy = domain.dy
        self.cell_surf = self.dx * self.dy
        # 1e-6 second
        self._dt_fudge = timedelta.resolution.total_seconds()
        self._dt: float = self._dt_fudge

    def update_flow_dir(self):
        """Deprecated."""
        return self

    def step(
        self,
        *,
        compute_vdir: bool = True,
        compute_froude: bool = True,
    ):
        """Run a full simulation time-step"""
        self.solve_q()
        self.update_h(
            compute_vdir=compute_vdir,
            compute_froude=compute_froude,
        )
        # in case of NaN/NULL cells, raise a NullError
        self.arr_err = np.isnan(self.dom.get_array("water_depth"))
        if np.any(self.arr_err):
            raise NullError
        self.swap_flow_arrays()
        return self

    def solve_dt(self):
        """Calculate the adaptive time-step
        The formula #15 in Almeida et al (2012) has been modified to
        accommodate non-square cells
        The time-step is limited by the maximum time-step dtmax.
        """
        maxh = float(np.amax(self.dom.get_array("water_depth")))
        min_dim = min(self.dx, self.dy)
        if maxh > 0:
            dt = estimate_surface_flow_timestep(self.cfl, min_dim, self.g, maxh)
            self._dt = float(min(self.dtmax, dt))
        else:
            self._dt = float(self.dtmax)
        if self._dt <= self._dt_fudge:
            raise DtError(f"Tiny computed dt ({self._dt}s)")
        return self

    @property
    def dt(self):
        return timedelta(seconds=self._dt)

    @dt.setter
    def dt(self, newdt: timedelta):
        """return an error if new dt is higher than current one or negative"""
        newdt_s = newdt.total_seconds()
        if newdt_s <= 0:
            raise DtError(f"dt must be positive, not {newdt_s}s")
        elif newdt_s > self._dt + self._dt_fudge:
            raise DtError(
                f"new dt cannot be longer than current one (old: {self._dt}s, new: {newdt_s}s)"
            )
        else:
            self._dt = newdt_s

    def update_h(
        self,
        *,
        compute_vdir: bool = True,
        compute_froude: bool = True,
    ):
        """Calculate new water depth, average velocity and Froude number"""
        solve_h(
            arr_ext=self.dom.get_padded("ext"),
            arr_qe=self.dom.get_padded("new_discharge_east"),
            arr_qs=self.dom.get_padded("new_discharge_south"),
            arr_bct=self.dom.get_padded("boundary_type"),
            arr_bcv=self.dom.get_padded("boundary_value"),
            arr_h=self.dom.get_padded("water_depth"),
            arr_hmax=self.dom.get_padded("max_water_depth"),
            arr_hfix=self.dom.get_padded("boundaries_accum"),
            arr_herr=self.dom.get_padded("error_depth_accum"),
            arr_hfe=self.dom.get_padded("flow_depth_east"),
            arr_hfs=self.dom.get_padded("flow_depth_south"),
            arr_v=self.dom.get_padded("flow_speed"),
            arr_vdir=self.dom.get_padded("flow_velocity_direction"),
            arr_vmax=self.dom.get_padded("max_flow_speed"),
            arr_fr=self.dom.get_padded("froude"),
            dx=self.dx,
            dy=self.dy,
            dt=self._dt,
            g=self.g,
            compute_vdir=compute_vdir,
            compute_froude=compute_froude,
        )
        assert not np.any(self.dom.get_array("water_depth") < 0)
        return self

    def solve_q(self):
        """Solve flow inside the domain using C/Cython function"""
        arr_qe_new = self.dom.get_padded("new_discharge_east")
        arr_qs_new = self.dom.get_padded("new_discharge_south")
        arr_boundaries_accum = self.dom.get_padded("boundaries_accum")
        solve_q(
            arr_z=self.dom.get_padded("ground_elevation"),
            arr_n=self.dom.get_padded("friction"),
            arr_h=self.dom.get_padded("water_depth"),
            arr_qe=self.dom.get_padded("old_discharge_east"),
            arr_qs=self.dom.get_padded("old_discharge_south"),
            arr_hfe=self.dom.get_padded("flow_depth_east"),
            arr_hfs=self.dom.get_padded("flow_depth_south"),
            arr_bctype=self.dom.get_padded("boundary_type"),
            arr_qe_new=arr_qe_new,
            arr_qs_new=arr_qs_new,
            dt=self._dt,
            dx=self.dx,
            dy=self.dy,
            g=self.g,
            theta=self.theta,
            hf_min=self.min_flow_depth,
            slope_threshold=self.slope_threshold,
            max_slope=self.max_slope,
        )
        accumulate_boundary_fluxes(
            arr_qe_new=arr_qe_new,
            arr_qs_new=arr_qs_new,
            arr_bcaccum=arr_boundaries_accum,
            dt=self._dt,
            dx=self.dx,
            dy=self.dy,
        )
        return self

    def swap_flow_arrays(self):
        """Swap flow arrays from calculated to input"""
        self.dom.swap_arrays("old_discharge_east", "new_discharge_east")
        self.dom.swap_arrays("old_discharge_south", "new_discharge_south")
        return self
