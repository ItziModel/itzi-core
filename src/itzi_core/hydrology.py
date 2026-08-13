"""
Copyright (C) 2016-2026 Laurent G. Courty

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 2.1
of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.
"""

from datetime import timedelta

from itzi_core.compute.hydrology import apply_hydrology, infiltration_user
from itzi_core.itzi_error import DtError


class Hydrology:
    """Coordinate rainfall and water-removal rates for the surface solver.

    At each hydrology event, the infiltration model and user-loss input first
    populate candidate-rate arrays. The combined removals are capped against
    the water depth present at the start of the event and written back as
    applied rates. Rainfall is not included in the water available for removal.

    The resulting effective-precipitation rate is held for the surface solver,
    while the applied infiltration and loss rates are integrated separately for
    reporting. Stateful infiltration models are committed only after capping.
    """

    def __init__(self, raster_domain, dt, infiltration):
        self.dom = raster_domain
        self._dt = dt
        # an infiltration model object
        self.infiltration = infiltration

    def solve_dt(self):
        """time-step is by default equal to the default time-step"""
        self.infiltration.solve_dt()
        return self

    @property
    def dt(self):
        return timedelta(seconds=self._dt)

    @dt.setter
    def dt(self, newdt):
        """return an error if new dt is higher than current one"""
        newdt_s = newdt.total_seconds()
        fudge = timedelta.resolution.total_seconds()
        if newdt_s > self._dt + fudge:
            raise DtError("new dt cannot be longer than current one")
        else:
            self._dt = newdt_s
            self.infiltration.dt = newdt_s

    def step(self):
        """Calculate, apply, and commit rates for one hydrology event.

        This method does not update water depth directly. It updates held rate
        arrays that the surface solver applies over subsequent flow timesteps.
        The ordering is significant because Green-Ampt state and mass-balance
        accumulators must use the same infiltration rate that reaches the
        surface-depth equation.
        """
        # These first two calls populate candidates; apply_hydrology overwrites
        # both arrays with the proportionally capped, applied rates.
        self.infiltration.step()
        self.cap_losses()
        self.apply_hydrology()
        # Commit only after the combined cap so rejected Green-Ampt infiltration
        # is not added to the cumulative soil state.
        self.infiltration.commit()
        return self

    def cap_losses(self):
        """Populate the user-loss candidate array.

        The historical method name is retained, but the shared availability
        cap is applied later together with infiltration.
        """
        infiltration_user(
            arr_h=self.dom.get_array("water_depth"),
            arr_inf_in=self.dom.get_array("losses"),
            arr_inf_out=self.dom.get_array("capped_losses"),
            dt=self._dt,
        )

    def apply_hydrology(self):
        """Apply the combined water-removal cap and update effective precipitation.

        The compute kernel replaces ``computed_infiltration`` and
        ``capped_losses`` candidates with applied rates. ``eff_precip`` is rain
        minus the applied infiltration and user-loss rates; drainage and user
        inflow are combined with it later when the surface solver's external-rate
        array is assembled.
        """
        apply_hydrology(
            arr_rain=self.dom.get_array("rain"),
            arr_inf=self.dom.get_array("computed_infiltration"),
            arr_capped_losses=self.dom.get_array("capped_losses"),
            arr_h=self.dom.get_array("water_depth"),
            arr_eff_precip=self.dom.get_array("eff_precip"),
            dt=self._dt,
        )
        return self
