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

from itzi_core.compute.hydrology import (
    commit_infiltration,
    infiltration_ga,
    infiltration_user,
)
from itzi_core.itzi_error import DtError


class InfiltrationModel:
    """Base lifecycle for infiltration models.

    ``step()`` writes a candidate rate to ``computed_infiltration``. Hydrology
    then caps infiltration together with user losses and overwrites that array
    with the applied rate. ``commit()`` may update model state from the applied
    rate; stateless models use this default no-op implementation.
    """

    def __init__(self, raster_domain, dt_inf):
        self.dom = raster_domain
        self.def_dt = dt_inf
        self._dt = self.def_dt

    def solve_dt(self):
        """time-step is by default equal to the default time-step"""
        self._dt = self.def_dt
        return self

    def commit(self):
        """Commit state derived from the applied rate, if the model has any."""
        return self

    @property
    def dt(self):
        """return the time-step as a timedelta"""
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


class InfConstantRate(InfiltrationModel):
    """Use a user-defined raster as the candidate infiltration rate.

    Availability is intentionally not checked here because infiltration and
    user losses must be capped together without giving either removal priority.
    """

    def step(self):
        """Copy the user-defined rate into the candidate array."""
        infiltration_user(
            arr_h=self.dom.get_array("water_depth"),
            arr_inf_in=self.dom.get_array("infiltration"),
            arr_inf_out=self.dom.get_array("computed_infiltration"),
            dt=self._dt,
        )
        return self


class InfGreenAmpt(InfiltrationModel):
    """Calculate candidate infiltration using the Green-Ampt formula.

    Candidate calculation reads cumulative applied infiltration but does not
    modify it. Hydrology first applies the combined water-removal cap, then
    ``commit()`` advances cumulative infiltration using only the accepted rate.
    """

    def step(self):
        """Calculate the Green-Ampt candidate without advancing model state."""
        infiltration_ga(
            arr_h=self.dom.get_array("water_depth"),
            arr_eff_por=self.dom.get_array("effective_porosity"),
            arr_pressure=self.dom.get_array("capillary_pressure"),
            arr_conduct=self.dom.get_array("hydraulic_conductivity"),
            arr_inf_amount=self.dom.get_array("total_infiltration"),
            arr_water_soil_content=self.dom.get_array("soil_water_content"),
            arr_inf_out=self.dom.get_array("computed_infiltration"),
            dt=self._dt,
        )
        return self

    def commit(self):
        """Integrate the applied rate into cumulative infiltration."""
        commit_infiltration(
            arr_inf_amount=self.dom.get_array("total_infiltration"),
            arr_applied_inf=self.dom.get_array("computed_infiltration"),
            dt=self._dt,
        )
        return self


class InfNull(InfiltrationModel):
    """No-op model used when infiltration is disabled."""

    def step(self):
        """Leave the zero infiltration-rate array unchanged."""
        return self
