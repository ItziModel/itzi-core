"""
Copyright (C) 2025-2026 Laurent G. Courty

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

import copy
import math
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from itzi_core.array_definitions import ARRAY_DEFINITIONS, ArrayCategory
from itzi_core.compute import rastermetrics
from itzi_core.const import TemporalType
from itzi_core.data_containers import MassBalanceData, SimulationData

if TYPE_CHECKING:
    from itzi_core.data_containers import DrainageNetworkAttributes, DrainageNetworkTopology
    from itzi_core.providers.base import (
        MassBalanceOutputProvider,
        RasterOutputProvider,
        VectorOutputProvider,
    )


def calculate_closure(
    volume_change: float,
    volume_terms: tuple[float, ...],
    active_domain_area: float,
) -> tuple[float, float]:
    """Calculate the signed closure residual and its throughput-normalized error."""
    accounted_change = math.fsum(volume_terms)
    closure_residual = volume_change - accounted_change
    throughput = math.fsum(abs(term) for term in volume_terms)
    normalizer = max(abs(volume_change), throughput)
    absolute_tolerance = max(1e-12, active_domain_area * 1e-9)

    if normalizer <= absolute_tolerance:
        relative_closure_error = (
            0.0 if abs(closure_residual) < absolute_tolerance else float("nan")
        )
    else:
        relative_closure_error = abs(closure_residual) / normalizer
    return closure_residual, relative_closure_error


class Report:
    """In charge of results reporting and writing"""

    def __init__(
        self,
        start_time: datetime,
        temporal_type: TemporalType,
        raster_output_provider: RasterOutputProvider,
        vector_output_provider: VectorOutputProvider,
        mass_balance_output_provider: MassBalanceOutputProvider | None,
        out_map_names: dict,
        dt,
    ):
        self.temporal_type = temporal_type
        self.start_time = copy.copy(start_time)
        self.record_counter = 0
        self.raster_provider = raster_output_provider
        self.vector_provider = vector_output_provider
        # The saved map names, defined by the user
        self.out_map_names = out_map_names
        self.mass_balance_output_provider = mass_balance_output_provider
        # a dict containing lists of maps written to gis to be registered
        self.output_maplist = {k: [] for k in self.out_map_names}
        self.dt = dt
        self.last_step = copy.copy(start_time)
        self._drainage_topology_written = False

    def start(self, drainage_topology: DrainageNetworkTopology | None) -> Report:
        """Write drainage topology once before report records with attributes."""
        if drainage_topology is None:
            return self
        if self._drainage_topology_written:
            return self
        self.vector_provider.write_topology(drainage_topology)
        self._drainage_topology_written = True
        return self

    def step(self, simulation_data: SimulationData):
        """write results at given time-step"""
        if (
            simulation_data.drainage_network_attributes is not None
            and not self._drainage_topology_written
        ):
            raise RuntimeError("Drainage attributes cannot be written before topology.")
        sim_time = simulation_data.sim_time
        if self.temporal_type == TemporalType.RELATIVE:
            converted_sim_time = sim_time - self.start_time
        else:
            converted_sim_time = sim_time
        output_arrays = self.get_output_arrays(simulation_data)
        self.raster_provider.write_arrays(array_dict=output_arrays, sim_time=converted_sim_time)
        if self.mass_balance_output_provider is not None:
            self.write_mass_balance(simulation_data, converted_sim_time)
        drainage_attributes = simulation_data.drainage_network_attributes
        if drainage_attributes is not None:
            self.save_drainage_values(drainage_attributes, converted_sim_time)
        self.record_counter += 1
        self.last_step = copy.copy(sim_time)
        return self

    def end(self):
        """Finalize output providers after the last report has been written."""
        self.raster_provider.finalize()
        self.vector_provider.finalize()
        if self.mass_balance_output_provider is not None:
            self.mass_balance_output_provider.finalize()
        return self

    def get_output_arrays(self, data: SimulationData) -> dict[str, np.ndarray]:
        """Returns a dict of arrays to be written to the disk"""
        output_arrays = {}
        raw = data.raw_arrays
        accum_arrays = data.accumulation_arrays
        interval_s = (data.sim_time - self.last_step).total_seconds()
        cell_dx = data.cell_dx
        cell_dy = data.cell_dy
        cell_area = cell_dx * cell_dy

        # Iterate through the output maps requested by the user
        for arr_key in self.out_map_names:
            if self.out_map_names[arr_key] is None:
                continue

            # --- Direct raw arrays ---
            if arr_key in ["water_depth", "v", "vdir", "froude", "hmax", "vmax"]:
                if arr_key in raw:
                    output_arrays[arr_key] = raw[arr_key]
                continue  # go to next key

            # --- Calculated arrays ---
            if arr_key == "water_surface_elevation":
                output_arrays[arr_key] = rastermetrics.calculate_wse(
                    raw["water_depth"], raw["dem"]
                )
            elif arr_key == "qx":
                output_arrays[arr_key] = rastermetrics.calculate_flux(raw["qe_new"], cell_dy)
            elif arr_key == "qy":
                output_arrays[arr_key] = rastermetrics.calculate_flux(raw["qs_new"], cell_dx)
            elif arr_key == "created_volume":
                output_arrays[arr_key] = accum_arrays["error_depth_accum"] * cell_area

        # --- Averaged accumulation arrays ---
        if interval_s <= 0:
            interval_s = data.time_step

        accum_mapping = {
            arr_def.key: arr_def.computes_from
            for arr_def in ARRAY_DEFINITIONS
            if arr_def.computes_from is not None and ArrayCategory.OUTPUT in arr_def.category
        }
        for output_name, accum_key in accum_mapping.items():
            if self.out_map_names.get(output_name) and accum_key in accum_arrays:
                if accum_key in ["rainfall_accum", "infiltration_accum", "losses_accum"]:
                    conversion_factor = 1000 * 3600  # m/s to mm/h
                else:
                    conversion_factor = 1.0
                output_arrays[output_name] = rastermetrics.calculate_average_rate_from_total(
                    accum_arrays[accum_key], interval_s, conversion_factor
                )
        return output_arrays

    def write_mass_balance(self, data: SimulationData, converted_sim_time: datetime | timedelta):
        """Calculate mass balance and log it."""
        continuity_data = data.continuity_data
        # 1. Calculate all volumes using rastermetrics
        cell_area = data.cell_dx * data.cell_dy

        boundary_vol = rastermetrics.calculate_total_volume(
            data.accumulation_arrays["boundaries_accum"], cell_area
        )
        rain_vol = rastermetrics.calculate_total_volume(
            data.accumulation_arrays["rainfall_accum"], cell_area
        )
        infiltration_vol = rastermetrics.calculate_total_volume(
            data.accumulation_arrays["infiltration_accum"], cell_area
        )
        inflow_vol = rastermetrics.calculate_total_volume(
            data.accumulation_arrays["inflow_accum"], cell_area
        )
        losses_vol = rastermetrics.calculate_total_volume(
            data.accumulation_arrays["losses_accum"], cell_area
        )
        drain_net_vol = rastermetrics.calculate_total_volume(
            data.accumulation_arrays["drainage_network_accum"], cell_area
        )

        signed_volume_terms = (
            boundary_vol,
            rain_vol,
            -infiltration_vol,
            inflow_vol,
            -losses_vol,
            drain_net_vol,
            continuity_data.created_volume,
        )
        active_cells = np.count_nonzero(np.isfinite(data.raw_arrays["water_depth"]))
        closure_residual, relative_closure_error = calculate_closure(
            continuity_data.volume_change,
            signed_volume_terms,
            float(active_cells * cell_area),
        )

        # 3. Assemble data and log
        interval_s = (data.sim_time - self.last_step).total_seconds()
        if data.time_steps_counter > 0:
            average_timestep = interval_s / data.time_steps_counter
        else:
            average_timestep = float("nan")
        report_data = MassBalanceData(
            simulation_time=converted_sim_time,
            average_timestep=average_timestep,
            timesteps=data.time_steps_counter,
            boundary_volume=boundary_vol,
            rainfall_volume=rain_vol,
            infiltration_volume=-infiltration_vol,  # negative because it leaves the domain
            inflow_volume=inflow_vol,
            losses_volume=-losses_vol,  # negative because it leaves the domain
            drainage_network_volume=drain_net_vol,
            domain_volume=continuity_data.new_domain_vol,
            volume_change=continuity_data.volume_change,
            created_volume=continuity_data.created_volume,
            created_volume_ratio=continuity_data.created_volume_ratio,
            closure_residual=closure_residual,
            relative_closure_error=relative_closure_error,
        )
        provider = self.mass_balance_output_provider
        assert provider is not None
        provider.log(report_data)
        return self

    def save_drainage_values(
        self, drainage_attributes: DrainageNetworkAttributes, sim_time: datetime | timedelta
    ) -> Report:
        """Write drainage attributes for a simulation record."""
        if not self._drainage_topology_written:
            raise RuntimeError("Drainage attributes cannot be written before topology.")
        self.vector_provider.write_attributes(drainage_attributes, sim_time)
        return self
