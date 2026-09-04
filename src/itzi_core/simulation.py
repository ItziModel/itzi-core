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
import io
import logging
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Self

import numpy as np

from itzi_core.array_definitions import ARRAY_DEFINITIONS, ArrayCategory
from itzi_core.compute import rastermetrics
from itzi_core.data_containers import (
    ContinuityData,
    DrainageNodeCouplingData,
    HotstartSimulationState,
    SimulationConfig,
    SimulationData,
)
from itzi_core.hotstart import HotstartWriter
from itzi_core.itzi_error import DtError, MassBalanceError, NullError
from itzi_core.simulation_schedule import SimulationSchedule
from itzi_core.timed_inputs import TimedInputManager

if TYPE_CHECKING:
    from itzi_core.data_containers import DrainageNodeCouplingData, SimulationConfig
    from itzi_core.drainage import DrainageSimulation
    from itzi_core.hydrology import Hydrology
    from itzi_core.providers.domain_data import DomainData
    from itzi_core.rasterdomain import RasterDomain
    from itzi_core.report import Report
    from itzi_core.surfaceflow import SurfaceFlowSimulation
    from itzi_core.timed_array import TimedArraySource


logger = logging.getLogger(__name__)


class Simulation:
    """Main interface to manage a simulation advance, time-stepping,
    models orchestration and output generation."""

    def __init__(
        self,
        sim_config: SimulationConfig,
        domain_data: DomainData,
        raster_domain: RasterDomain,
        schedule: SimulationSchedule,
        timed_input_manager: TimedInputManager | None,
        hydrology_model: Hydrology,
        surface_flow: SurfaceFlowSimulation,
        drainage_model: DrainageSimulation | None,
        drainage_nodes_list: tuple[DrainageNodeCouplingData, ...],
        report: Report,
    ):
        self.sim_config = sim_config
        self.raster_domain = raster_domain
        self.schedule = schedule
        self.domain_data = domain_data
        self.timed_input_manager = timed_input_manager
        self.hydrology_model = hydrology_model
        self.drainage_model = drainage_model
        self.drainage_nodes_list = drainage_nodes_list
        self.surface_flow = surface_flow
        self.report = report

        # Mass balance error checking
        self.old_domain_volume = rastermetrics.calculate_total_volume(
            depth_array=self.raster_domain.get_padded("water_depth"),
            cell_surface_area=self.raster_domain.cell_area,
            padded=True,
        )
        self.continuity_data: ContinuityData = self.get_continuity_data()
        self.mass_balance_error_threshold = sim_config.surface_flow_parameters.max_error
        # A mapping between source array and the corresponding accumulation array
        self.accum_mapping: dict[str, str] = {
            arr_def.computes_from: arr_def.key
            for arr_def in ARRAY_DEFINITIONS
            if arr_def.computes_from is not None and ArrayCategory.ACCUMULATION in arr_def.category
        }
        self.accum_update_time: dict[str, datetime] = {
            accum: self.sim_time for accum in self.accum_mapping.values()
        }
        self._initialized = False
        self.node_id_to_loc: dict[str, tuple[int, int]] = {}
        if self.drainage_model:
            self.node_id_to_loc = {
                n.node_id: (n.row, n.col)
                for n in self.drainage_nodes_list
                if n.node_object.is_coupled() and n.row is not None and n.col is not None
            }

        if self.sim_config.hotstart_config:
            self.hotstart_step: timedelta = self.sim_config.hotstart_config.wallclock_step
            self.hotstart_filename = self.sim_config.hotstart_config.save_file_name
            self.last_hotstart: datetime = datetime.now(tz=UTC)

        # Grid spacing (for BMI)
        self.spacing = (self.raster_domain.dy, self.raster_domain.dx)
        # time step counter
        self.time_steps_counters: dict[str, int] = {
            "since_start": 0,
            "since_last_report": 0,
        }

    @property
    def sim_time(self) -> datetime:
        return self.schedule.now

    @property
    def dt(self) -> timedelta:
        return self.schedule.dt

    @property
    def next_ts(self) -> Mapping[str, datetime]:
        return self.schedule.deadlines

    @property
    def nextstep(self) -> datetime:
        return self.schedule.nextstep

    @property
    def start_time(self) -> datetime:
        return self.schedule.start_time

    @property
    def end_time(self) -> datetime:
        return self.schedule.end_time

    @property
    def timed_arrays(self) -> dict[str, TimedArraySource] | None:
        if self.timed_input_manager is None:
            return None
        return self.timed_input_manager.timed_arrays

    def initialize(self) -> Self:
        """Record the initial stage of the simulation, before time-stepping."""
        self.old_domain_volume = rastermetrics.calculate_total_volume(
            depth_array=self.raster_domain.get_padded("water_depth"),
            cell_surface_area=self.raster_domain.cell_area,
            padded=True,
        )

        self._update_maximum("water_depth", "hmax")
        self._update_maximum("v", "vmax")

        for arr_key in self.accum_mapping:
            self._update_accum_array(arr_key, self.sim_time)
        self.continuity_data = self.get_continuity_data()
        # Pass data to the reporting module
        self.report.start(
            self.drainage_model.get_drainage_network_topology() if self.drainage_model else None
        )
        self.report.step(self._build_simulation_data(self.sim_time, 0))

        # d. Reset accumulators
        self.raster_domain.reset_accumulations()
        for key in self.accum_update_time:
            self.accum_update_time[key] = self.sim_time
        self._initialized = True
        return self

    def update(self) -> Self:
        """Advance one physical interval and report it at the interval end."""
        step_start: datetime = self.sim_time
        if step_start >= self.end_time:
            raise ValueError("Simulation has reached its configured end time")

        if step_start == self.schedule.deadline("hydrology"):
            self.hydrology_model.solve_dt()
            self.schedule.advance_event("hydrology", self.hydrology_model.dt)
            self.hydrology_model.step()
        if step_start == self.schedule.deadline("drainage") and self.drainage_model:
            self.drainage_model.step()
            self._apply_drainage_coupling()
            self.schedule.advance_event("drainage", self.drainage_model.dt)

        # Choose the current interval width from the state at step_start
        try:
            self.surface_flow.solve_dt()
        except DtError as e:
            raise DtError(f"{step_start}: Time-step computation error detected in simulation: {e}")
        step_end = self.schedule.select_step_end(step_start + self.surface_flow.dt)
        is_final_ts = step_end == self.end_time
        is_record_due = step_end == self.schedule.deadline("record")
        should_write_report = is_record_due or is_final_ts
        is_vdir_requested = self.report.out_map_names.get("vdir") is not None
        is_froude_requested = self.report.out_map_names.get("froude") is not None
        compute_vdir = should_write_report and is_vdir_requested
        compute_froude = should_write_report and is_froude_requested

        # surface flow #
        # update arrays of infiltration, rainfall etc.
        self.raster_domain.update_ext_array()
        # force time-step to be the general time-step
        try:
            self.surface_flow.dt = self.schedule.dt
        except DtError as e:
            raise DtError(f"{step_start}: Time-step errors detected in simulation: {e}")
        # surface_flow.step() raise NullError in case of NaN/NULL cell
        # if this happen, stop simulation
        try:
            self.surface_flow.step(
                compute_vdir=compute_vdir,
                compute_froude=compute_froude,
            )
        except NullError:
            raise NullError(f"{step_start}: Null value detected in simulation")

        # Align timed inputs to the interval end before closing and reporting it
        # under that time label. Due submodels will consume that label on the next update cycle.
        if self.timed_input_manager is not None:
            updates, next_input = self.timed_input_manager.read_at(step_end)
            for array_key, array in updates:
                self.set_array(array_key, array, step_end)
            self.schedule.set_deadline("input", next_input)

        # Update accumulation arrays
        for arr_key in self.accum_mapping:
            self._update_accum_array(arr_key, step_end)

        # Compute continuity error every x time steps
        steps_since_start = self.time_steps_counters["since_start"] + 1
        steps_since_report = self.time_steps_counters["since_last_report"] + 1
        is_first_ts = step_start == self.start_time
        is_ts_over_threshold = steps_since_report % 200 == 0
        is_error_comp_due = is_first_ts or is_ts_over_threshold or should_write_report
        if is_error_comp_due:
            self.continuity_data = self.get_continuity_data()

        # Reporting last to get simulated values #
        if should_write_report:
            logger.debug(f"{step_end}: Writing output maps...")
            self.report.start(
                self.drainage_model.get_drainage_network_topology()
                if self.drainage_model
                else None
            )
            self.report.step(
                self._build_simulation_data(
                    sim_time=step_end,
                    time_steps_counter=steps_since_report,
                )
            )

            self.old_domain_volume = copy.deepcopy(self.continuity_data.new_domain_vol)
            self.raster_domain.reset_accumulations()
            for key in self.accum_update_time:
                self.accum_update_time[key] = step_end
            if is_record_due:
                self.schedule.advance_event("record", self.report.dt)
            steps_since_report = 0

        error_threshold_exceeded = (
            self.continuity_data.created_volume_ratio > self.mass_balance_error_threshold
        )
        if is_error_comp_due and error_threshold_exceeded:
            raise MassBalanceError(
                f"{step_end}: "
                f"Created volume ratio {self.continuity_data.created_volume_ratio:.2f} "
                f"exceeds threshold {self.mass_balance_error_threshold:.2f}."
            )

        # Reset time and counters for next time-step
        self.schedule.commit_step(step_end)
        self.time_steps_counters["since_start"] = steps_since_start
        self.time_steps_counters["since_last_report"] = steps_since_report

        # Save hotstart
        if self.sim_config.hotstart_config:
            wall_time_now: datetime = datetime.now(UTC)
            elapsed: timedelta = wall_time_now - self.last_hotstart
            if elapsed >= self.hotstart_step:
                hotstart_bytes: io.BytesIO = self.create_hotstart()
                with open(self.hotstart_filename, "wb") as f:
                    f.write(hotstart_bytes.getbuffer())
                self.last_hotstart = wall_time_now

        return self

    def update_until(self, then: timedelta) -> Self:
        """Run the simulation until a time in seconds after start_time"""
        end_time = self.start_time + then
        if end_time <= self.sim_time:
            raise ValueError("End time must be superior to current time")
        if end_time > self.end_time:
            raise ValueError("End time must not exceed the configured simulation end time")
        with self.schedule.stop_at(end_time):
            while self.sim_time < end_time:
                self.update()
        # Make sure everything went well
        assert self.sim_time == end_time
        return self

    def finalize(self) -> None:
        """Flush already-written results and close runtime resources."""
        # The last interval is reported by update() when its end lands on end_time.
        self.report.end()
        if self.drainage_model:
            self.drainage_model.close()

    def _apply_drainage_coupling(self) -> None:
        """Update the drainage exchange array from the current time label state."""
        assert self.drainage_model is not None
        surface_states = {}
        cell_area = self.raster_domain.cell_area
        arr_z = self.raster_domain.get_array("dem")
        arr_h = self.raster_domain.get_array("water_depth")
        for node_id, (row, col) in self.node_id_to_loc.items():
            surface_states[node_id] = {"z": arr_z[row, col], "h": arr_h[row, col]}
        coupling_flows = self.drainage_model.apply_coupling_to_nodes(surface_states, cell_area)
        arr_qd = self.raster_domain.get_array("n_drain")
        for node_id, coupling_flow in coupling_flows.items():
            row, col = self.node_id_to_loc[node_id]
            arr_qd[row, col] = coupling_flow / cell_area

    def _build_simulation_data(
        self,
        sim_time: datetime,
        time_steps_counter: int,
    ) -> SimulationData:
        """Package the current domain state for reporting or provider finalization."""
        raw_arrays = {
            k: self.raster_domain.get_unmasked(k)
            for k in self.raster_domain.k_all
            if k not in self.raster_domain.k_accum
        }
        accumulation_arrays = {
            k: self.raster_domain.get_unmasked(k) for k in self.raster_domain.k_accum
        }
        if self.drainage_model:
            drainage_network_attributes = self.drainage_model.get_drainage_network_attributes()
        else:
            drainage_network_attributes = None
        return SimulationData(
            sim_time=sim_time,
            time_step=self.dt.total_seconds(),
            time_steps_counter=time_steps_counter,
            continuity_data=self.continuity_data,
            raw_arrays=raw_arrays,
            accumulation_arrays=accumulation_arrays,
            cell_dx=self.raster_domain.dx,
            cell_dy=self.raster_domain.dy,
            drainage_network_attributes=drainage_network_attributes,
        )

    def set_array(
        self,
        arr_id: str,
        arr: np.ndarray,
        sim_time: datetime | None = None,
    ) -> Self:
        """Set an array of the simulation domain."""
        current_time = self.sim_time if sim_time is None else sim_time
        if arr_id in ["inflow", "rain"]:
            self._update_accum_array(arr_id, current_time)
        self.raster_domain.update_array(arr_id, arr)
        if arr_id in {"water_depth", "water_surface_elevation"}:
            self._update_maximum("water_depth", "hmax")
        elif arr_id == "v":
            self._update_maximum("v", "vmax")
        if arr_id == "dem":
            self.surface_flow.update_flow_dir()
        return self

    def get_array(self, arr_id: str) -> np.ndarray:
        """Return an array through the BMI interface.

        Between reports, ``vdir`` and ``froude`` contain their values from the
        most recent report step when those outputs are enabled.
        """
        return self.raster_domain.get_array(arr_id)

    def get_continuity_data(self) -> ContinuityData:
        """Estimate numerical continuity error."""
        relative_volume_threshold = 1e-5
        cell_area = self.raster_domain.cell_area
        new_domain_vol = rastermetrics.calculate_total_volume(
            depth_array=self.raster_domain.get_padded("water_depth"),
            cell_surface_area=cell_area,
            padded=True,
        )
        volume_change = new_domain_vol - self.old_domain_volume
        created_volume = rastermetrics.calculate_total_volume(
            depth_array=self.raster_domain.get_padded("error_depth_accum"),
            cell_surface_area=cell_area,
            padded=True,
        )

        if new_domain_vol > 0:
            relative_volume_change = volume_change / new_domain_vol
        else:
            relative_volume_change = 0

        if created_volume == 0:
            created_volume_ratio = 0.0
        # Prevent returning artificially high error close to steady state
        elif abs(relative_volume_change) < relative_volume_threshold or volume_change == 0:
            created_volume_ratio = float("nan")
        else:
            created_volume_ratio = created_volume / volume_change

        return ContinuityData(
            new_domain_vol=new_domain_vol,
            volume_change=volume_change,
            created_volume=created_volume,
            created_volume_ratio=created_volume_ratio,
        )

    def _update_accum_array(self, k: str, sim_time: datetime) -> None:
        """Integrate a held rate into its current reporting-interval total.

        ``accum_mapping`` links each rate to a depth accumulator. In particular,
        ``computed_infiltration`` and ``capped_losses`` contain applied, not
        candidate, rates by the time this method runs. Multiplication by elapsed
        seconds produces accumulated depth; reporting later reduces that depth
        over active cells and multiplies it by cell area to obtain volume.
        """
        ak = self.accum_mapping[k]
        last_update = self.accum_update_time[ak]
        time_diff = (sim_time - last_update).total_seconds()
        if time_diff > 0:
            rate_array = self.raster_domain.get_padded(k)
            accum_array = self.raster_domain.get_padded(ak)
            rastermetrics.accumulate_rate_to_total(accum_array, rate_array, time_diff, padded=True)
            self.accum_update_time[ak] = sim_time

    def _update_maximum(self, value_key: str, maximum_key: str) -> None:
        """Synchronize a cumulative maximum with its current value array."""
        values = self.raster_domain.get_array(value_key)
        maximum = self.raster_domain.get_array(maximum_key)
        np.maximum(maximum, values, out=maximum)

    def create_hotstart(self) -> io.BytesIO:
        """Create a hotstart file with the current state of the simulation.

        Raises:
            RuntimeError: If called before initialize() has established valid state.
        """
        if not self._initialized:
            raise RuntimeError(
                "Cannot create hotstart: simulation has not been initialized. "
                "Call initialize() before creating a hotstart."
            )

        # Get SWMM hotstart bytes if drainage is enabled
        swmm_hotstart_bytes: bytes | None = None
        if self.drainage_model:
            swmm_hotstart: io.BytesIO = self.drainage_model.get_hotstart()
            swmm_hotstart_bytes = swmm_hotstart.getvalue()

        # Get raster domain state bytes
        raster_state: io.BytesIO = self.raster_domain.save_state()
        raster_state_bytes = raster_state.getvalue()

        # Build simulation state using Pydantic model.
        swmm_elapsed_time = self.drainage_model.elapsed_time if self.drainage_model else None
        simulation_state = HotstartSimulationState(
            sim_time=self.sim_time,
            dt=self.dt.total_seconds(),
            next_ts=self.schedule.snapshot_deadlines(),
            time_steps_counters=self.time_steps_counters,
            accum_update_time=dict(self.accum_update_time),
            old_domain_volume=self.old_domain_volume,
            swmm_elapsed_time=swmm_elapsed_time,
        )

        # Delegate archive creation to HotstartWriter
        return HotstartWriter.create(
            domain_data=self.domain_data,
            simulation_config=self.sim_config,
            simulation_state=simulation_state,
            raster_state_bytes=raster_state_bytes,
            swmm_hotstart_bytes=swmm_hotstart_bytes,
        )

    def restore_drainage_coupling_state(self) -> None:
        """Restore DrainageNode.coupling_flow and SWMM generated_inflow from n_drain.

        Must be called after raster domain state is restored from hotstart.

        Fixes two hotstart issues with the drainage coupling:

        1. DrainageNode.coupling_flow is always initialised to 0.0 on object creation.
           With RELAXATION_FACTOR=0.8 the first apply_coupling() call blends the new
           flow with 0 instead of the saved previous flow, producing wrong inflow to SWMM
           and a wrong n_drain value used by the surface-flow solver.

        2. SWMM's internal generated_inflow (set via pyswmm) is not persisted in the
           SWMM hotstart binary, so without this restoration the first swmm_step()
           runs with zero lateral inflow at each coupled junction.

        Both problems are fixed by reading the saved n_drain raster (restored from
        the hotstart zip) and using those values to initialise coupling_flow and to
        pre-inject the inflows into SWMM before the first drainage step.
        """
        if not self.drainage_model or not self.drainage_nodes_list:
            return
        arr_qd = self.raster_domain.get_array("n_drain")
        cell_area = self.raster_domain.cell_area
        for node_data in self.drainage_nodes_list:
            if node_data.row is None or node_data.col is None:
                continue  # node is not in the domain, skip
            node = node_data.node_object
            row, col = node_data.row, node_data.col
            coupling_flow = float(arr_qd[row, col]) * cell_area
            # Restore previous coupling_flow for correct relaxation/damping behaviour
            node.coupling_flow = coupling_flow
            # Pre-inject into SWMM so the first drainage step sees the correct inflow
            node.pyswmm_node.generated_inflow(np.float64(-coupling_flow))

    def restore_state(self, simulation_state: HotstartSimulationState) -> Self:
        """Restore simulation runtime state from hotstart data.

        This method restores scheduler and runtime state after raster state
        has been loaded. It must be called after the simulation object exists
        and after raster domain state has been restored.

        Args:
            simulation_state: Validated hotstart simulation state containing
                sim_time, dt, next_ts, counters, accum_update_time, and
                old_domain_volume.

        Returns:
            Self for method chaining.

        This method does NOT restore raster state; use RasterDomain.load_state() for that purpose.
        """
        self.schedule.restore(
            simulation_state.sim_time,
            timedelta(seconds=simulation_state.dt),
            simulation_state.next_ts,
        )

        # Restore time step counters
        self.time_steps_counters = dict(simulation_state.time_steps_counters)

        # Restore accumulation update timestamps
        self.accum_update_time = dict(simulation_state.accum_update_time)
        self._initialized = True

        # Restore old domain volume for continuity tracking
        self.old_domain_volume = simulation_state.old_domain_volume
        self.continuity_data = self.get_continuity_data()

        return self

    def reconcile_hotstart_resume(self, hotstart_config: SimulationConfig) -> Self:
        """Apply resume-time config changes allowed after hotstart restoration.
        This method is called after `restore_state()`,
        and reconciles the new user-provided config with the original config from the hotstart file."""
        if self.end_time != hotstart_config.end_time:
            self.schedule.set_deadline("end", self.end_time)
            self.schedule.set_deadline(
                "input", min(self.schedule.deadline("input"), self.end_time)
            )
            if not self.drainage_model:
                self.schedule.set_deadline("drainage", self.end_time)

        # Accumulators begin at the last archived report boundary, independently
        # of the cadence selected for the resumed run.
        self.report.last_step = self.schedule.deadline("record") - hotstart_config.record_step
        if self.report.dt != hotstart_config.record_step:
            self.schedule.set_deadline(
                "record", min(self.end_time, self.sim_time + self.report.dt)
            )

        if self.hydrology_model.dt != timedelta(seconds=hotstart_config.dtinf):
            self.schedule.set_deadline(
                "hydrology", min(self.end_time, self.sim_time + self.hydrology_model.dt)
            )

        return self
