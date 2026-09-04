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

import io
import tempfile
from collections.abc import Iterable
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pyswmm
from numpy.typing import ArrayLike, DTypeLike

from itzi_core import infiltration
from itzi_core.array_definitions import ARRAY_DEFINITIONS, ArrayCategory
from itzi_core.const import InfiltrationModelType
from itzi_core.data_containers import DrainageNodeCouplingData
from itzi_core.drainage import CouplingTypes, DrainageLink, DrainageNode, DrainageSimulation
from itzi_core.hotstart import HotstartLoader
from itzi_core.hydrology import Hydrology
from itzi_core.itzi_error import HotstartError
from itzi_core.rasterdomain import RasterDomain
from itzi_core.report import Report
from itzi_core.simulation import Simulation
from itzi_core.simulation_schedule import SimulationSchedule
from itzi_core.surfaceflow import SurfaceFlowSimulation
from itzi_core.swmm_input_parser import SwmmInputParser
from itzi_core.timed_array import TimedArray
from itzi_core.timed_inputs import TimedInputManager

if TYPE_CHECKING:
    from itzi_core.data_containers import (
        HotstartSimulationState,
        SimulationConfig,
        SurfaceFlowParameters,
    )
    from itzi_core.providers.base import (
        MassBalanceOutputProvider,
        RasterInputProvider,
        RasterOutputProvider,
        VectorOutputProvider,
    )
    from itzi_core.providers.domain_data import DomainData


class SimulationBuilder:
    """Builder for creating Simulation objects with different provider configurations."""

    _STAGE_INPUT_KEYS = frozenset({"water_depth", "water_surface_elevation"})
    _HYDROLOGY_INPUT_KEYS = frozenset(
        {
            "rain",
            "losses",
            "infiltration",
            "effective_porosity",
            "capillary_pressure",
            "hydraulic_conductivity",
            "soil_water_content",
        }
    )
    _ALLOWED_RESUME_SURFACE_FLOW_CHANGES: ClassVar[set[str]] = {
        "cfl",
        "theta",
        "dtmax",
        "slope_threshold",
        "max_slope",
        "max_error",
    }

    def __init__(
        self,
        sim_config: SimulationConfig,
        arr_mask: ArrayLike,
        dtype: DTypeLike = np.float32,
    ) -> None:
        self.sim_config = sim_config
        self.arr_mask = np.asarray(arr_mask)
        self.dtype = dtype

        # Optional components (set via builder methods)
        self.raster_input_provider: RasterInputProvider | None = None
        self.domain_data: DomainData | None = None
        self.raster_output_provider: RasterOutputProvider | None = None
        self.vector_output_provider: VectorOutputProvider | None = None
        self.mass_balance_output_provider: MassBalanceOutputProvider | None = None

        # Hotstart data (set via with_hotstart)
        self.hotstart_loader: HotstartLoader | None = None

    def with_hotstart(
        self, hotstart_path_or_bytes: Path | str | io.BytesIO | bytes
    ) -> SimulationBuilder:
        """Load and store validated hotstart data for state restoration during build.

        This method loads and validates the hotstart archive but does not perform
        congruence checks against providers. Congruence validation happens during
        build() when all providers are available.

        Args:
            hotstart_path_or_bytes: Path to hotstart file, or hotstart data as
                BytesIO/bytes.

        Returns:
            self for method chaining.

        Raises:
            HotstartError: If the hotstart archive is invalid or corrupted.
        """
        if isinstance(hotstart_path_or_bytes, (Path, str)):
            self.hotstart_loader = HotstartLoader.from_file(hotstart_path_or_bytes)
        else:
            self.hotstart_loader = HotstartLoader.from_bytes(hotstart_path_or_bytes)
        return self

    def with_input_provider(self, provider: RasterInputProvider) -> SimulationBuilder:
        """Set the raster input provider."""
        self.raster_input_provider = provider
        self.domain_data = provider.get_domain_data()
        return self

    def with_domain_data(self, domain_data: DomainData) -> SimulationBuilder:
        """Set domain data directly (for memory simulations without input provider)."""
        self.domain_data = domain_data
        return self

    def with_raster_output_provider(self, provider: RasterOutputProvider) -> SimulationBuilder:
        """Set the raster output provider."""
        self.raster_output_provider = provider
        return self

    def with_vector_output_provider(self, provider: VectorOutputProvider) -> SimulationBuilder:
        """Set the vector output provider."""
        self.vector_output_provider = provider
        return self

    def with_mass_balance_output_provider(
        self,
        provider: MassBalanceOutputProvider,
    ) -> SimulationBuilder:
        """Set the provider used to persist mass-balance reports."""
        self.mass_balance_output_provider = provider
        return self

    def _validate_hotstart_congruence(self, hotstart_loader: HotstartLoader) -> None:
        """Validate hotstart data against builder configuration.

        This method performs congruence checks between the hotstart metadata
        and the current builder configuration. It must be called after all
        providers are attached but before any state mutation.

        Raises:
            HotstartError: If any congruence check fails.
        """

        hotstart_domain = hotstart_loader.get_domain_data()
        hotstart_config = hotstart_loader.get_simulation_config()
        hotstart_state = hotstart_loader.get_simulation_state()

        # Validate domain metadata
        self._validate_domain_congruence(hotstart_domain)

        # Validate mask compatibility
        self._validate_mask_congruence(hotstart_domain)

        # Validate drainage expectations
        self._validate_drainage_congruence(hotstart_config, hotstart_loader)

        # Validate resume-time configuration compatibility
        self._validate_resume_config_congruence(hotstart_config, hotstart_state)

    def _validate_resume_config_congruence(
        self,
        hotstart_config: SimulationConfig,
        hotstart_state: HotstartSimulationState,
    ) -> None:
        """Validate which runtime settings may change across a hotstart resume."""
        hotstart_sim_time = hotstart_state.sim_time

        if self.sim_config.start_time != hotstart_config.start_time:
            raise HotstartError(
                "Hotstart start_time mismatch: "
                f"current={self.sim_config.start_time}, "
                f"hotstart={hotstart_config.start_time}. "
                "Resume must keep the archived start_time unchanged."
            )

        # Keep this defensive check here even though SimulationConfig also validates
        # user input: model_copy(update=...) can bypass Pydantic validation in tests
        # and internal resume flows.
        if self.sim_config.record_step <= timedelta(0):
            raise HotstartError(
                f"Resume record_step must be positive, not {self.sim_config.record_step}"
            )

        if (
            self.sim_config.end_time != hotstart_config.end_time
            and self.sim_config.end_time <= hotstart_sim_time
        ):
            raise HotstartError(
                "Resume end_time must be strictly after the hotstart simulation time: "
                f"end_time={self.sim_config.end_time}, hotstart_sim_time={hotstart_sim_time}"
            )

        if self.sim_config.infiltration_model != hotstart_config.infiltration_model:
            raise HotstartError(
                "Hotstart infiltration model mismatch: "
                f"current={self.sim_config.infiltration_model}, "
                f"hotstart={hotstart_config.infiltration_model}"
            )

        changed_input_keys = self._changed_input_keys(hotstart_config)
        changed_stage_keys = changed_input_keys & self._STAGE_INPUT_KEYS
        if changed_stage_keys:
            raise HotstartError(
                "Hotstart input map changes are not supported for evolved stage inputs: "
                f"{', '.join(sorted(changed_stage_keys))}"
            )
        if changed_input_keys and self.raster_input_provider is None:
            raise HotstartError(
                "Hotstart changed input map names require an input provider for resume"
            )

        self._validate_surface_flow_parameter_congruence(hotstart_config.surface_flow_parameters)

    def _changed_input_keys(self, hotstart_config: SimulationConfig) -> set[str]:
        """Return canonical inputs whose configured external source changed on resume."""
        archived_names = hotstart_config.input_map_names
        resumed_names = self.sim_config.input_map_names
        return {
            key
            for key in archived_names.keys() | resumed_names.keys()
            if archived_names.get(key) != resumed_names.get(key)
        }

    def _validate_surface_flow_parameter_congruence(
        self,
        hotstart_surface_flow_parameters: SurfaceFlowParameters,
    ) -> None:
        """Validate the subset of surface-flow parameters that must not change."""
        current_surface_flow_parameters = self.sim_config.surface_flow_parameters

        for field_name in type(current_surface_flow_parameters).model_fields:
            if field_name in self._ALLOWED_RESUME_SURFACE_FLOW_CHANGES:
                continue

            current_value = getattr(current_surface_flow_parameters, field_name)
            hotstart_value = getattr(hotstart_surface_flow_parameters, field_name)

            if not np.isclose(current_value, hotstart_value):
                raise HotstartError(
                    "Surface flow parameter mismatch for "
                    f"{field_name}: current={current_value}, hotstart={hotstart_value}"
                )

    def _validate_domain_congruence(self, hotstart_domain: DomainData) -> None:
        """Validate that domain metadata matches between hotstart and builder."""
        assert self.domain_data is not None  # Already validated in build()

        # Check spatial bounds
        if not np.isclose(self.domain_data.north, hotstart_domain.north):
            raise HotstartError(
                f"Domain north mismatch: builder={self.domain_data.north}, "
                f"hotstart={hotstart_domain.north}"
            )
        if not np.isclose(self.domain_data.south, hotstart_domain.south):
            raise HotstartError(
                f"Domain south mismatch: builder={self.domain_data.south}, "
                f"hotstart={hotstart_domain.south}"
            )
        if not np.isclose(self.domain_data.east, hotstart_domain.east):
            raise HotstartError(
                f"Domain east mismatch: builder={self.domain_data.east}, "
                f"hotstart={hotstart_domain.east}"
            )
        if not np.isclose(self.domain_data.west, hotstart_domain.west):
            raise HotstartError(
                f"Domain west mismatch: builder={self.domain_data.west}, "
                f"hotstart={hotstart_domain.west}"
            )

        # Check dimensions
        if self.domain_data.rows != hotstart_domain.rows:
            raise HotstartError(
                f"Domain rows mismatch: builder={self.domain_data.rows}, "
                f"hotstart={hotstart_domain.rows}"
            )
        if self.domain_data.cols != hotstart_domain.cols:
            raise HotstartError(
                f"Domain cols mismatch: builder={self.domain_data.cols}, "
                f"hotstart={hotstart_domain.cols}"
            )

        # Check CRS
        if self.domain_data.crs_wkt != hotstart_domain.crs_wkt:
            raise HotstartError(
                "Domain CRS mismatch: builder and hotstart have different coordinate reference systems."
            )

    def _validate_mask_congruence(self, hotstart_domain: DomainData) -> None:
        """Validate that the builder mask is compatible with hotstart mask.

        The mask shape must match. The actual mask values will be validated
        during raster state restoration in RasterDomain.load_state().
        """
        expected_shape = hotstart_domain.shape
        builder_shape = self.arr_mask.shape

        if builder_shape != expected_shape:
            raise HotstartError(
                f"Mask shape mismatch: builder mask has shape {builder_shape}, "
                f"hotstart expects {expected_shape}"
            )

    def _validate_drainage_congruence(
        self,
        hotstart_config: SimulationConfig,
        hotstart_loader: HotstartLoader,
    ) -> None:
        """Validate drainage expectations match between hotstart and current config."""
        hotstart_has_drainage = hotstart_config.swmm_inp is not None
        builder_has_drainage = self.sim_config.swmm_inp is not None

        if hotstart_has_drainage and not builder_has_drainage:
            raise HotstartError(
                "Hotstart contains drainage state but current configuration has no drainage model"
            )

        if not hotstart_has_drainage and builder_has_drainage:
            raise HotstartError(
                "Hotstart has no drainage state but current configuration includes a drainage model"
            )

        # If both have drainage, check that SWMM hotstart bytes are present
        if hotstart_has_drainage and not hotstart_loader.has_swmm_hotstart():
            raise HotstartError(
                "Hotstart metadata indicates drainage but SWMM hotstart file is missing from archive"
            )

    def build(self) -> Simulation:
        """Build a simulation and explicitly load or prime provider-backed inputs."""
        domain_data = self.domain_data
        raster_output_provider = self.raster_output_provider
        vector_output_provider = self.vector_output_provider
        input_provider = self.raster_input_provider
        hotstart_loader = self.hotstart_loader

        if domain_data is None:
            raise ValueError("Domain data must be set via input provider or directly")
        if raster_output_provider is None or vector_output_provider is None:
            raise ValueError("Output providers are mandatory")

        # Validate hotstart congruence before building
        if hotstart_loader is not None:
            self._validate_hotstart_congruence(hotstart_loader)

        # Create timed arrays if input provider exists
        timed_arrays = None
        if input_provider is not None:
            timed_arrays = self._create_timed_arrays(input_provider, domain_data)

        # Create raster domain
        raster_domain = self._create_raster_domain(domain_data.cell_shape)

        # Create models
        infiltration_model = self._create_infiltration_model(raster_domain)
        hydrology_model = Hydrology(raster_domain, self.sim_config.dtinf, infiltration_model)
        surface_flow = SurfaceFlowSimulation(
            raster_domain, self.sim_config.surface_flow_parameters
        )

        # Create drainage with optional SWMM hotstart injection
        nodes_list, drainage_sim = self._create_drainage_simulation(domain_data)
        schedule = SimulationSchedule(
            self.sim_config.start_time,
            self.sim_config.end_time,
            self.sim_config.record_step,
            has_drainage=drainage_sim is not None,
        )
        timed_input_manager = None
        if timed_arrays is not None:
            timed_input_manager = TimedInputManager(
                timed_arrays,
                input_wse=bool(self.sim_config.input_map_names.get("water_surface_elevation")),
                end_time=self.sim_config.end_time,
                mask=raster_domain.mask,
            )

        # Create report
        report = Report(
            start_time=self.sim_config.start_time,
            temporal_type=self.sim_config.temporal_type,
            raster_output_provider=raster_output_provider,
            vector_output_provider=vector_output_provider,
            mass_balance_output_provider=self.mass_balance_output_provider,
            out_map_names=self.sim_config.output_map_names,
            dt=self.sim_config.record_step,
        )

        # Create simulation
        simulation = Simulation(
            self.sim_config,
            domain_data,
            raster_domain,
            schedule,
            timed_input_manager,
            hydrology_model,
            surface_flow,
            drainage_sim,
            nodes_list,
            report=report,
        )

        # Apply hotstart restore if hotstart data is present
        if hotstart_loader is not None:
            raster_state_buffer = hotstart_loader.get_raster_state_buffer()
            raster_domain.load_state(raster_state_buffer)

            simulation_state = hotstart_loader.get_simulation_state()
            simulation.restore_state(simulation_state)
            hotstart_config = hotstart_loader.get_simulation_config()
            changed_input_keys = self._changed_input_keys(hotstart_config)
            restored_input_deadline = simulation.schedule.deadline("input")
            restored_end_deadline = simulation.schedule.deadline("end")
            end_time_changed = self.sim_config.end_time != hotstart_config.end_time
            simulation.reconcile_hotstart_resume(hotstart_config)

            if timed_input_manager is None:
                if restored_input_deadline < restored_end_deadline:
                    raise HotstartError(
                        "Hotstart has a pending timed-input deadline but no input provider "
                        "is configured for resume"
                    )
                simulation.schedule.set_deadline("input", simulation.end_time)
            else:
                updates, primed_input_deadline = timed_input_manager.prepare_resume_at(
                    simulation.sim_time,
                    changed_input_keys,
                )
                for array_key, array in updates:
                    simulation.set_array(array_key, array, simulation.sim_time)
                if changed_input_keys & self._HYDROLOGY_INPUT_KEYS:
                    simulation.schedule.set_deadline("hydrology", simulation.sim_time)
                if (
                    not changed_input_keys
                    and not end_time_changed
                    and primed_input_deadline != restored_input_deadline
                ):
                    raise HotstartError(
                        "Hotstart timed-input boundary conflicts with the restored schedule: "
                        f"provider={primed_input_deadline}, restored={restored_input_deadline}"
                    )
                simulation.schedule.set_deadline("input", primed_input_deadline)

            simulation.restore_drainage_coupling_state()
        elif timed_input_manager is not None:
            updates, next_input = timed_input_manager.read_at(self.sim_config.start_time)
            for array_key, array in updates:
                simulation.set_array(array_key, array, self.sim_config.start_time)
            schedule.set_deadline("input", next_input)

        return simulation

    def _create_timed_arrays(
        self,
        input_provider: RasterInputProvider,
        domain_data: DomainData,
    ) -> dict[str, TimedArray]:
        """Create configured time-varying raster inputs."""
        timed_arrays = {}
        input_keys = [
            arr_def.key for arr_def in ARRAY_DEFINITIONS if ArrayCategory.INPUT in arr_def.category
        ]
        raster_shape = (domain_data.rows, domain_data.cols)

        def zeros_array_func() -> np.ndarray:
            return np.zeros(shape=raster_shape, dtype=self.dtype)

        for arr_key in input_keys:
            timed_arrays[arr_key] = TimedArray(arr_key, input_provider, zeros_array_func)
        return timed_arrays

    def _create_raster_domain(self, cell_shape) -> RasterDomain:
        """Create a raster domain."""
        try:
            raster_domain = RasterDomain(
                dtype=self.dtype,
                arr_mask=self.arr_mask,
                cell_shape=cell_shape,
            )
        except MemoryError:
            raise MemoryError("Cannot create the domain: Out of memory.")
        return raster_domain

    def _create_infiltration_model(
        self,
        raster_domain: RasterDomain,
    ) -> infiltration.InfiltrationModel:
        """Create an infiltration model based on configuration."""
        inf_model = self.sim_config.infiltration_model
        dtinf = self.sim_config.dtinf

        inf_class = {
            InfiltrationModelType.CONSTANT: infiltration.InfConstantRate,
            InfiltrationModelType.GREEN_AMPT: infiltration.InfGreenAmpt,
            InfiltrationModelType.NULL: infiltration.InfNull,
        }
        try:
            infiltration_model = inf_class[inf_model](raster_domain, dtinf)
        except KeyError:
            assert False, f"Unknow infiltration model: {inf_model}"
        return infiltration_model

    def _create_drainage_simulation(
        self,
        domain_data: DomainData,
    ) -> tuple[tuple[DrainageNodeCouplingData, ...], DrainageSimulation | None]:
        """Create drainage simulation components if SWMM input is provided.

        If hotstart data includes SWMM state, writes the SWMM hotstart bytes
        to a temporary file and passes it to DrainageSimulation for restoration.
        The temporary file is cleaned up after DrainageSimulation reads it.

        When resuming from a hotstart with time-varying SWMM inputs (hydrographs,
        patterns etc.), the SWMM simulation start datetime is advanced via
        setSimulationDateTime() so that those inputs are read from the correct point
        in their timeseries rather than restarting from T=0.
        """
        if not self.sim_config.swmm_inp:
            return (), None

        swmm_input_path = str(self.sim_config.swmm_inp)

        swmm_sim = pyswmm.Simulation(swmm_input_path)

        # Parse the .inp file for node/link coordinates and start datetime
        swmm_inp = SwmmInputParser(swmm_input_path)

        # Compute hotstart_start_datetime if resuming from a hotstart
        hotstart_start_datetime = None
        if self.hotstart_loader is not None:
            sim_state = self.hotstart_loader.get_simulation_state()
            swmm_elapsed = sim_state.swmm_elapsed_time
            if swmm_elapsed is not None and swmm_elapsed > 0:
                original_start = swmm_inp.get_start_datetime()
                if original_start is not None:
                    hotstart_start_datetime = original_start + timedelta(seconds=swmm_elapsed)

        # Create Node objects
        all_nodes = pyswmm.Nodes(swmm_sim)
        nodes_coors_dict = swmm_inp.get_nodes_id_as_dict()
        nodes_list = self._get_nodes(
            all_nodes,
            nodes_coors_dict,
            domain_data=domain_data,
            orifice_coeff=self.sim_config.orifice_coeff,
            free_weir_coeff=self.sim_config.free_weir_coeff,
            submerged_weir_coeff=self.sim_config.submerged_weir_coeff,
            g=self.sim_config.surface_flow_parameters.g,
        )

        # Create Link objects
        links_vertices_dict = swmm_inp.get_links_id_as_dict()
        links_list = get_links(pyswmm.Links(swmm_sim), links_vertices_dict, nodes_coors_dict)
        node_objects_only = tuple(i.node_object for i in nodes_list)

        # Handle SWMM hotstart injection if present
        if self.hotstart_loader is not None and self.hotstart_loader.has_swmm_hotstart():
            swmm_bytes = self.hotstart_loader.get_swmm_hotstart_bytes()
            if swmm_bytes is None:
                raise HotstartError("SWMM hotstart data is missing from the archive")
            # Create a temporary file for SWMM to read.
            # delete_on_close=False keeps the file after closing so SWMM can open it.
            with tempfile.NamedTemporaryFile(suffix=".hsf", delete_on_close=False) as tmp:
                tmp.write(swmm_bytes)
                hotstart_filename = tmp.name
                tmp.close()  # Allows SWMM to exclusively open the file
                drainage_sim = DrainageSimulation(
                    swmm_sim,
                    node_objects_only,
                    links_list,
                    hotstart_filename=hotstart_filename,
                    hotstart_start_datetime=hotstart_start_datetime,
                )
        else:
            drainage_sim = DrainageSimulation(swmm_sim, node_objects_only, links_list)

        return nodes_list, drainage_sim

    def _get_nodes(
        self,
        pswmm_nodes: Iterable[Any],
        nodes_coor_dict: dict[str, Any],
        domain_data: DomainData,
        orifice_coeff: float,
        free_weir_coeff: float,
        submerged_weir_coeff: float,
        g: float,
    ) -> tuple[DrainageNodeCouplingData, ...]:
        """Check if the drainage nodes are inside the region and can be coupled.
        A node without coordinates cannot be coupled.
        """
        nodes_list = []
        for pyswmm_node in pswmm_nodes:
            coors = nodes_coor_dict[pyswmm_node.nodeid]
            node = DrainageNode(
                node_object=pyswmm_node,
                coordinates=coors,
                coupling_type=CouplingTypes.NOT_COUPLED,
                orifice_coeff=orifice_coeff,
                free_weir_coeff=free_weir_coeff,
                submerged_weir_coeff=submerged_weir_coeff,
                g=g,
            )
            pixel = (
                None if coors is None else domain_data.coordinates_to_pixel(x=coors.x, y=coors.y)
            )
            if pixel is None:
                x_coor = None
                y_coor = None
                row = None
                col = None
            else:
                # Set node as coupled with no flow
                node.coupling_type = CouplingTypes.COUPLED_NO_FLOW
                x_coor = coors.x
                y_coor = coors.y
                row, col = pixel
            # populate list
            drainage_node_data = DrainageNodeCouplingData(
                node_id=pyswmm_node.nodeid, node_object=node, x=x_coor, y=y_coor, row=row, col=col
            )
            nodes_list.append(drainage_node_data)
        return tuple(nodes_list)


# Not in the main class to allow manual creation of a DrainageModel object for testing
def get_links(pyswmm_links, links_vertices_dict, nodes_coor_dict) -> tuple[DrainageLink, ...]:
    """Build drainage links from parsed SWMM geometry."""
    links_list = []
    for pyswmm_link in pyswmm_links:
        # Add nodes coordinates to the vertices list
        in_node_coor = nodes_coor_dict[pyswmm_link.inlet_node]
        out_node_coor = nodes_coor_dict[pyswmm_link.outlet_node]
        vertices = [in_node_coor]
        vertices.extend(links_vertices_dict[pyswmm_link.linkid].vertices)
        vertices.append(out_node_coor)
        link = DrainageLink(link_object=pyswmm_link, vertices=tuple(vertices))
        # add link to the list
        links_list.append(link)
    return tuple(links_list)
