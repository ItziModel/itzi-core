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

from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    field_validator,
)

from itzi_core.const import DefaultValues, InfiltrationModelType, TemporalType
from itzi_core.providers.domain_data import DomainData

if TYPE_CHECKING:
    from itzi_core.drainage import DrainageNode


class DrainageNodeCouplingData(BaseModel):
    """Store the translation between coordinates and array location for a given drainage node."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    node_id: str  # Name of the drainage node
    node_object: DrainageNode
    # Location in the coordinate system
    x: float | None
    y: float | None
    # Location in the array
    row: int | None
    col: int | None


class DrainageAttributes(BaseModel):
    """A base class for drainage data attributes."""

    model_config = ConfigDict(frozen=True)

    @classmethod
    def get_columns_definition(cls, cat_primary_key=True) -> list[tuple[str, str]]:
        """Return a list of tuples to create DB columns"""
        type_mapping = {str: "TEXT", int: "INT", float: "REAL"}
        db_columns_def = [("cat", "INTEGER PRIMARY KEY")]
        if not cat_primary_key:
            db_columns_def = []
        for field_name, field_info in cls.model_fields.items():
            db_field = (field_name, type_mapping[field_info.annotation])
            db_columns_def.append(db_field)
        return db_columns_def


class DrainageLinkAttributes(DrainageAttributes):
    link_id: str
    link_type: str
    flow: float
    depth: float
    volume: float
    inlet_offset: float
    outlet_offset: float
    froude: float


class DrainageLinkTopology(BaseModel):
    """Store the fixed topology of a drainage link."""

    model_config = ConfigDict(frozen=True)

    link_id: str
    start_node_id: str
    end_node_id: str
    vertices: None | tuple[tuple[float, float] | None, ...]


class DrainageNodeAttributes(DrainageAttributes):
    node_id: str
    node_type: str
    coupling_type: str
    coupling_flow: float
    inflow: float
    outflow: float
    lateral_inflow: float
    losses: float
    overflow: float
    depth: float
    head: float
    # crownElev: float
    crest_elevation: float
    invert_elevation: float
    initial_depth: float
    full_depth: float
    surcharge_depth: float
    ponding_area: float
    # degree: int
    volume: float
    full_volume: float


class DrainageNodeTopology(BaseModel):
    """Store the fixed topology of a drainage node."""

    model_config = ConfigDict(frozen=True)

    node_id: str
    coordinates: None | tuple[float, float]


class DrainageNetworkTopology(BaseModel):
    model_config = ConfigDict(frozen=True)

    nodes: tuple[DrainageNodeTopology, ...]
    links: tuple[DrainageLinkTopology, ...]


class DrainageNetworkAttributes(BaseModel):
    """Store the time-varying state of a drainage network."""

    model_config = ConfigDict(frozen=True)

    nodes: tuple[DrainageNodeAttributes, ...]
    links: tuple[DrainageLinkAttributes, ...]


class ContinuityData(BaseModel):
    """Store information about simulation continuity"""

    model_config = ConfigDict(frozen=True)

    new_domain_vol: float
    volume_change: float
    created_volume: float
    created_volume_ratio: float


class SimulationData(BaseModel):
    """Immutable data container for passing raw simulation state to Report.

    This is a pure data structure containing only the "raw ingredients"
    needed for a report. All report-specific calculations (e.g., WSE,
    average rates) are performed by the Report class itself.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    sim_time: datetime
    time_step: float  # time step duration
    time_steps_counter: int  # number of time steps since last update
    continuity_data: ContinuityData
    raw_arrays: dict[str, np.ndarray]
    accumulation_arrays: dict[str, np.ndarray]
    cell_dx: PositiveFloat  # cell size in east-west direction
    cell_dy: PositiveFloat  # cell size in north-south direction
    drainage_network_attributes: DrainageNetworkAttributes | None


class MassBalanceData(BaseModel):
    """Contains the fields written to the mass balance file"""

    model_config = ConfigDict(frozen=True)

    simulation_time: datetime | timedelta
    average_timestep: float
    timesteps: NonNegativeInt
    boundary_volume: float
    rainfall_volume: float
    infiltration_volume: float
    inflow_volume: float
    losses_volume: float
    drainage_network_volume: float
    domain_volume: float
    volume_change: float
    created_volume: float
    created_volume_ratio: float
    closure_residual: float
    relative_closure_error: float


class SurfaceFlowParameters(BaseModel):
    """Parameters for the surface flow model."""

    model_config = ConfigDict(frozen=True)

    hmin: NonNegativeFloat = DefaultValues.HFMIN
    cfl: PositiveFloat = Field(DefaultValues.CFL, ge=0.01, le=1)
    theta: NonNegativeFloat = Field(DefaultValues.THETA, ge=0, le=1)
    g: NonNegativeFloat = DefaultValues.G
    dtmax: PositiveFloat = DefaultValues.DTMAX
    slope_threshold: NonNegativeFloat = DefaultValues.SLOPE_THRESHOLD
    max_slope: NonNegativeFloat = DefaultValues.MAX_SLOPE
    max_error: PositiveFloat = DefaultValues.MAX_ERROR


class HotstartRunConfig(BaseModel):
    """Configuration to restart a simulation from a hotstart file."""

    wallclock_step: timedelta
    save_file_name: str | Path


class SimulationConfig(BaseModel):
    """Configuration data for a simulation run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Simulation times
    start_time: datetime
    end_time: datetime
    record_step: timedelta
    temporal_type: TemporalType
    # Hotstart config
    hotstart_config: HotstartRunConfig | None = None
    # Input and output raster maps
    input_map_names: dict[str, str]
    output_map_names: dict[str, str]
    # Surface flow parameters
    surface_flow_parameters: SurfaceFlowParameters
    # Hydrology parameters
    dtinf: PositiveFloat = DefaultValues.DTINF
    infiltration_model: InfiltrationModelType = InfiltrationModelType.NULL
    # Drainage parameters
    swmm_inp: Path | None = None
    drainage_output: str | None = None
    orifice_coeff: NonNegativeFloat = Field(DefaultValues.ORIFICE_COEFF, ge=0, le=1)
    free_weir_coeff: NonNegativeFloat = Field(DefaultValues.FREE_WEIR_COEFF, ge=0, le=1)
    submerged_weir_coeff: NonNegativeFloat = Field(DefaultValues.SUBMERGED_WEIR_COEFF, ge=0, le=1)

    @field_validator("input_map_names", "output_map_names", mode="before")
    @classmethod
    def remove_inactive_map_names(cls, value: object) -> object:
        """Normalize legacy null-valued map entries to omitted inactive entries."""
        if isinstance(value, dict):
            return {key: map_name for key, map_name in value.items() if map_name is not None}
        return value

    def as_str_dict(self) -> dict:
        """Convert the configuration to a dictionary with string representations."""
        raw_dict = self.model_dump()
        raw_dict["start_time"] = self.start_time.isoformat()
        raw_dict["end_time"] = self.end_time.isoformat()
        raw_dict["record_step"] = self.record_step.total_seconds()
        return raw_dict


class HotstartSimulationState(BaseModel):
    """Runtime state to be restored from a hotstart file."""

    model_config = ConfigDict(frozen=True)

    sim_time: datetime
    dt: float  # seconds
    next_ts: dict[str, datetime]
    time_steps_counters: dict[str, int]
    accum_update_time: dict[str, datetime]
    old_domain_volume: float
    # Hashes are computed by HotstartWriter and injected before serialization;
    # callers building the state before archive creation leave them as empty defaults.
    raster_domain_hash: str = ""
    swmm_hotstart_hash: str | None = None
    # SWMM elapsed time in seconds at the hotstart point.
    # Required to correctly initialise DrainageSimulation.elapsed_time so that
    # the first swmm_step() after hotstart restoration computes the correct _dt.
    swmm_elapsed_time: float | None = None


class HotstartMetadata(BaseModel):
    """Metadata schema for hotstart archive files.

    Provides a single source of truth for hotstart metadata structure,
    enabling validation during both creation and loading.
    """

    model_config = ConfigDict(frozen=True)

    creation_date: datetime
    itzi_version: str
    hotstart_version: int
    domain_data: DomainData
    simulation_config: SimulationConfig
    simulation_state: HotstartSimulationState
