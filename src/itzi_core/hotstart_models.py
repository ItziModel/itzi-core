"""Models defining the version 2 hotstart archive format."""

from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveFloat

from itzi_core.const import InfiltrationModelType
from itzi_core.domain_data import DomainData


class HotstartSimulationState(BaseModel):
    """Runtime state to be restored from a hotstart file."""

    model_config = ConfigDict(frozen=True)

    sim_time: datetime
    dt: float  # seconds
    next_ts: dict[str, datetime]
    time_steps_counters: dict[str, int]
    accum_update_time: dict[str, datetime]
    old_domain_volume: float
    # Hashes are computed during archive creation and injected before serialization;
    # callers building the state before archive creation leave them as empty defaults.
    raster_domain_hash: str = ""
    swmm_hotstart_hash: str | None = None
    # SWMM elapsed time in seconds at the hotstart point.
    # Required to correctly initialise DrainageSimulation.elapsed_time so that
    # the first swmm_step() after hotstart restoration computes the correct _dt.
    swmm_elapsed_time: float | None = None


class SurfaceFlowResumeConfig(BaseModel):
    """Surface-flow setting that must remain unchanged across a resume."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    g: NonNegativeFloat


class HotstartResumeConfig(BaseModel):
    """Subset of SimulationConfig required to validate and reconcile a resume."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    original_start_time: datetime
    configured_end_time: datetime
    record_step: timedelta
    input_sources: dict[str, str]
    surface_flow: SurfaceFlowResumeConfig
    hydrology_step_seconds: PositiveFloat
    infiltration_model: InfiltrationModelType


class HotstartMetadataV2(BaseModel):
    """Strict metadata schema for version 2 hotstart archives."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hotstart_version: Literal[2]
    creation_date: datetime
    itzi_version: str
    domain_data: DomainData
    resume_config: HotstartResumeConfig
    simulation_state: HotstartSimulationState
