"""Backend-independent, versioned simulation checkpoint data."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt, PositiveFloat

from itzi_core.const import InfiltrationModelType
from itzi_core.domain_data import DomainData


class HotstartSimulationState(BaseModel):
    """Runtime state to be restored from a checkpoint."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", revalidate_instances="always", allow_inf_nan=False
    )

    sim_time: datetime
    dt: NonNegativeFloat  # seconds
    next_ts: dict[str, datetime]
    time_steps_counters: dict[str, NonNegativeInt]
    accum_update_time: dict[str, datetime]
    old_domain_volume: float
    # SWMM elapsed time in seconds at the hotstart point.
    # Required to correctly initialise DrainageSimulation.elapsed_time so that
    # the first swmm_step() after hotstart restoration computes the correct _dt.
    swmm_elapsed_time: NonNegativeFloat | None = None


class SurfaceFlowResumeConfig(BaseModel):
    """Surface-flow setting that must remain unchanged across a resume."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    g: NonNegativeFloat


class HotstartResumeConfig(BaseModel):
    """Subset of SimulationConfig required to validate and reconcile a resume."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    original_start_time: datetime
    configured_end_time: datetime
    record_step: timedelta = Field(gt=timedelta(0))
    input_sources: dict[str, str]
    surface_flow: SurfaceFlowResumeConfig
    hydrology_step_seconds: PositiveFloat
    infiltration_model: InfiltrationModelType


class HotstartMetadata(BaseModel):
    """Versioned logical state metadata, independent of storage representation."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    state_version: Literal[1] = 1
    creation_date: datetime
    itzi_version: str
    domain_data: DomainData
    resume_config: HotstartResumeConfig
    simulation_state: HotstartSimulationState


class RasterHotstartState(BaseModel):
    """Exact raster state, including the one-cell solver boundary padding.

    The mask is unpadded. Arrays retain their original dtypes and masked-cell
    values. Storage implementations must preserve these values losslessly.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=True, revalidate_instances="always"
    )

    mask: np.ndarray
    padded_arrays: dict[str, np.ndarray]


class HotstartData(BaseModel):
    """Complete checkpoint exchanged between the simulation and its caller.

    Captured checkpoints own copies of all arrays and mutable metadata. Frozen
    models prevent field reassignment, but their arrays and mappings are mutable.
    Persistence and payload integrity checks are the caller's responsibility.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    metadata: HotstartMetadata
    raster_state: RasterHotstartState
    swmm_hotstart_bytes: bytes | None = None
