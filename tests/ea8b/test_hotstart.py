"""
Integration tests for hotstart functionality with the EA test case 8b.

The shared expensive simulation (EA8b with drainage, running from t=0 to t=3h20m)
is built once by the ``ea8b_simulation`` fixture in conftest.py. It captures structured
checkpoints at the split point and at the end for comparison with a resumed run.

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

from __future__ import annotations

import os
from datetime import timedelta

import numpy as np
import pytest

from itzi_core.const import TemporalType
from itzi_core.data_containers import SimulationConfig, SurfaceFlowParameters
from tests.ea8b.helpers import (
    EA8B_FINAL_ARRAY_ATOL,
    assert_matches_reference,
    build_resumed_simulation,
    drainage_data_to_coupling_series,
    get_reference_metrics,
)

pytestmark = pytest.mark.xarray


@pytest.mark.slow
def test_ea8b_hotstart_roundtrip(
    ea8b_simulation,
    ea8b_reference,
    ea8b_data,
    ea8b_temp_path,
    helpers,
):
    """Test that resuming from hotstart at the split point reproduces the full run.

    Verifies that:
    - Simulation time and raster state are correctly restored from hotstart
    - Resumed drainage results satisfy the XPSTORM acceptance criteria
    - Final raster arrays are close to the uninterrupted reference within
      deterministic tolerances

    The raster tolerances are intentionally higher than the usual exact/near-exact
    expectations because EA8b with drainage and ponding is not restart-exact after
    a SWMM hotstart resume.  The remaining differences are traced to SWMM's
    ponding-related state, but both uninterrupted and resumed runs stay within the
    accepted XPSTORM reference thresholds.
    """
    hotstart_split = ea8b_simulation["hotstart_split"]
    hotstart_end = ea8b_simulation["hotstart_end"]
    split_time = ea8b_simulation["split_time"]
    sim_start_time = ea8b_simulation["sim_start_time"]
    sim_end_time = sim_start_time + timedelta(hours=3, minutes=20)

    surface_flow_params = SurfaceFlowParameters(cfl=0.5, theta=0.7)
    sim_config = SimulationConfig(
        start_time=sim_start_time,
        end_time=sim_end_time,
        record_step=timedelta(seconds=30),
        temporal_type=TemporalType.RELATIVE,
        input_map_names={"ground_elevation": "dem", "friction": "friction"},
        output_map_names={"water_depth": "test_water_depth"},
        drainage_output="out_drainage",
        swmm_inp=str(ea8b_simulation["swmm_inp"]),
        surface_flow_parameters=surface_flow_params,
        orifice_coeff=1.0,
    )

    os.chdir(ea8b_temp_path)

    simulation, vector_output = build_resumed_simulation(sim_config, ea8b_data, hotstart_split)

    assert simulation.sim_time == split_time

    for key in simulation.raster_domain.k_all:
        arr_restored = simulation.raster_domain.get_padded(key)
        arr_saved = hotstart_split.raster_state.padded_arrays[key]
        np.testing.assert_allclose(
            arr_restored,
            arr_saved,
            err_msg=f"Raster state {key} not restored correctly",
        )

    while simulation.sim_time < simulation.end_time:
        simulation.update()
    simulation.finalize()

    resumed_results = drainage_data_to_coupling_series(vector_output.drainage_attributes)
    resumed_metrics = get_reference_metrics(resumed_results, ea8b_reference, helpers)

    assert_matches_reference(resumed_metrics, label="Resumed hotstart run")

    for key in ["water_depth", "discharge_east", "discharge_south"]:
        arr_resumed = simulation.raster_domain.get_array(key)
        arr_uninterrupted = hotstart_end.raster_state.padded_arrays[key][1:-1, 1:-1]
        np.testing.assert_allclose(
            arr_resumed,
            arr_uninterrupted,
            rtol=0.0,
            atol=EA8B_FINAL_ARRAY_ATOL[key],
            err_msg=f"Final {key} mismatch between uninterrupted and resumed simulations",
        )
