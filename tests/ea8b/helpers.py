"""
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

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from itzi_core.data_containers import DrainageNetworkData, SimulationConfig
from itzi_core.providers.memory_output import (
    MemoryRasterOutputProvider,
    MemoryVectorOutputProvider,
)
from itzi_core.providers.xarray_input import XarrayRasterInputProvider
from itzi_core.simulation_builder import SimulationBuilder

EA8B_REFERENCE_MIN_NSE = 0.99
EA8B_REFERENCE_MAX_RSR = 0.01
EA8B_FINAL_ARRAY_ATOL: dict[str, float] = {
    # Hotstart resume is not restart-exact with SWMM ponding enabled. The current
    # scheduler semantics keep the resumed run within XPSTORM acceptance while a
    # few qs cells can drift slightly above the historical restart tolerances.
    "water_depth": 7.0e-3,
    "qe": 2.9e-3,
    "qs": 1.6e-3,
}


def drainage_data_to_coupling_series(
    drainage_records: list[tuple[datetime | timedelta, DrainageNetworkData]],
) -> pd.Series:
    rows: list[dict[str, object]] = []
    for sim_time, drainage_data in drainage_records:
        for node in drainage_data.nodes:
            rows.append({"sim_time": sim_time, **node.attributes.model_dump()})

    if not rows:
        return pd.Series(name="coupling_flow", dtype=float)

    df_results = pd.DataFrame(rows)
    df_results["sim_time"] = pd.to_timedelta(df_results["sim_time"])
    df_results["start_time"] = df_results["sim_time"].dt.total_seconds().astype(int)
    df_results.set_index("start_time", inplace=True)
    df_results.drop(columns=["sim_time"], inplace=True)
    df_results = df_results[df_results.index >= 3000]
    df_results.index = pd.to_timedelta(df_results.index, unit="s")
    return df_results["coupling_flow"]


def get_reference_metrics(
    results: pd.Series, reference: pd.Series, helpers
) -> dict[str, float | bool]:
    nse = helpers.get_nse(results, reference)
    rsr = helpers.get_rsr(results, reference)
    return {
        "nse": float(nse),
        "rsr": float(rsr),
        "matches_reference": bool(nse > EA8B_REFERENCE_MIN_NSE and rsr < EA8B_REFERENCE_MAX_RSR),
    }


def assert_matches_reference(metrics: dict[str, float | bool], label: str) -> None:
    assert metrics["nse"] > EA8B_REFERENCE_MIN_NSE, (
        f"{label} NSE below XPSTORM tolerance: "
        f"{metrics['nse']:.6f} <= {EA8B_REFERENCE_MIN_NSE:.2f}"
    )
    assert metrics["rsr"] < EA8B_REFERENCE_MAX_RSR, (
        f"{label} RSR above XPSTORM tolerance: "
        f"{metrics['rsr']:.6f} >= {EA8B_REFERENCE_MAX_RSR:.2f}"
    )


def build_resumed_simulation(
    sim_config: SimulationConfig,
    ea8b_data: dict,
    hotstart_bytes: bytes,
):
    arr_mask = np.zeros((ea8b_data["rows"], ea8b_data["cols"]), dtype=bool)

    raster_input_provider = XarrayRasterInputProvider(
        {
            "dataset": ea8b_data["dataset"],
            "input_map_names": sim_config.input_map_names,
            "simulation_start_time": sim_config.start_time,
            "simulation_end_time": sim_config.end_time,
        }
    )
    raster_output_provider = MemoryRasterOutputProvider(sim_config.output_map_names)
    vector_output_provider = MemoryVectorOutputProvider()

    simulation = (
        SimulationBuilder(sim_config, arr_mask)
        .with_input_provider(raster_input_provider)
        .with_raster_output_provider(raster_output_provider)
        .with_vector_output_provider(vector_output_provider)
        .with_hotstart(hotstart_bytes)
        .build()
    )

    return simulation, vector_output_provider
