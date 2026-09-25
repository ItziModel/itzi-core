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

from collections import namedtuple
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from itzi_core import DomainData
from itzi_core.const import TemporalType
from itzi_core.data_containers import SimulationConfig, SurfaceFlowParameters
from itzi_core.providers.memory_output import (
    MemoryMassBalanceOutputProvider,
    MemoryRasterOutputProvider,
    MemoryVectorOutputProvider,
)
from itzi_core.simulation_builder import SimulationBuilder

ASCIIMetadata = namedtuple(
    "ASCIIMetadata", ["ncols", "nrows", "xllcorner", "yllcorner", "cellsize"]
)


def read_ascii_grid(filepath):
    filepath = Path(filepath)
    with filepath.open("r") as f:
        # Read header
        ncols = int(f.readline().split()[1])
        nrows = int(f.readline().split()[1])
        xllcorner = float(f.readline().split()[1])
        yllcorner = float(f.readline().split()[1])
        cellsize = float(f.readline().split()[1])
        # Read the grid data
        data = np.loadtxt(f)
    assert data.shape == (nrows, ncols)
    metadata = ASCIIMetadata(ncols, nrows, xllcorner, yllcorner, cellsize)
    return data, metadata


def metadata_to_domaindata(metadata: ASCIIMetadata) -> DomainData:
    rows = metadata.nrows
    cols = metadata.ncols
    west = metadata.xllcorner
    south = metadata.yllcorner

    return DomainData(
        north=south + rows * metadata.cellsize,
        south=south,
        east=west + cols * metadata.cellsize,
        west=west,
        rows=rows,
        cols=cols,
        crs_wkt="",
    )


def output_map_names(prefix, keys):
    return {key: f"{prefix}_{key}" for key in keys}


@pytest.fixture(scope="module")
def mcdo_norain_sim(test_data_path):
    """Run a simulation for MacDonald 1D solution long channel without rain.
    Delestre, O., Lucas, C., Ksinant, P.-A., Darboux, F., Laguerre, C., Vo, T.-N.-T., … Cordier, S. (2013).
    SWASHES: a compilation of shallow water analytic solutions for hydraulic and environmental studies.
    International Journal for Numerical Methods in Fluids, 72(3), 269–300. https://doi.org/10.1002/fld.3741
    """
    test_data_path = Path(test_data_path)
    data_dir = test_data_path / Path("analytic")
    reference_path = data_dir / Path("mcdo_norain.csv")
    reference = pd.read_csv(reference_path)
    arr_topo = reference["topo"].values
    # Create DEM
    arr_dem = np.tile(arr_topo, (3, 1))
    assert arr_dem.shape == (3, 200)
    domain_data = DomainData(
        north=5 * 200, south=0, east=5 * 200, west=0, rows=3, cols=200, crs_wkt=""
    )
    # Manning
    arr_n = np.full_like(arr_dem, fill_value=0.033)
    # Inflow at westmost boundary
    arr_inflow = np.zeros_like(arr_dem)
    arr_inflow[:, 0] = 0.4
    # free eastmost boundary
    arr_bctype = np.ones_like(arr_dem)
    arr_bctype[:, -1] = 2
    # No mask. Whole domain.
    array_mask = np.full(shape=arr_dem.shape, fill_value=False, dtype=np.bool_)

    config = SimulationConfig(
        start_time=datetime(2000, 1, 1),
        end_time=datetime(2000, 1, 1, 0, 20),
        record_step=timedelta(minutes=5),
        temporal_type=TemporalType.RELATIVE,
        input_map_names={
            "ground_elevation": "dem@mcdo_norain",
            "boundary_type": "bctype@mcdo_norain",
            "inflow": "inflow@mcdo_norain",
            "friction": "n@mcdo_norain",
        },
        output_map_names=output_map_names(
            "out_mcdo_norain",
            ["water_depth", "water_surface_elevation", "flow_rate_x", "flow_rate_y"],
        ),
        surface_flow_parameters=SurfaceFlowParameters(dtmax=2, cfl=0.5),
    )
    raster_output = MemoryRasterOutputProvider()
    mass_balance_output = MemoryMassBalanceOutputProvider()
    simulation = (
        SimulationBuilder(config, array_mask, np.float32)
        .with_domain_data(domain_data)
        .with_raster_output_provider(raster_output)
        .with_vector_output_provider(MemoryVectorOutputProvider())
        .with_mass_balance_output_provider(mass_balance_output)
        .build()
    )
    # Set the input arrays
    simulation.set_array("ground_elevation", arr_dem)
    simulation.set_array("boundary_type", arr_bctype)
    simulation.set_array("inflow", arr_inflow)
    simulation.set_array("friction", arr_n)
    # run the simulation
    simulation.initialize()
    while simulation.sim_time < simulation.end_time:
        simulation.update()
    simulation.finalize()
    return simulation, reference, mass_balance_output.reports


class TestMcdo_norain:
    def test_mcdo_norain(self, mcdo_norain_sim):
        simulation, reference, _ = mcdo_norain_sim
        raster_results = simulation.report.raster_provider.output_maps_dict
        _, wse_array = raster_results["water_surface_elevation"][-1]
        wse_centerline = pd.DataFrame({"wse_model": wse_array[1, :]})
        df_results = reference.join(wse_centerline)
        df_results["abs_error"] = np.abs(df_results["wse_model"] - df_results["wse"])
        mae = np.mean(df_results["abs_error"])
        assert mae < 0.03

    def test_flow_is_unidimensional(self, mcdo_norain_sim):
        simulation, _, _ = mcdo_norain_sim
        """In the MacDonald 1D test, flow should be unidimensional in the X dimension"""
        flow_rate_y_arrays = simulation.report.raster_provider.output_maps_dict["flow_rate_y"]
        for _, flow_rate_y_array in flow_rate_y_arrays:
            print(flow_rate_y_array)
            # univar = gscript.parse_command("r.univar", map=raster, flags="g")
            assert np.min(flow_rate_y_array) == 0
            assert np.max(flow_rate_y_array) == 0

    def test_mass_balance_is_coherent(self, mcdo_norain_sim):
        _, _, reports = mcdo_norain_sim
        assert reports
        assert all(isinstance(report.simulation_time, timedelta) for report in reports)

        volume_change = np.array([report.volume_change for report in reports])
        created_volume = np.array([report.created_volume for report in reports])
        created_volume_ratio = np.array([report.created_volume_ratio for report in reports])
        expected_ratio = np.zeros_like(volume_change)
        np.divide(created_volume, volume_change, out=expected_ratio, where=volume_change != 0)
        assert np.allclose(created_volume_ratio, expected_ratio, atol=0.0001)

        volume_change_ref = np.array(
            [
                report.boundary_volume
                + report.rainfall_volume
                + report.infiltration_volume
                + report.inflow_volume
                + report.losses_volume
                + report.drainage_network_volume
                + report.created_volume
                for report in reports
            ]
        )
        assert np.allclose(volume_change_ref, volume_change, atol=1, rtol=0.01)


@pytest.fixture(scope="module")
def mcdo_rain_sim(test_data_path):
    """Run a simulation for MacDonald 1D solution long channel with rain.
    Delestre, O., Lucas, C., Ksinant, P.-A., Darboux, F., Laguerre, C., Vo, T.-N.-T., … Cordier, S. (2013).
    SWASHES: a compilation of shallow water analytic solutions for hydraulic and environmental studies.
    International Journal for Numerical Methods in Fluids, 72(3), 269–300. https://doi.org/10.1002/fld.3741
    """
    data_dir = Path(test_data_path) / Path("analytic")

    # Create DEM
    reference_path = data_dir / Path("mcdo_rain.csv")
    reference = pd.read_csv(reference_path)
    arr_topo = reference["topo"].values
    arr_dem = np.tile(arr_topo, (3, 1))
    assert arr_dem.shape == (3, 200)
    domain_data = DomainData(
        north=5 * 200, south=0, east=5 * 200, west=0, rows=3, cols=200, crs_wkt=""
    )
    # Create Manning map
    arr_n = np.full_like(arr_dem, fill_value=0.033)
    # Inflow at westmost boundary
    arr_inflow = np.zeros_like(arr_dem)
    arr_inflow[:, 0] = 0.2
    # free eastmost boundary
    arr_bctype = np.ones_like(arr_dem)
    arr_bctype[:, -1] = 2
    # Simulation object takes rainfall input in m/s.
    arr_rain = np.full_like(arr_dem, fill_value=0.001)
    # No mask. Whole domain.
    array_mask = np.full(shape=arr_dem.shape, fill_value=False, dtype=np.bool_)

    config = SimulationConfig(
        start_time=datetime(2020, 2, 1, 0, 1),
        end_time=datetime(2020, 2, 1, 0, 41),
        record_step=timedelta(minutes=10),
        temporal_type=TemporalType.ABSOLUTE,
        input_map_names={
            "ground_elevation": "dem@mcdo_rain",
            "boundary_type": "bctype@mcdo_rain",
            "inflow": "inflow@mcdo_rain",
            "friction": "n@mcdo_rain",
            "rainfall_rate": "rain@mcdo_rain",
        },
        output_map_names=output_map_names(
            "out_mcdo_rain", ["water_depth", "water_surface_elevation"]
        ),
        surface_flow_parameters=SurfaceFlowParameters(dtmax=2, cfl=0.5),
        dtinf=1,
    )
    raster_output = MemoryRasterOutputProvider()
    mass_balance_output = MemoryMassBalanceOutputProvider()
    simulation = (
        SimulationBuilder(config, array_mask, np.float32)
        .with_domain_data(domain_data)
        .with_raster_output_provider(raster_output)
        .with_vector_output_provider(MemoryVectorOutputProvider())
        .with_mass_balance_output_provider(mass_balance_output)
        .build()
    )
    # Set the input arrays
    simulation.set_array("ground_elevation", arr_dem)
    simulation.set_array("boundary_type", arr_bctype)
    simulation.set_array("inflow", arr_inflow)
    simulation.set_array("friction", arr_n)
    simulation.set_array("rainfall_rate", arr_rain)
    # run the simulation
    simulation.initialize()
    while simulation.sim_time < simulation.end_time:
        simulation.update()
    simulation.finalize()
    return simulation, reference, mass_balance_output.reports


def test_mcdo_rain(mcdo_rain_sim):
    simulation, reference, reports = mcdo_rain_sim
    raster_results = simulation.report.raster_provider.output_maps_dict
    _, wse_array = raster_results["water_surface_elevation"][-1]
    wse_centerline = pd.DataFrame({"wse_model": wse_array[1, :]})
    df_results = reference.join(wse_centerline)
    df_results["abs_error"] = np.abs(df_results["wse_model"] - df_results["wse"])
    mae = np.mean(df_results["abs_error"])
    print(mae)
    assert mae < 0.035

    assert reports
    assert all(isinstance(report.simulation_time, datetime) for report in reports)
