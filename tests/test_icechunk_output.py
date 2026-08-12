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

import tempfile
from collections.abc import Mapping
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# Skip entire module if optional dependencies are missing
pytest.importorskip("icechunk")
pytest.importorskip("xarray")
pytest.importorskip("pyproj")

import icechunk
import pyproj
import xarray as xr

from itzi_core.array_definitions import ARRAY_DEFINITIONS, ArrayCategory
from itzi_core.providers.icechunk_output import IcechunkRasterOutputProvider

# Mark all tests in this module as cloud tests
pytestmark = pytest.mark.cloud


@pytest.fixture(scope="function")
def temp_dir():
    return tempfile.TemporaryDirectory()


@pytest.fixture(scope="module")
def maps_dict():
    """A dict representing the arrays to be written to disk"""
    key_list = [
        arr_def.key for arr_def in ARRAY_DEFINITIONS if ArrayCategory.OUTPUT in arr_def.category
    ]
    rng = np.random.default_rng()
    arr_shape = (6, 9)
    return {key: rng.random(size=arr_shape, dtype=np.float32) for key in key_list}


@pytest.fixture(scope="module")
def coordinates(maps_dict: dict):
    """Generate x and y coordinates for the test arrays"""
    arr_shape = next(iter(maps_dict.values())).shape
    y_coords = np.linspace(start=1234, stop=1234 + arr_shape[0], num=arr_shape[0])
    x_coords = np.linspace(start=1234, stop=1234 + arr_shape[1], num=arr_shape[1])
    return {"x_coords": x_coords, "y_coords": y_coords}


@pytest.fixture(scope="module")
def crs():
    """CRS for the test data"""
    return pyproj.CRS.from_epsg(6372)  # Mexico LCC


@pytest.fixture(scope="module")
def out_map_names(maps_dict: dict):
    """Output map names mapping for the test arrays"""
    return {key: f"test_{key}" for key in maps_dict.keys()}


@pytest.fixture
def icechunk_provider(
    temp_dir: tempfile.TemporaryDirectory, coordinates: dict, crs: pyproj.CRS, out_map_names: list
):
    storage = icechunk.local_filesystem_storage(temp_dir.name)
    provider_config = {
        "out_map_names": out_map_names,
        "crs": crs,
        "x_coords": coordinates["x_coords"],
        "y_coords": coordinates["y_coords"],
        "icechunk_storage": storage,
    }
    icechunk_p = IcechunkRasterOutputProvider(provider_config)
    return icechunk_p


def test_missing_zarr_group_is_treated_as_empty(
    icechunk_provider: IcechunkRasterOutputProvider,
):
    assert not icechunk_provider.has_existing_data()
    assert icechunk_provider.get_latest_timestamp() is None
    with pytest.raises(ValueError, match="not a valid zarr store"):
        icechunk_provider.check_repo_match()


def test_missing_crs_metadata_is_rejected(
    icechunk_provider: IcechunkRasterOutputProvider,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(xr, "open_zarr", lambda store: xr.Dataset(attrs={}))

    with pytest.raises(KeyError, match="Existing repository has no 'crs_wkt' attribute"):
        icechunk_provider.check_repo_match()


def test_invalid_crs_metadata_is_rejected(
    icechunk_provider: IcechunkRasterOutputProvider,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(xr, "open_zarr", lambda store: xr.Dataset(attrs={"crs_wkt": "invalid"}))

    with pytest.raises(
        ValueError, match="Existing repository 'crs_wkt' attribute is not valid WKT"
    ):
        icechunk_provider.check_repo_match()


@pytest.mark.parametrize(
    "method_name", ["has_existing_data", "get_latest_timestamp", "check_repo_match"]
)
def test_unexpected_zarr_errors_propagate(
    icechunk_provider: IcechunkRasterOutputProvider,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
):
    def raise_unexpected_error(*args, **kwargs):
        raise RuntimeError("unexpected error")

    monkeypatch.setattr(xr, "open_zarr", raise_unexpected_error)

    with pytest.raises(RuntimeError, match="unexpected error"):
        getattr(icechunk_provider, method_name)()


def test_unexpected_crs_errors_propagate(
    icechunk_provider: IcechunkRasterOutputProvider,
    monkeypatch: pytest.MonkeyPatch,
):
    def raise_unexpected_error(*args, **kwargs):
        raise RuntimeError("unexpected error")

    monkeypatch.setattr(xr, "open_zarr", lambda store: xr.Dataset(attrs={"crs_wkt": "valid"}))
    monkeypatch.setattr(pyproj.CRS, "from_wkt", raise_unexpected_error)

    with pytest.raises(RuntimeError, match="unexpected error"):
        icechunk_provider.check_repo_match()


@pytest.mark.parametrize("start_year", [1, 1978, 3456])
@pytest.mark.parametrize("time_step_s", [1, 60, 300])
def test_write_arrays_absolute(
    icechunk_provider: IcechunkRasterOutputProvider,
    temp_dir: tempfile.TemporaryDirectory,
    start_year: int,
    time_step_s: int,
    maps_dict: dict,
):
    # Write timesteps
    time_steps_num = 3
    sim_time = datetime(year=start_year, month=1, day=1)
    reference_timestep = timedelta(seconds=time_step_s)
    expected_times = []
    for t in range(time_steps_num):
        sim_time += reference_timestep
        expected_times.append(sim_time)
        print(sim_time)
        icechunk_provider.write_arrays(maps_dict, sim_time)

    # Read the data
    storage = icechunk.local_filesystem_storage(temp_dir.name)
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")

    ds = xr.open_zarr(session.store)
    da_time = ds["time"]
    assert da_time.shape == (time_steps_num,)
    timestep = da_time[1] - da_time[0]

    # Assert that the timestep is correct
    timestep_py = pd.to_timedelta(timestep.data).to_pytimedelta()
    assert timestep_py == reference_timestep

    # Assert that all individual timestamps are correct
    actual_times = [pd.to_datetime(t).to_pydatetime() for t in da_time.values]
    assert len(actual_times) == len(expected_times)
    for actual, expected in zip(actual_times, expected_times):
        assert actual == expected, f"Expected {expected}, got {actual}"


@pytest.mark.parametrize(
    "start_seconds",
    [
        0,
        300,
    ],
)
@pytest.mark.parametrize("time_step_s", [1, 60, 3600])
def test_write_arrays_relative(
    icechunk_provider: IcechunkRasterOutputProvider,
    temp_dir: tempfile.TemporaryDirectory,
    start_seconds: int,
    time_step_s: int,
    maps_dict: dict,
):
    """Test writing arrays with relative time (timedelta)"""
    # Write timesteps, with 1 minute in between
    time_steps_num = 3
    sim_time = timedelta(seconds=start_seconds)
    reference_timestep = timedelta(seconds=time_step_s)
    expected_times = []
    for t in range(time_steps_num):
        sim_time += reference_timestep
        expected_times.append(sim_time)
        print(sim_time)
        icechunk_provider.write_arrays(maps_dict, sim_time)

    # Read the data
    storage = icechunk.local_filesystem_storage(temp_dir.name)
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")

    ds = xr.open_zarr(session.store)
    da_time = ds["time"]
    assert da_time.shape == (time_steps_num,)
    timestep = da_time[1] - da_time[0]

    # Assert that the timestep is correct
    timestep_py = pd.to_timedelta(timestep.data).to_pytimedelta()
    assert timestep_py == reference_timestep

    # Assert that all individual timestamps are correct
    actual_times = [pd.to_timedelta(t).to_pytimedelta() for t in da_time.values]
    assert len(actual_times) == len(expected_times)
    for actual, expected in zip(actual_times, expected_times):
        assert actual == expected, f"Expected {expected}, got {actual}"


def test_data_consistency(
    icechunk_provider: IcechunkRasterOutputProvider,
    temp_dir: tempfile.TemporaryDirectory,
    maps_dict: Mapping[str, np.ndarray],
    coordinates: dict,
    crs: pyproj.CRS,
    out_map_names: Mapping[str, str],
):
    """Test that data values and coordinates are correctly
    preserved when reading from zarr with successive writes."""
    # Create first maps dict (use the fixture data)
    maps_dict_1 = maps_dict

    # Create second maps dict with different data
    key_list = list(maps_dict.keys())
    rng = np.random.default_rng(seed=42)  # Use seed for reproducible different data
    arr_shape = next(iter(maps_dict.values())).shape
    maps_dict_2 = {key: rng.random(size=arr_shape, dtype=np.float32) for key in key_list}

    # Write first timestep
    sim_time_1 = datetime(year=2023, month=1, day=1, hour=12)
    icechunk_provider.write_arrays(maps_dict_1, sim_time_1)

    # Write second timestep with different data
    sim_time_2 = datetime(year=2023, month=1, day=1, hour=13)
    icechunk_provider.write_arrays(maps_dict_2, sim_time_2)

    # Read the data back
    storage = icechunk.local_filesystem_storage(temp_dir.name)
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store)

    # Assert that we have 2 timesteps
    assert ds.sizes["time"] == 2

    # Assert that all expected data variables are present
    expected_var_names = set(out_map_names.values())
    actual_var_names = set(ds.data_vars.keys())
    assert expected_var_names == actual_var_names, (
        f"Expected {expected_var_names}, actual {actual_var_names}"
    )

    # Assert that spatial coordinates are preserved
    assert "x" in ds.coords
    assert "y" in ds.coords

    # Verify x coordinates
    expected_x = coordinates["x_coords"]
    actual_x = ds.coords["x"].values
    assert np.allclose(actual_x, expected_x)

    # Verify y coordinates
    expected_y = coordinates["y_coords"]
    actual_y = ds.coords["y"].values
    assert np.allclose(actual_y, expected_y)

    # Assert that data values are preserved for each variable at both timesteps
    for internal_key, zarr_var_name in out_map_names.items():
        if zarr_var_name in ds.data_vars:
            assert ds[zarr_var_name].dims == ("time", "y", "x")
            # Check first timestep data
            original_data_1 = maps_dict_1[internal_key]
            actual_data_1 = ds[zarr_var_name].isel(time=0).values
            assert np.allclose(actual_data_1, original_data_1), (
                f"First timestep data mismatch for {zarr_var_name}"
            )

            # Check second timestep data
            original_data_2 = maps_dict_2[internal_key]
            actual_data_2 = ds[zarr_var_name].isel(time=1).values
            assert np.allclose(actual_data_2, original_data_2), (
                f"Second timestep data mismatch for {zarr_var_name}"
            )

            # Ensure the two timesteps have different data (they should not be identical)
            assert not np.allclose(actual_data_1, actual_data_2), (
                f"Timesteps should have different data for {zarr_var_name}"
            )

    # Assert that CRS information is preserved
    crs_actual = pyproj.CRS.from_wkt(ds.attrs["crs_wkt"])
    assert crs == crs_actual

    # Verify that timestamps are correct
    expected_times = [sim_time_1, sim_time_2]
    actual_times = [pd.to_datetime(t).to_pydatetime() for t in ds["time"].values]
    assert len(actual_times) == len(expected_times)
    for actual, expected in zip(actual_times, expected_times):
        assert actual == expected, f"Expected {expected}, got {actual}"


def test_non_matching_shape(
    temp_dir: tempfile.TemporaryDirectory,
    maps_dict: dict,
    coordinates: dict,
    crs: pyproj.CRS,
    out_map_names: Mapping[str, str],
):
    """Test that writing arrays to an existing repository
    that does not match the new data fails."""

    # Create and write original arrays (6x9 from fixture)
    storage = icechunk.local_filesystem_storage(temp_dir.name)
    provider_config_1 = {
        "out_map_names": out_map_names,
        "crs": crs,
        "x_coords": coordinates["x_coords"],
        "y_coords": coordinates["y_coords"],
        "icechunk_storage": storage,
    }
    icechunk_p1 = IcechunkRasterOutputProvider(provider_config_1)

    # Write original data
    sim_time_1 = datetime(year=2023, month=1, day=1, hour=12)
    icechunk_p1.write_arrays(maps_dict, sim_time_1)

    # Create arrays with different dimensions (4x7)
    new_arr_shape = (4, 7)  # Different from original (6, 9)

    # Create new coordinates for the different dimensions
    new_y_coords = np.linspace(start=5000, stop=5000 + new_arr_shape[0], num=new_arr_shape[0])
    new_x_coords = np.linspace(start=5000, stop=5000 + new_arr_shape[1], num=new_arr_shape[1])
    new_coordinates = {"x_coords": new_x_coords, "y_coords": new_y_coords}

    # Create new provider with different dimensions but same storage
    provider_config_2 = {
        "out_map_names": out_map_names,
        "crs": crs,
        "x_coords": new_coordinates["x_coords"],
        "y_coords": new_coordinates["y_coords"],
        "icechunk_storage": storage,  # Same storage as before
    }
    # Non-matching coordinates should raise ValueError
    with pytest.raises(ValueError):
        IcechunkRasterOutputProvider(provider_config_2)


def test_non_matching_variable_names(
    temp_dir: tempfile.TemporaryDirectory,
    maps_dict: dict,
    coordinates: dict,
    crs: pyproj.CRS,
    out_map_names: Mapping[str, str],
):
    """Test that writing arrays to an existing repository
    with different variable names fails."""

    # Create and write original arrays
    storage = icechunk.local_filesystem_storage(temp_dir.name)
    provider_config_1 = {
        "out_map_names": out_map_names,
        "crs": crs,
        "x_coords": coordinates["x_coords"],
        "y_coords": coordinates["y_coords"],
        "icechunk_storage": storage,
    }
    icechunk_p1 = IcechunkRasterOutputProvider(provider_config_1)

    # Write original data
    sim_time_1 = datetime(year=2023, month=1, day=1, hour=12)
    icechunk_p1.write_arrays(maps_dict, sim_time_1)

    # Create provider with different variable names
    different_map_names = {key: value + "_different" for key, value in out_map_names.items()}
    provider_config_2 = {
        "out_map_names": different_map_names,
        "crs": crs,
        "x_coords": coordinates["x_coords"],
        "y_coords": coordinates["y_coords"],
        "icechunk_storage": storage,  # Same storage as before
    }

    # Non-matching variable names should raise ValueError
    with pytest.raises(ValueError):
        IcechunkRasterOutputProvider(provider_config_2)


def test_non_matching_number_of_variables(
    temp_dir: tempfile.TemporaryDirectory,
    maps_dict: dict,
    coordinates: dict,
    crs: pyproj.CRS,
    out_map_names: Mapping[str, str],
):
    """Test that writing arrays to an existing repository
    with different number of variables fails."""

    # Create and write original arrays
    storage = icechunk.local_filesystem_storage(temp_dir.name)
    provider_config_1 = {
        "out_map_names": out_map_names,
        "crs": crs,
        "x_coords": coordinates["x_coords"],
        "y_coords": coordinates["y_coords"],
        "icechunk_storage": storage,
    }
    icechunk_p1 = IcechunkRasterOutputProvider(provider_config_1)

    # Write original data
    sim_time_1 = datetime(year=2023, month=1, day=1, hour=12)
    icechunk_p1.write_arrays(maps_dict, sim_time_1)

    # Create provider with fewer variables
    fewer_map_names = dict(out_map_names.items())
    del fewer_map_names["water_depth"]
    provider_config_2 = {
        "out_map_names": fewer_map_names,
        "crs": crs,
        "x_coords": coordinates["x_coords"],
        "y_coords": coordinates["y_coords"],
        "icechunk_storage": storage,  # Same storage as before
    }

    # Different number of variables should raise ValueError
    with pytest.raises(ValueError):
        IcechunkRasterOutputProvider(provider_config_2)


def test_non_matching_coordinates_same_dimensions(
    temp_dir: tempfile.TemporaryDirectory,
    maps_dict: dict,
    coordinates: dict,
    crs: pyproj.CRS,
    out_map_names: Mapping[str, str],
):
    """Test that writing arrays to an existing repository
    with same dimension names and sizes but different coordinate values fails."""

    # Create and write original arrays
    storage = icechunk.local_filesystem_storage(temp_dir.name)
    provider_config_1 = {
        "out_map_names": out_map_names,
        "crs": crs,
        "x_coords": coordinates["x_coords"],
        "y_coords": coordinates["y_coords"],
        "icechunk_storage": storage,
    }
    icechunk_p1 = IcechunkRasterOutputProvider(provider_config_1)

    # Write original data
    sim_time_1 = datetime(year=2023, month=1, day=1, hour=12)
    icechunk_p1.write_arrays(maps_dict, sim_time_1)

    # Create coordinates with same shape but different values
    arr_shape = next(iter(maps_dict.values())).shape
    different_y_coords = np.linspace(start=9999, stop=9999 + arr_shape[0], num=arr_shape[0])
    different_x_coords = np.linspace(start=9999, stop=9999 + arr_shape[1], num=arr_shape[1])

    provider_config_2 = {
        "out_map_names": out_map_names,
        "crs": crs,
        "x_coords": different_x_coords,  # Same shape, different values
        "y_coords": different_y_coords,  # Same shape, different values
        "icechunk_storage": storage,  # Same storage as before
    }

    # Non-matching coordinate values should raise ValueError
    with pytest.raises(ValueError):
        IcechunkRasterOutputProvider(provider_config_2)


def test_non_matching_crs(
    temp_dir: tempfile.TemporaryDirectory,
    maps_dict: dict,
    coordinates: dict,
    crs: pyproj.CRS,
    out_map_names: Mapping[str, str],
):
    """Test that writing arrays to an existing repository
    with different CRS fails."""

    # Create and write original arrays
    storage = icechunk.local_filesystem_storage(temp_dir.name)
    provider_config_1 = {
        "out_map_names": out_map_names,
        "crs": crs,
        "x_coords": coordinates["x_coords"],
        "y_coords": coordinates["y_coords"],
        "icechunk_storage": storage,
    }
    icechunk_p1 = IcechunkRasterOutputProvider(provider_config_1)

    # Write original data
    sim_time_1 = datetime(year=2023, month=1, day=1, hour=12)
    icechunk_p1.write_arrays(maps_dict, sim_time_1)

    # Create provider with different CRS
    different_crs = pyproj.CRS.from_epsg(4326)  # WGS84, different from Mexico LCC
    provider_config_2 = {
        "out_map_names": out_map_names,
        "crs": different_crs,  # Different CRS
        "x_coords": coordinates["x_coords"],
        "y_coords": coordinates["y_coords"],
        "icechunk_storage": storage,  # Same storage as before
    }

    # Non-matching CRS should raise ValueError
    with pytest.raises(ValueError):
        IcechunkRasterOutputProvider(provider_config_2)


def test_multi_session_data_persistence(
    temp_dir: tempfile.TemporaryDirectory,
    maps_dict: dict,
    coordinates: dict,
    crs: pyproj.CRS,
    out_map_names: Mapping[str, str],
):
    """Test that writing data with a new provider instance does not overwrite
    data written with a previous provider session."""

    # Create first maps dict (use the fixture data)
    maps_dict_session1 = maps_dict

    # Create second maps dict with different data for session 2
    key_list = list(maps_dict.keys())
    rng = np.random.default_rng(seed=123)  # Use seed for reproducible different data
    arr_shape = next(iter(maps_dict.values())).shape
    maps_dict_session2 = {key: rng.random(size=arr_shape, dtype=np.float32) for key in key_list}

    # Session 1: Create first provider and write initial data
    storage = icechunk.local_filesystem_storage(temp_dir.name)
    provider_config_1 = {
        "out_map_names": out_map_names,
        "crs": crs,
        "x_coords": coordinates["x_coords"],
        "y_coords": coordinates["y_coords"],
        "icechunk_storage": storage,
    }
    icechunk_p1 = IcechunkRasterOutputProvider(provider_config_1)

    # Write data from session 1
    sim_time_1 = datetime(year=2023, month=1, day=1, hour=10)
    sim_time_2 = datetime(year=2023, month=1, day=1, hour=11)
    icechunk_p1.write_arrays(maps_dict_session1, sim_time_1)
    icechunk_p1.write_arrays(maps_dict_session1, sim_time_2)  # Write same data twice

    # Session 2: Create new provider instance with same storage and compatible config
    provider_config_2 = {
        "out_map_names": out_map_names,
        "crs": crs,
        "x_coords": coordinates["x_coords"],
        "y_coords": coordinates["y_coords"],
        "icechunk_storage": storage,  # Same storage as session 1
    }
    icechunk_p2 = IcechunkRasterOutputProvider(provider_config_2)

    # Write new data from session 2
    sim_time_3 = datetime(year=2023, month=1, day=1, hour=12)
    sim_time_4 = datetime(year=2023, month=1, day=1, hour=13)
    icechunk_p2.write_arrays(maps_dict_session2, sim_time_3)
    icechunk_p2.write_arrays(maps_dict_session2, sim_time_4)  # Write different data twice

    # Read all data back
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store)

    # Assert that we have all 4 timesteps
    assert ds.sizes["time"] == 4, f"Expected 4 timesteps, got {ds.sizes['time']}"

    # Assert that all expected data variables are present
    expected_var_names = set(out_map_names.values())
    actual_var_names = set(ds.data_vars.keys())
    assert expected_var_names.issubset(actual_var_names)

    # Verify timestamps are correct
    expected_times = [sim_time_1, sim_time_2, sim_time_3, sim_time_4]
    actual_times = [pd.to_datetime(t).to_pydatetime() for t in ds["time"].values]
    assert len(actual_times) == len(expected_times)
    for actual, expected in zip(actual_times, expected_times):
        assert actual == expected, f"Expected {expected}, got {actual}"

    # Assert that data values are preserved for each variable at all timesteps
    for internal_key, zarr_var_name in out_map_names.items():
        if zarr_var_name in ds.data_vars:
            # Check session 1 data (timesteps 0 and 1)
            original_data_session1 = maps_dict_session1[internal_key]
            actual_data_t0 = ds[zarr_var_name].isel(time=0).values
            actual_data_t1 = ds[zarr_var_name].isel(time=1).values
            assert np.allclose(actual_data_t0, original_data_session1), (
                f"Session 1 timestep 0 data mismatch for {zarr_var_name}"
            )
            assert np.allclose(actual_data_t1, original_data_session1), (
                f"Session 1 timestep 1 data mismatch for {zarr_var_name}"
            )
            # Check session 2 data (timesteps 2 and 3)
            original_data_session2 = maps_dict_session2[internal_key]
            actual_data_t2 = ds[zarr_var_name].isel(time=2).values
            actual_data_t3 = ds[zarr_var_name].isel(time=3).values
            assert np.allclose(actual_data_t2, original_data_session2), (
                f"Session 2 timestep 2 data mismatch for {zarr_var_name}"
            )
            assert np.allclose(actual_data_t3, original_data_session2), (
                f"Session 2 timestep 3 data mismatch for {zarr_var_name}"
            )
            # Ensure session 1 and session 2 data are different
            assert not np.allclose(actual_data_t0, actual_data_t2), (
                f"Session 1 and session 2 should have different data for {zarr_var_name}"
            )
            assert not np.allclose(actual_data_t1, actual_data_t3), (
                f"Session 1 and session 2 should have different data for {zarr_var_name}"
            )
            # Ensure data within each session is consistent
            assert np.allclose(actual_data_t0, actual_data_t1), (
                f"Session 1 data should be consistent across timesteps for {zarr_var_name}"
            )
            assert np.allclose(actual_data_t2, actual_data_t3), (
                f"Session 2 data should be consistent across timesteps for {zarr_var_name}"
            )
    # Assert that spatial coordinates are preserved
    assert "x" in ds.coords
    assert "y" in ds.coords
    expected_x = coordinates["x_coords"]
    actual_x = ds.coords["x"].values
    assert np.allclose(actual_x, expected_x)
    expected_y = coordinates["y_coords"]
    actual_y = ds.coords["y"].values
    assert np.allclose(actual_y, expected_y)

    # Assert that CRS information is preserved
    crs_actual = pyproj.CRS.from_wkt(ds.attrs["crs_wkt"])
    assert crs == crs_actual


def test_maxima_use_configured_names_without_base_arrays(
    temp_dir: tempfile.TemporaryDirectory,
    coordinates: dict,
    crs: pyproj.CRS,
):
    storage = icechunk.local_filesystem_storage(temp_dir.name)
    out_map_names = {"hmax": "maximum_water_depth", "vmax": "maximum_water_speed"}
    provider = IcechunkRasterOutputProvider(
        {
            "out_map_names": out_map_names,
            "crs": crs,
            "x_coords": coordinates["x_coords"],
            "y_coords": coordinates["y_coords"],
            "icechunk_storage": storage,
        }
    )
    arrays = {
        "hmax": np.full((6, 9), 2.0, dtype=np.float32),
        "vmax": np.full((6, 9), 3.0, dtype=np.float32),
    }
    provider.write_arrays(arrays, timedelta(seconds=30))
    provider.finalize()

    ds = xr.open_zarr(icechunk.Repository.open(storage).readonly_session("main").store)
    assert set(ds.data_vars) == set(out_map_names.values())
    for key, name in out_map_names.items():
        assert ds[name].dims == ("time", "y", "x")
        np.testing.assert_array_equal(ds[name].isel(time=0).values, arrays[key])


def test_legacy_static_maximum_schema_is_rejected(
    temp_dir: tempfile.TemporaryDirectory,
    coordinates: dict,
    crs: pyproj.CRS,
):
    storage = icechunk.local_filesystem_storage(temp_dir.name)
    repo = icechunk.Repository.create(storage)
    legacy = xr.Dataset(
        {"maximum_water_depth": (("y", "x"), np.zeros((6, 9), dtype=np.float32))},
        coords={"x": coordinates["x_coords"], "y": coordinates["y_coords"]},
        attrs={"crs_wkt": crs.to_wkt()},
    )
    session = repo.writable_session("main")
    icechunk.xarray.to_icechunk(legacy, session, mode="w-")
    session.commit("legacy maximum")

    with pytest.raises(ValueError, match="dimensions.*new repository.*migrate"):
        IcechunkRasterOutputProvider(
            {
                "out_map_names": {"hmax": "maximum_water_depth"},
                "crs": crs,
                "x_coords": coordinates["x_coords"],
                "y_coords": coordinates["y_coords"],
                "icechunk_storage": storage,
            }
        )
