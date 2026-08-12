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

from collections.abc import Mapping
from datetime import datetime, timedelta
from importlib.metadata import version
from typing import TypedDict

import numpy as np

try:
    import icechunk
    import icechunk.xarray
    import pyproj
    import xarray as xr
    import zarr
except ImportError:
    raise ImportError(
        "To use the Icechunk backend, install itzi with: "
        "'uv tool install itzi[cloud]' "
        "or 'pip install itzi[cloud]'"
    )

from itzi_core.array_definitions import ARRAY_DEFINITIONS
from itzi_core.providers.base import RasterOutputProvider


class IcechunkRasterOutputConfig(TypedDict):
    # A list of var names to be written
    out_map_names: Mapping[str, str]
    crs: pyproj.CRS
    x_coords: np.ndarray
    y_coords: np.ndarray
    icechunk_storage: icechunk.Storage


class IcechunkRasterOutputProvider(RasterOutputProvider):
    """Save raster results in an Icechunk repo."""

    def __init__(self, config: IcechunkRasterOutputConfig) -> None:
        """Create a repo in case it does not exists already"""
        self.out_map_names = config["out_map_names"]
        self.crs = config["crs"]
        self.x_coords = config["x_coords"]
        self.y_coords = config["y_coords"]
        storage = config["icechunk_storage"]

        try:
            self.repo = icechunk.Repository.open(storage)
        except icechunk.IcechunkError as e:
            if "repository doesn't exist" in str(e):
                self.repo = icechunk.Repository.create(storage)
            else:
                raise
        self.spatial_coordinates = self._get_spatial_coordinates()
        self.append_mode = False
        self.time_is_datetime: bool | None = None
        # Make sure new data matches existing one
        if self.has_existing_data():
            self.check_repo_match()
            self.append_mode = True

        self.cf_units = {arr_def.key: arr_def.cf_unit for arr_def in ARRAY_DEFINITIONS}
        self.cf_names = {arr_def.key: arr_def.cf_name for arr_def in ARRAY_DEFINITIONS}
        self.descriptions = {arr_def.key: arr_def.description for arr_def in ARRAY_DEFINITIONS}

    def _get_spatial_coordinates(self) -> list[tuple[str, np.ndarray, dict[str, str]]]:
        # Assume both axis have the same unit
        unit_name = self.crs.axis_info[0].unit_name
        y_attrs = {
            "axis": "Y",
            "units": unit_name,
            "long_name": "y coordinate of projection",
            "standard_name": "projection_y_coordinate",
        }
        x_attrs = {
            "axis": "X",
            "units": unit_name,
            "long_name": "x coordinate of projection",
            "standard_name": "projection_x_coordinate",
        }
        spatial_coordinates = [
            ("y", self.y_coords, y_attrs),
            ("x", self.x_coords, x_attrs),
        ]
        return spatial_coordinates

    def has_existing_data(self) -> bool:
        """Check if the repository already contains data."""
        try:
            session = self.repo.readonly_session("main")
            existing_ds = xr.open_zarr(session.store)
            return len(existing_ds.data_vars) > 0
        except Exception:
            return False

    def get_latest_timestamp(self) -> datetime | timedelta | None:
        """Get the latest timestamp from existing data, or None if no data exists."""
        session = self.repo.readonly_session("main")
        try:
            existing_ds = xr.open_zarr(session.store)
            if "time" in existing_ds.coords and len(existing_ds.coords["time"]) > 0:
                latest_time = existing_ds.coords["time"][-1]
                # Convert numpy datetime64/timedelta64 back to Python types
                if np.issubdtype(latest_time.dtype, np.datetime64):
                    return latest_time.values.astype("datetime64[ms]").astype(datetime)
                elif np.issubdtype(latest_time.dtype, np.timedelta64):
                    return timedelta(
                        milliseconds=int(latest_time.values.astype("timedelta64[ms]").astype(int))
                    )
                else:
                    return None
            else:
                return None
        except Exception:
            return None

    def check_repo_match(self) -> None:
        """Raises ValueError if entry data does not match the existing repo."""
        session = self.repo.readonly_session("main")
        try:
            existing_ds = xr.open_zarr(session.store)
        except Exception as e:
            raise ValueError(f"Existing {session.store} is not a valid zarr store: {e}")

        try:
            existing_crs = pyproj.CRS.from_wkt(existing_ds.attrs["crs_wkt"])
        except Exception as e:
            raise ValueError("Existing repository has no valid CRS metadata") from e
        if existing_crs != self.crs:
            raise ValueError(
                "Provided CRS does not match existing icechunk repository: "
                f"existing={existing_crs.to_epsg()}, configured={self.crs.to_epsg()}"
            )

        existing_names = set(existing_ds.data_vars)
        expected_names = set(self.out_map_names.values())
        if existing_names != expected_names:
            raise ValueError(
                "Configured output names do not match existing icechunk repository: "
                f"existing={sorted(existing_names)}, configured={sorted(expected_names)}"
            )

        expected_dims = ("time", "y", "x")
        for name, variable in existing_ds.data_vars.items():
            if variable.dims != expected_dims:
                raise ValueError(
                    f"Existing variable {name!r} has dimensions {variable.dims}; expected "
                    f"{expected_dims}. Create a new repository or explicitly migrate the "
                    "legacy repository before appending."
                )
            try:
                x_matches = np.allclose(variable.coords["x"].values, self.x_coords)
                y_matches = np.allclose(variable.coords["y"].values, self.y_coords)
            except (KeyError, ValueError):
                x_matches = y_matches = False
            if not x_matches or not y_matches:
                raise ValueError(
                    f"Coordinates for existing variable {name!r} do not match the configured grid"
                )

        if "time" not in existing_ds.coords:
            raise ValueError("Existing repository has no compatible temporal time coordinate")
        time_dtype = existing_ds["time"].dtype
        if np.issubdtype(time_dtype, np.datetime64):
            self.time_is_datetime = True
        elif np.issubdtype(time_dtype, np.timedelta64):
            self.time_is_datetime = False
        else:
            raise ValueError("Existing repository has no compatible temporal time coordinate")

    def write_arrays(
        self, array_dict: Mapping[str, np.ndarray], sim_time: datetime | timedelta
    ) -> None:
        """Write results to an icechunk repository"""
        incoming_time_is_datetime = isinstance(sim_time, datetime)
        if (
            self.time_is_datetime is not None
            and self.time_is_datetime != incoming_time_is_datetime
        ):
            raise ValueError(
                "Incoming time coordinate type does not match the existing icechunk repository"
            )
        self.time_is_datetime = incoming_time_is_datetime
        vars_to_write = list(array_dict.keys())
        expected_map_keys = set(self.out_map_names.keys())
        if not expected_map_keys == set(vars_to_write):
            raise ValueError(
                "Variables names do not match: "
                f"Expected: {expected_map_keys}, "
                f"Received: {vars_to_write}"
            )
        # prepare the data
        dataset = self._build_dataset(array_dict, sim_time)
        first_var_name = next(iter(self.out_map_names.values()))
        time_encoding = dataset[first_var_name].encoding["time"]
        # Commit to the repo
        commit_message = f"itzi results for simulation time {sim_time}"
        icechunk_session = self.repo.writable_session("main")
        if not self.append_mode:
            icechunk.xarray.to_icechunk(
                dataset, icechunk_session, mode="w-", encoding={"time": time_encoding}
            )
            self.append_mode = True  # Now there is data
        else:
            # Use zarr append to preserve time encoding
            self._zarr_append(icechunk_session.store, dataset)
        icechunk_session.commit(commit_message)

    def _build_dataset(
        self, array_dict: Mapping[str, np.ndarray], sim_time: datetime | timedelta
    ) -> xr.Dataset:
        """From a dict of arrays, return an xarray dataset."""
        data_vars = {}
        if isinstance(sim_time, datetime):
            time_dtype = "datetime64[ms]"
            sim_time_np = np.datetime64(sim_time, "ms")
            time_unit = "milliseconds since 1970-01-01T00:00:00"
        elif isinstance(sim_time, timedelta):
            time_dtype = "timedelta64[s]"
            sim_time_np = np.timedelta64(sim_time, "ms")
            time_unit = "milliseconds"
        else:
            raise ValueError(f"Unknown temporal type: {type(sim_time)}")

        time_coordinate = np.array([sim_time_np], dtype=time_dtype)
        time_encoding = {
            "units": time_unit,
            "dtype": time_dtype,
        }
        coordinates = [("time", time_coordinate, {})] + self.spatial_coordinates

        for key, arr in array_dict.items():
            var_name = self.out_map_names[key]
            coords_shape = (len(self.y_coords), len(self.x_coords))
            if arr.shape != coords_shape:
                raise ValueError(
                    f"Array shape {arr.shape} incompatible with coordinates shape {coords_shape}"
                )

            var_attributes = {
                "units": self.cf_units[key],
                "standard_name": self.cf_names[key],
                "long_name": self.descriptions[key],
            }
            arr = np.expand_dims(arr, axis=0)

            data_array = xr.DataArray(
                data=arr,
                coords=coordinates,
                name=var_name,  # Write the requested name
                attrs=var_attributes,
            )
            assert data_array["time"].dtype == time_dtype
            data_array.encoding["time"] = time_encoding
            data_vars[var_name] = data_array
        dataset_attributes = {
            "crs_wkt": self.crs.to_wkt(),
            "source": f"itzi version {version('itzi_core')},",
        }
        dataset = xr.Dataset(data_vars, attrs=dataset_attributes)

        dataset["time"].encoding.update(time_encoding)

        return dataset

    def _zarr_append(self, store, dataset: xr.Dataset) -> None:
        """Optimized zarr append using direct indexing instead of concatenation."""
        # Open the zarr group
        z_group = zarr.open_group(store, mode="r+")

        # Get the new time value
        new_time = dataset["time"].values[0]

        # Append time coordinate
        current_time_size = z_group["time"].shape[0]
        z_group["time"].resize(current_time_size + 1)
        z_group["time"][current_time_size] = new_time

        # Append data for each variable
        for var_name, data_array in dataset.data_vars.items():
            current_shape = z_group[var_name].shape
            new_shape = (current_shape[0] + 1,) + current_shape[1:]
            z_group[var_name].resize(new_shape)
            # Use direct assignment
            z_group[var_name][current_shape[0]] = data_array.values[0]
