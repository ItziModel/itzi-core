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

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

try:
    import xarray as xr
except ImportError:
    raise ImportError(
        "To use the xarray input backend, install itzi-core with: "
        "'uv add itzi-core[xarray]' or 'pip install itzi-core[xarray]'"
    )

from itzi_core.array_definitions import INPUT_ARRAY_KEYS
from itzi_core.const import TemporalType
from itzi_core.domain_data import DomainData
from itzi_core.providers.base import RasterInputProvider

__all__ = ["XarrayDimensions", "XarrayRasterInputConfig", "XarrayRasterInputProvider"]


class XarrayDimensions(BaseModel):
    """Names of the time and spatial dimensions for one data variable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    time: str = Field(default="time", min_length=1)
    y: str = Field(default="y", min_length=1)
    x: str = Field(default="x", min_length=1)


class XarrayRasterInputConfig(BaseModel):
    """Configuration for an xarray-backed raster input provider."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    dataset: xr.Dataset
    input_map_names: dict[str, str] = Field(min_length=1)
    dimension_names: dict[str, XarrayDimensions] = Field(default_factory=dict)
    simulation_start_time: datetime
    simulation_end_time: datetime

    @model_validator(mode="after")
    def validate_configuration(self) -> XarrayRasterInputConfig:
        if self.simulation_start_time >= self.simulation_end_time:
            raise ValueError("simulation_start_time must be before simulation_end_time")

        invalid_input_keys = sorted(set(self.input_map_names) - INPUT_ARRAY_KEYS)
        if invalid_input_keys:
            raise ValueError(
                f"input_map_names contain invalid input keys: {', '.join(invalid_input_keys)}"
            )

        dataset_var_names = {str(var_name) for var_name in self.dataset.data_vars}
        unknown_map_variables = sorted(set(self.input_map_names.values()) - dataset_var_names)
        if unknown_map_variables:
            raise ValueError(
                "input_map_names reference variables not found in the dataset: "
                f"{', '.join(unknown_map_variables)}"
            )

        input_var_names = set(self.input_map_names.values())
        unknown_dimension_variables = sorted(set(self.dimension_names) - input_var_names)
        if unknown_dimension_variables:
            raise ValueError(
                "dimension_names reference variables not selected by input_map_names: "
                f"{', '.join(unknown_dimension_variables)}"
            )

        return self


class XarrayRasterInputProvider(RasterInputProvider):
    """Provide raster simulation inputs from an xarray dataset."""

    def __init__(self, config: XarrayRasterInputConfig) -> None:
        if not isinstance(config, XarrayRasterInputConfig):
            raise TypeError("config must be an XarrayRasterInputConfig instance")

        self.sim_start_time = config.simulation_start_time
        self.sim_end_time = config.simulation_end_time
        self.input_map_names = config.input_map_names

        self.dataset = config.dataset
        self.crs_wkt: str = self.dataset.attrs.get("crs_wkt", "")

        input_var_names = set(config.input_map_names.values())
        self.dataset_dims: dict[str, XarrayDimensions] = {
            var_name: config.dimension_names.get(var_name, XarrayDimensions())
            for var_name in input_var_names
        }

        # Data validation
        self._validate_dimensions()
        self._validate_variables_dimensionality()
        self._validate_equal_spacing_of_spatial_dims()
        self._validate_coordinates_are_sorted()
        self._validate_equality_of_spatial_dims()

        self.temporal_types: dict[str, TemporalType] = self.detect_temporal_type()

    def _validate_dimensions(self) -> None:
        """Validate that:
        - the specified spatial dimensions exist in the dataset.
        - The dimensions are one-dimensional."""
        for var_name in self.input_map_names.values():
            da_var: xr.DataArray = self.dataset[var_name]
            var_dims = {str(dim) for dim in da_var.dims}
            dimensions = self.dataset_dims[var_name]
            for dim_type, dim_name in (("x", dimensions.x), ("y", dimensions.y)):
                if dim_name not in var_dims:
                    raise ValueError(
                        f"{dim_type} dimension '{dim_name}' not found in variable {var_name}. "
                        f"Available dimensions: {var_dims}"
                    )
                da_dim: xr.DataArray = self.dataset[dim_name]
                if da_dim.ndim != 1:
                    raise ValueError(
                        f"Dimension '{dim_name}' not one-dimensional. "
                        f"Found {da_dim.ndim} dimensions with shape {da_dim.shape}"
                    )

    def _validate_variables_dimensionality(self) -> None:
        """Validate that all variables are either 2D[y, x] or 3D[time, y, x]."""
        for var_name in self.input_map_names.values():
            da_var: xr.DataArray = self.dataset[var_name]
            var_dims = {str(dim) for dim in da_var.dims}
            num_dims = len(da_var.dims)
            dimensions = self.dataset_dims[var_name]
            x_dim = dimensions.x
            y_dim = dimensions.y

            # Check if variable is 2D or 3D
            if num_dims == 2:
                # Must have x and y dimensions
                if x_dim not in var_dims or y_dim not in var_dims:
                    raise ValueError(
                        f"2D variable '{var_name}' must have dimensions [{y_dim}, {x_dim}]. "
                        f"Found: {list(da_var.dims)}"
                    )
            elif num_dims == 3:
                time_dim = dimensions.time

                # Must have x, y, and time dimensions
                if time_dim not in var_dims:
                    raise ValueError(
                        f"3D variable '{var_name}' must have time dimension '{time_dim}'. "
                        f"Found: {list(da_var.dims)}"
                    )
                if x_dim not in var_dims or y_dim not in var_dims:
                    raise ValueError(
                        f"3D variable '{var_name}' must have "
                        f"dimensions [{time_dim}, {y_dim}, {x_dim}]. "
                        f"Found: {list(da_var.dims)}"
                    )
            else:
                raise ValueError(
                    f"Variable '{var_name}' must be either 2D[y, x] or 3D[time, y, x]. "
                    f"Found {num_dims}D: {list(da_var.dims)}"
                )

    def _validate_equal_spacing_of_spatial_dims(self) -> None:
        """Check if spatial coordinates are equally spaced."""
        for var_name in self.input_map_names.values():
            dimensions = self.dataset_dims[var_name]
            x_dim = dimensions.x
            y_dim = dimensions.y
            for dim_name in [x_dim, y_dim]:
                coord: xr.DataArray = self.dataset[dim_name]
                diffs = np.diff(coord.values if hasattr(coord, "values") else coord)
                # no coordinates present
                if len(diffs) == 0:
                    continue
                if not np.allclose(diffs, diffs[0]):
                    raise ValueError(
                        f"Dimension {dim_name} of variable {var_name} not equally spaced."
                    )

    def _validate_equality_of_spatial_dims(self):
        """Check that all spatial dimensions are equals.
        ⚠️ Must check first that they are all one-dimensional,
        if not, the check might wrongly pass because of numpy broadcasting.
        """
        for dim_type, dim_names in (
            ("x", {self.dataset_dims[var_name].x for var_name in self.input_map_names.values()}),
            ("y", {self.dataset_dims[var_name].y for var_name in self.input_map_names.values()}),
        ):
            if len(dim_names) == 0:
                continue
            da_list: list[xr.DataArray] = [self.dataset[dim_name] for dim_name in dim_names]
            ref_da: xr.DataArray = da_list[0]
            for da in da_list:
                if not np.allclose(ref_da.values, da.values):
                    raise ValueError(
                        f"{dim_type} dimensions {ref_da.name} and {da.name} are not equals"
                    )

    def _validate_coordinates_are_sorted(self):
        """Check if coordinates used by selected input variables are sorted."""
        spatial_coord_names: set[str] = set()
        time_coord_names: set[str] = set()
        for var_name in self.input_map_names.values():
            dimensions = self.dataset_dims[var_name]
            spatial_coord_names.update((dimensions.x, dimensions.y))
            if dimensions.time in self.dataset[var_name].dims:
                time_coord_names.add(dimensions.time)

        for coord_name in spatial_coord_names:
            da_coord: xr.DataArray = self.dataset[coord_name]
            arr_coord: np.ndarray = da_coord.values
            coord_ascending = self.is_array_sorted(arr_coord, ascending=True)
            coord_descending = self.is_array_sorted(arr_coord, ascending=False)
            dim_is_sorted = coord_ascending or coord_descending
            if not dim_is_sorted:
                raise ValueError(f"Coordinates array {coord_name} is not sorted.")

        for coord_name in time_coord_names:
            arr_coord = self.dataset[coord_name].values
            if not self.is_array_sorted(arr_coord, ascending=True):
                raise ValueError(
                    f"Time coordinates array {coord_name} must be sorted in ascending order."
                )

    def detect_temporal_type(self) -> dict[str, TemporalType]:
        """Detect if time coordinates are relative (timedelta) or absolute (datetime)."""
        temporal_type_dict: dict[str, TemporalType] = {}
        try:
            # Only collect time dimension names if they actually exist in the variable's dimensions
            time_dim_names: set[str] = set()
            for var_name in self.input_map_names.values():
                da_var: xr.DataArray = self.dataset[var_name]
                time_dim = self.dataset_dims[var_name].time
                # Check if this variable actually has the time dimension
                if time_dim in da_var.dims:
                    time_dim_names.add(time_dim)
        # If no time dimension, return empty dict
        except KeyError:
            return temporal_type_dict

        for time_dim_name in time_dim_names:
            time_coords = self.dataset[time_dim_name]
            # Check the dtype of time coordinates
            if np.issubdtype(time_coords.dtype, np.timedelta64):
                temporal_type_dict[time_dim_name] = TemporalType.RELATIVE
            elif np.issubdtype(time_coords.dtype, np.datetime64):
                temporal_type_dict[time_dim_name] = TemporalType.ABSOLUTE
            else:
                raise ValueError(f"Unsupported temporal type: {time_coords.dtype}")
        return temporal_type_dict

    def get_domain_data(self) -> DomainData:
        """Return a DomainData object."""
        # get the first coords. They are all the same (checked at init).
        y_dim_name: str = next(
            self.dataset_dims[var_name].y for var_name in self.input_map_names.values()
        )
        x_dim_name: str = next(
            self.dataset_dims[var_name].x for var_name in self.input_map_names.values()
        )
        # Coordinates are at the center of the cells
        y_coords: xr.DataArray = self.dataset[y_dim_name]
        rows: int = len(y_coords)
        northernmost_cell = y_coords.values.max()
        southernmost_cell = y_coords.values.min()
        nsres = float((northernmost_cell - southernmost_cell) / (rows - 1))
        north = float(northernmost_cell + nsres / 2)
        south = float(southernmost_cell - nsres / 2)

        x_coords: xr.DataArray = self.dataset[x_dim_name]
        cols: int = len(x_coords)
        easternmost_cell = x_coords.values.max()
        westernmost_cell = x_coords.values.min()
        ewres = float((easternmost_cell - westernmost_cell) / (cols - 1))
        east = float(easternmost_cell + ewres / 2)
        west = float(westernmost_cell - ewres / 2)

        return DomainData(
            north=north,
            south=south,
            east=east,
            west=west,
            rows=rows,
            cols=cols,
            crs_wkt=self.crs_wkt,
        )

    @staticmethod
    def is_array_sorted(arr: np.ndarray, ascending: bool = True) -> bool:
        differences = np.diff(arr)
        zero = np.timedelta64(0, "ns") if np.issubdtype(differences.dtype, np.timedelta64) else 0
        if ascending:
            return bool(np.all(differences >= zero))
        else:
            return bool(np.all(differences <= zero))

    @staticmethod
    def get_active_window(coord_array, target_value):
        """Return the active slice index and half-open validity window for a time value.

        Exact-boundary lookups belong to the new slice, not the previous one, hence the
        use of `side="right"`.
        """
        active_idx = int(np.searchsorted(coord_array, target_value, side="right") - 1)
        if active_idx < 0:
            return None

        start_value = coord_array[active_idx]
        if active_idx + 1 < len(coord_array):
            end_value = coord_array[active_idx + 1]
        else:
            end_value = None
        return active_idx, start_value, end_value

    def _to_datetime(self, time_value: np.timedelta64 | np.datetime64, time_dim: str) -> datetime:
        if np.isnat(time_value):
            raise ValueError("Time coordinate values cannot be NaT.")

        if self.temporal_types[time_dim] == TemporalType.RELATIVE:
            nanoseconds = time_value.astype("timedelta64[ns]").astype(np.int64)
            microseconds = int((nanoseconds + 500) // 1_000)
            return self.sim_start_time + timedelta(microseconds=microseconds)

        value: datetime = time_value.astype("datetime64[us]").item()
        return value

    def get_array(
        self, map_key: str, current_time: datetime
    ) -> tuple[np.ndarray | None, datetime, datetime]:
        """Take a given map key and current time
        return a numpy array associated with its start and end time
        if no map is found, return None instead of an array
        and a default start_time and end_time."""
        # Return None if no variable with requested name
        try:
            var_name: str = self.input_map_names[map_key]
        except KeyError:
            return None, self.sim_start_time, self.sim_end_time

        da: xr.DataArray = self.dataset[var_name]
        time_dim = self.dataset_dims[var_name].time

        if time_dim in da.dims:
            if not self.sim_start_time <= current_time < self.sim_end_time:
                return None, self.sim_start_time, self.sim_end_time

            da_time: xr.DataArray = da[time_dim]
            if self.temporal_types[time_dim] == TemporalType.RELATIVE:
                # convert datetime to timedelta for the search
                np_time = np.timedelta64(current_time - self.sim_start_time)
            else:  # absolute time
                np_time = np.datetime64(current_time)

            active_window = self.get_active_window(da_time.values, np_time)
            if active_window is None:
                first_time = min(self.sim_end_time, self._to_datetime(da_time.values[0], time_dim))
                return None, self.sim_start_time, first_time

            active_idx, start_np, end_np = active_window
            da_selected: xr.DataArray = da.isel({time_dim: active_idx})
            start_time = max(self.sim_start_time, self._to_datetime(start_np, time_dim))
            end_time = self.sim_end_time
            if end_np is not None:
                end_time = min(self.sim_end_time, self._to_datetime(end_np, time_dim))
        else:
            # If no time dimension, send back the whole array
            start_time: datetime = self.sim_start_time
            end_time: datetime = self.sim_end_time
            da_selected: xr.DataArray = da

        assert len(da_selected.shape) == 2
        return da_selected.values, start_time, end_time
