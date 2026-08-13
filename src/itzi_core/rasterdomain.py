"""
Copyright (C) 2016-2026 Laurent G. Courty

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 2.1
of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.
"""

import io
from typing import Self

import numpy as np

from itzi_core.array_definitions import ARRAY_DEFINITIONS, ArrayCategory
from itzi_core.itzi_error import HotstartError

from .compute import rastermetrics


class RasterDomain:
    """Group all rasters for the raster domain.
    Store them as np.ndarray in a dict
    Include management of the masking and unmasking of arrays.
    """

    def __init__(self, dtype, arr_mask: np.ndarray, cell_shape: tuple[float, float]) -> None:
        # data type
        self.dtype = dtype
        # geographical data
        self.shape = arr_mask.shape
        self.dx, self.dy = cell_shape
        self.cell_area = self.dx * self.dy
        self.mask: np.ndarray = arr_mask

        # slice for a simple padding (allow stencil calculation on boundary)
        self.simple_pad = (slice(1, -1), slice(1, -1))
        # Fill values
        self.fill_values = {arr_def.key: arr_def.fill_value for arr_def in ARRAY_DEFINITIONS}

        # all keys that will be used for the arrays
        self.k_input = [
            arr_def.key for arr_def in ARRAY_DEFINITIONS if ArrayCategory.INPUT in arr_def.category
        ]
        self.k_internal = [
            arr_def.key
            for arr_def in ARRAY_DEFINITIONS
            if ArrayCategory.INTERNAL in arr_def.category
        ]
        # arrays gathering the cumulated water depth from corresponding array
        self.k_accum = [
            arr_def.key
            for arr_def in ARRAY_DEFINITIONS
            if ArrayCategory.ACCUMULATION in arr_def.category
        ]
        self.k_all = set(self.k_input + self.k_internal + self.k_accum)
        self.dtypes = {
            arr_def.key: np.dtype(self.dtype if arr_def.dtype is None else arr_def.dtype)
            for arr_def in ARRAY_DEFINITIONS
            if arr_def.key in self.k_all
        }
        # Instantiate arrays and padded arrays filled with zeros
        self.arr: dict[str, np.ndarray] = {}
        self.arrp: dict[str, np.ndarray] = {}
        self._create_arrays()

    def pad_array(self, arr) -> tuple[np.ndarray, np.ndarray]:
        """Return the original input array
        as a slice of a larger padded array with one cell
        """
        arr_p = np.pad(arr, 1, "edge")
        arr = arr_p[self.simple_pad]
        return arr, arr_p

    def _create_arrays(self) -> Self:
        """Instantiate masked arrays and padded arrays
        the unpadded arrays are a slice of the padded ones
        """
        for k in self.k_all:
            arr = np.full(fill_value=self.fill_values[k], shape=self.shape, dtype=self.dtypes[k])
            self.arr[k], self.arrp[k] = self.pad_array(arr)
        return self

    def update_mask(self, arr: np.ndarray) -> Self:
        """Create a mask array by marking NULL values from arr as True."""
        # self.mask[:] = np.isnan(arr)
        return self

    def mask_array(self, arr: np.ndarray, default_value: float) -> Self:
        """Replace NULL values in the input array by the default_value"""
        mask = np.logical_or(np.isnan(arr), self.mask)
        arr[mask] = default_value
        assert not np.any(np.isnan(arr))
        return self

    def unmask_array(self, arr: np.ndarray) -> np.ndarray:
        """Replace values in the input array by NULL values from mask"""
        unmasked_array = np.copy(arr)
        if np.issubdtype(unmasked_array.dtype, np.integer):
            unmasked_array = unmasked_array.astype(self.dtype)
        unmasked_array[self.mask] = np.nan
        return unmasked_array

    def update_ext_array(self) -> Self:
        """If one of the external input array has been updated,
        combine them into a unique array 'ext' in m/s.
        This applies for inputs that are needed to be taken into account
        at every timestep, like inflows from user or drainage.
        """
        rastermetrics.set_ext_array(
            self.arr["inflow"],
            self.arr["n_drain"],
            self.arr["eff_precip"],
            self.arr["ext"],
        )
        return self

    def swap_arrays(self, k1: str, k2: str) -> Self:
        """swap values of two arrays"""
        self.arr[k1], self.arr[k2] = self.arr[k2], self.arr[k1]
        self.arrp[k1], self.arrp[k2] = self.arrp[k2], self.arrp[k1]
        return self

    def update_array(self, arr_key: str, arr: np.ndarray) -> Self:
        """Update the values of an array with those of a given array."""
        if arr.shape != self.shape:
            raise ValueError(f"Updated values for array '{arr_key}' do not match domain size.")
        if arr_key == "water_surface_elevation":
            # Calculate actual depth and update the internal depth array
            arr = rastermetrics.calculate_h_from_wse(arr_wse=arr, arr_dem=self.get_array("dem"))
            arr_key = "water_depth"
        elif arr_key == "bctype":
            arr = self._prepare_bctype(arr)
        self.mask_array(arr, self.fill_values[arr_key])
        self.arr[arr_key][:], self.arrp[arr_key][:] = self.pad_array(arr)
        return self

    def _prepare_bctype(self, arr: np.ndarray) -> np.ndarray:
        """Mask and validate boundary codes before assigning them to uint8 storage."""
        if not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)):
            raise ValueError("Invalid values for 'bctype': expected an integer or floating array.")

        candidate = np.array(arr, copy=True)
        self.mask_array(candidate, self.fill_values["bctype"])
        valid = np.isin(candidate, (0, 1, 2, 3, 4))
        if not np.all(valid):
            invalid_values = candidate[~valid].reshape(-1)[:5].tolist()
            raise ValueError(f"Invalid values for 'bctype': {invalid_values}")
        return candidate

    def get_array(self, k: str) -> np.ndarray:
        """return the unpadded, masked array of key 'k'"""
        return self.arr[k]

    def get_padded(self, k: str) -> np.ndarray:
        """return the padded, masked array of key 'k'"""
        return self.arrp[k]

    def get_unmasked(self, k: str) -> np.ndarray:
        """return unpadded array with NaN"""
        return self.unmask_array(self.arr[k])

    def reset_accumulations(self) -> Self:
        """Set accumulation arrays to zeros"""
        for k in self.k_accum:
            self.arr[k][:] = 0.0
        return self

    def save_state(self) -> io.BytesIO:
        """Pack all the padded arrays of the domain in a npz file."""
        npz_file = io.BytesIO()
        arrays: dict[str, np.ndarray] = {"mask": self.mask}
        # Save padded arrays to preserve solver-computed boundary values
        arrays.update(self.arrp)
        np.savez(npz_file, allow_pickle=False, **arrays)
        npz_file.seek(0)
        return npz_file

    def load_state(self, npz_data: io.BytesIO) -> Self:
        """Restore domain arrays from an in-memory npz buffer.

        This method validates the loaded state against the current domain
        configuration before mutating any state.

        Args:
            npz_data: BytesIO containing an npz archive from save_state().

        Returns:
            Self for method chaining.

        Raises:
            HotstartError: If the state is incompatible with the current domain.
        """
        # Load the npz archive from the in-memory buffer
        npz_data.seek(0)
        try:
            npz = np.load(npz_data, allow_pickle=False)
        except Exception as e:
            raise HotstartError(f"Failed to load raster state: {e}") from e

        # Get the set of keys from the archive (excluding 'mask')
        archive_keys = set(npz.files) - {"mask"}
        expected_keys = self.k_all

        # Verify required keys are present
        missing_keys = expected_keys - archive_keys
        if missing_keys:
            raise HotstartError(
                f"Raster state missing required arrays: {', '.join(sorted(missing_keys))}"
            )

        # Check for unexpected keys (warning, not error - for forward compatibility)
        extra_keys = archive_keys - expected_keys
        if extra_keys:
            # Log or ignore extra keys - they're not harmful
            pass

        # Verify mask is present
        if "mask" not in npz.files:
            raise HotstartError("Raster state missing 'mask' array")

        # Verify mask shape matches
        stored_mask = npz["mask"]
        if stored_mask.shape != self.mask.shape:
            raise HotstartError(
                f"Mask shape mismatch: archive has {stored_mask.shape}, "
                f"domain expects {self.mask.shape}"
            )

        # Verify mask content matches
        if not np.array_equal(stored_mask, self.mask):
            raise HotstartError(
                "Mask content mismatch: the hotstart domain mask does not match "
                "the current domain mask"
            )

        # Padded shape is (rows+2, cols+2) due to 1-cell padding on all sides
        padded_shape = (self.shape[0] + 2, self.shape[1] + 2)

        # Verify all array shapes match the padded domain
        for key in expected_keys:
            stored_arr = npz[key]
            if stored_arr.shape != padded_shape:
                raise HotstartError(
                    f"Array '{key}' shape mismatch: archive has {stored_arr.shape}, "
                    f"domain expects padded shape {padded_shape}"
                )

        # Verify dtype compatibility (allow safe casting), while accepting valid
        # floating-point bctype arrays from legacy state archives.
        converted_arrays: dict[str, np.ndarray] = {}
        for key in expected_keys:
            stored_arr = npz[key]
            target_dtype = self.dtypes[key]
            if key == "bctype":
                if stored_arr.dtype != target_dtype and not np.issubdtype(
                    stored_arr.dtype, np.floating
                ):
                    raise HotstartError(
                        f"Array '{key}' dtype mismatch: archive has {stored_arr.dtype}, "
                        f"domain expects {target_dtype} (or a safely castable type)"
                    )
                candidate = np.array(stored_arr, copy=True)
                padded_mask = np.pad(self.mask, 1, mode="edge")
                masked = padded_mask
                if np.issubdtype(stored_arr.dtype, np.floating):
                    masked = np.logical_or(np.isnan(candidate), masked)
                candidate[masked] = self.fill_values["bctype"]
                valid = np.isin(candidate, (0, 1, 2, 3, 4))
                if not np.all(valid):
                    invalid_values = candidate[~valid].reshape(-1)[:5].tolist()
                    raise HotstartError(
                        f"Invalid values for 'bctype' in raster state: {invalid_values}"
                    )
                converted_arrays[key] = candidate.astype(target_dtype)
            elif not np.can_cast(stored_arr.dtype, target_dtype, casting="safe"):
                raise HotstartError(
                    f"Array '{key}' dtype mismatch: archive has {stored_arr.dtype}, "
                    f"domain expects {target_dtype} (or a safely castable type)"
                )
            else:
                converted_arrays[key] = stored_arr.astype(target_dtype)

        # All validations passed - restore the arrays
        for key in expected_keys:
            arrp = converted_arrays[key]
            # Store the padded array directly
            self.arrp[key][:] = arrp
            # Extract the interior (unpadded) slice for self.arr using simple_pad
            self.arr[key][:] = arrp[self.simple_pad]

        return self
