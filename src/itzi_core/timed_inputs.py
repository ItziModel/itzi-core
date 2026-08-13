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

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from itzi_core.itzi_error import NullError
from itzi_core.rasterdomain import TimedArray

logger = logging.getLogger(__name__)

_RATE_INPUTS = frozenset({"rain", "hydraulic_conductivity", "infiltration", "losses"})
_LENGTH_INPUTS = frozenset({"capillary_pressure"})


class TimedInputManager:
    """Fetch, validate, convert, and cache timed input arrays without applying them."""

    def __init__(
        self,
        timed_arrays: dict[str, TimedArray],
        input_wse: bool,
        end_time: datetime,
        mask: np.ndarray,
    ) -> None:
        self.timed_arrays = timed_arrays
        self.input_wse = input_wse
        self.end_time = end_time
        self.mask = mask

    def read_at(self, sim_time: datetime) -> tuple[list[tuple[str, np.ndarray]], datetime]:
        """Prepare detached arrays to apply at ``sim_time`` and the next input boundary."""
        return self._prepare_at(sim_time)

    def prime_at(self, sim_time: datetime) -> datetime:
        """Align input caches to ``sim_time`` without replacing restored raster state."""
        _, next_input = self._prepare_at(sim_time, update_keys=frozenset())
        return next_input

    def prepare_resume_at(
        self,
        sim_time: datetime,
        changed_keys: set[str],
    ) -> tuple[list[tuple[str, np.ndarray]], datetime]:
        """Prime all inputs and return updates only for sources changed on resume."""
        return self._prepare_at(sim_time, update_keys=changed_keys)

    def _prepare_at(
        self,
        sim_time: datetime,
        update_keys: set[str] | frozenset[str] | None = None,
    ) -> tuple[list[tuple[str, np.ndarray]], datetime]:
        if sim_time >= self.end_time:
            return [], self.end_time

        cache_state = {
            key: (timed_array.arr_start, timed_array.arr_end, timed_array.arr)
            for key, timed_array in self.timed_arrays.items()
        }
        try:
            updates: list[tuple[str, np.ndarray]] = []
            # WSE conversion depends on the DEM at the same time label.
            self._prepare_array("dem", sim_time, updates, update_keys)
            for key in self.timed_arrays:
                if key == "dem" or not self._is_active(key):
                    continue
                self._prepare_array(key, sim_time, updates, update_keys)

            next_input = self.end_time
            for key, timed_array in self.timed_arrays.items():
                if self._is_active(key) and timed_array.is_valid(sim_time):
                    next_input = min(next_input, timed_array.arr_end)
            return updates, next_input
        except Exception:
            # TimedArray.get() updates its cache before validation succeeds. Restore all
            # cache entries so a retry cannot skip an input that was never applied.
            for key, (arr_start, arr_end, array) in cache_state.items():
                timed_array = self.timed_arrays[key]
                timed_array.arr_start = arr_start
                timed_array.arr_end = arr_end
                timed_array.arr = array
            raise

    def _prepare_array(
        self,
        key: str,
        sim_time: datetime,
        updates: list[tuple[str, np.ndarray]],
        update_keys: set[str] | frozenset[str] | None,
    ) -> None:
        timed_array = self.timed_arrays[key]
        if timed_array.is_valid(sim_time):
            return
        # RasterDomain.mask_array mutates its input. Own the update before returning it.
        array = np.array(self._convert(key, timed_array.get(sim_time)), copy=True)
        self._validate_array(key, array, sim_time)
        logger.debug("%s: update input array <%s>", sim_time, key)
        if update_keys is None or key in update_keys:
            updates.append((key, array))

    def _is_active(self, key: str) -> bool:
        return not (
            (key == "water_depth" and self.input_wse)
            or (key == "water_surface_elevation" and not self.input_wse)
        )

    def _convert(self, key: str, array: np.ndarray) -> np.ndarray:
        if key in _RATE_INPUTS:
            return array / (1000 * 3600)
        if key in _LENGTH_INPUTS:
            return array / 1000
        return array

    def _validate_array(self, key: str, array: np.ndarray, sim_time: datetime) -> None:
        active_cells = array[~self.mask]
        if np.count_nonzero(np.isfinite(active_cells)) > 0:
            return
        if active_cells.size == 0:
            message = f"{sim_time}: active domain contains no cells for input map <{key}>"
        else:
            message = f"{sim_time}: input map <{key}> contains only NULL/NaN cells inside the active domain"
        if key == "dem":
            raise NullError(message)
        raise RuntimeWarning(message)
