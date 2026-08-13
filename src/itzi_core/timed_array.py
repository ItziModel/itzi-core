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

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, Self

import numpy as np

if TYPE_CHECKING:
    from itzi_core.providers.base import RasterInputProvider


class TimedArraySource(Protocol):
    arr_start: datetime
    arr_end: datetime
    arr: np.ndarray | None

    def is_valid(self, sim_time: datetime) -> bool: ...

    def get(self, sim_time: datetime) -> np.ndarray: ...


class TimedArray:
    """A container for np.ndarray with time information.
    Update the array value according to the simulation time.
    array is accessed via get()
    """

    def __init__(
        self,
        mkey: str,
        raster_provider: RasterInputProvider,
        default_array_func: Callable[[], np.ndarray],
    ) -> None:
        assert isinstance(mkey, str), "not a string!"
        assert hasattr(default_array_func, "__call__"), "not a function!"
        self.mkey = mkey  # An array identifier
        self.raster_provider = raster_provider
        # A function to generate a default array
        self.default_array_func = default_array_func
        # default values for start and end
        # intended to trigger update when is_valid() is first called
        self.arr_start = datetime(1, 1, 2)
        self.arr_end = datetime(1, 1, 1)
        # Necessary for BMI implementation
        self.origin = raster_provider.get_origin()
        # The array is loaded lazily on first access.
        self.arr: np.ndarray | None = None

    def get(self, sim_time: datetime) -> np.ndarray:
        """Return a numpy array valid for the given time
        If the array stored is not valid, update the values of the object
        """
        assert isinstance(sim_time, datetime), "not a datetime object!"
        if not self.is_valid(sim_time):
            self.update_values(sim_time)
        assert self.arr is not None
        return self.arr

    def is_valid(self, sim_time: datetime) -> bool:
        """input being a time in datetime
        If the current stored array is within the half-open range [start, end),
        return True
        If not return False
        """
        return bool(self.arr_start <= sim_time < self.arr_end)

    def update_values(self, sim_time: datetime) -> Self:
        """Update array, start_time and end_time from provider
        if the provider returns None, set array to default value
        """
        # Retrieve values
        arr, arr_start, arr_end = self.raster_provider.get_array(self.mkey, sim_time)
        # set to default if no array retrieved
        if not isinstance(arr, np.ndarray):
            arr = self.default_array_func()
        # check retrieved values
        assert isinstance(arr_start, datetime), "not a datetime object!"
        assert isinstance(arr_end, datetime), "not a datetime object!"
        assert arr_start <= sim_time < arr_end, "wrong time retrieved!"
        # update object values
        self.arr_start = arr_start
        self.arr_end = arr_end
        self.arr = arr
        return self
