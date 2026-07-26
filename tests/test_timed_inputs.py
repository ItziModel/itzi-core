from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from itzi_core.timed_inputs import TimedInputManager


class StubTimedArray:
    def __init__(self, array: np.ndarray, start: datetime, end: datetime) -> None:
        self.array = array
        self.source_start = start
        self.source_end = end
        self.arr_start = datetime.min
        self.arr_end = datetime.min

    def is_valid(self, sim_time: datetime) -> bool:
        return self.arr_start <= sim_time < self.arr_end

    def get(self, sim_time: datetime) -> np.ndarray:
        self.arr_start = self.source_start
        self.arr_end = self.source_end
        return self.array


def test_read_at_orders_dem_converts_rates_and_owns_arrays() -> None:
    start = datetime(2000, 1, 1)
    end = start + timedelta(seconds=10)
    dem = np.full((2, 2), 5.0, dtype=np.float32)
    rain = np.full((2, 2), 360.0, dtype=np.float32)
    dem.setflags(write=False)
    rain.setflags(write=False)
    manager = TimedInputManager(
        {
            "rain": StubTimedArray(rain, start, end),
            "dem": StubTimedArray(dem, start, end),
        },
        input_wse=False,
        end_time=end,
        mask=np.zeros((2, 2), dtype=bool),
    )

    updates, next_input = manager.read_at(start)

    assert [key for key, _ in updates] == ["dem", "rain"]
    assert next_input == end
    assert updates[0][1].flags.writeable
    assert updates[1][1].flags.writeable
    np.testing.assert_allclose(updates[1][1], 0.0001)
    updates[0][1][0, 0] = 99.0
    updates[1][1][0, 0] = 99.0
    assert dem[0, 0] == 5.0
    assert rain[0, 0] == 360.0


def test_prime_at_aligns_caches_without_returning_updates() -> None:
    start = datetime(2000, 1, 1)
    boundary = start + timedelta(seconds=5)
    end = start + timedelta(seconds=10)
    dem = StubTimedArray(np.ones((1, 1)), start, end)
    rain = StubTimedArray(np.ones((1, 1)), start, boundary)
    manager = TimedInputManager(
        {"dem": dem, "rain": rain},
        input_wse=False,
        end_time=end,
        mask=np.zeros((1, 1), dtype=bool),
    )

    assert manager.prime_at(start) == boundary
    assert dem.is_valid(start)
    assert rain.is_valid(start)
