from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from itzi_core.itzi_error import NullError
from itzi_core.timed_inputs import TimedInputManager


class StubTimedArray:
    def __init__(self, array: np.ndarray, start: datetime, end: datetime) -> None:
        self.array = array
        self.source_start = start
        self.source_end = end
        self.arr_start = datetime.min
        self.arr_end = datetime.min
        self.arr: np.ndarray | None = None

    def is_valid(self, sim_time: datetime) -> bool:
        return self.arr_start <= sim_time < self.arr_end

    def get(self, sim_time: datetime) -> np.ndarray:
        self.arr_start = self.source_start
        self.arr_end = self.source_end
        self.arr = self.array
        return self.arr


class FailOnceTimedArray(StubTimedArray):
    def __init__(self, array: np.ndarray, start: datetime, end: datetime) -> None:
        super().__init__(array, start, end)
        self._fail_once = True

    def get(self, sim_time: datetime) -> np.ndarray:
        array = super().get(sim_time)
        if self._fail_once:
            self._fail_once = False
            return np.full_like(array, np.nan)
        return array


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


def test_read_at_rolls_back_caches_when_a_later_input_fails() -> None:
    start = datetime(2000, 1, 1)
    end = start + timedelta(seconds=10)
    dem = StubTimedArray(np.ones((1, 1)), start, end)
    rain = FailOnceTimedArray(np.ones((1, 1)), start, end)
    manager = TimedInputManager(
        {"dem": dem, "rain": rain},
        input_wse=False,
        end_time=end,
        mask=np.zeros((1, 1), dtype=bool),
    )

    with pytest.raises(RuntimeWarning, match=r"input map <rain>"):
        manager.read_at(start)

    assert not dem.is_valid(start)
    assert not rain.is_valid(start)

    updates, next_input = manager.read_at(start)

    assert [key for key, _ in updates] == ["dem", "rain"]
    assert next_input == end


@pytest.mark.parametrize(
    ("input_wse", "active_key", "inactive_key"),
    [
        (False, "water_depth", "water_surface_elevation"),
        (True, "water_surface_elevation", "water_depth"),
    ],
)
def test_read_at_selects_only_the_configured_stage_input(
    input_wse: bool,
    active_key: str,
    inactive_key: str,
) -> None:
    start = datetime(2000, 1, 1)
    inactive_boundary = start + timedelta(seconds=5)
    end = start + timedelta(seconds=10)
    values = {
        "water_depth": np.full((1, 1), 2.0, dtype=np.float32),
        "water_surface_elevation": np.full((1, 1), 12.0, dtype=np.float32),
    }
    timed_arrays = {
        "dem": StubTimedArray(np.full((1, 1), 10.0, dtype=np.float32), start, end),
        "water_depth": StubTimedArray(
            values["water_depth"],
            start,
            end if active_key == "water_depth" else inactive_boundary,
        ),
        "water_surface_elevation": StubTimedArray(
            values["water_surface_elevation"],
            start,
            end if active_key == "water_surface_elevation" else inactive_boundary,
        ),
    }
    manager = TimedInputManager(
        timed_arrays,
        input_wse=input_wse,
        end_time=end,
        mask=np.zeros((1, 1), dtype=bool),
    )

    updates, next_input = manager.read_at(start)

    update_map = dict(updates)
    assert list(update_map) == ["dem", active_key]
    np.testing.assert_allclose(update_map[active_key], values[active_key])
    assert timed_arrays[inactive_key].arr is None
    assert next_input == end


@pytest.mark.parametrize(
    ("key", "source_value", "expected_value"),
    [
        ("rain", 3_600_000.0, 1.0),
        ("hydraulic_conductivity", 3_600_000.0, 1.0),
        ("infiltration", 3_600_000.0, 1.0),
        ("losses", 3_600_000.0, 1.0),
        ("capillary_pressure", 1_000.0, 1.0),
        ("inflow", 0.125, 0.125),
    ],
)
def test_read_at_converts_each_input_unit_family(
    key: str,
    source_value: float,
    expected_value: float,
) -> None:
    start = datetime(2000, 1, 1)
    end = start + timedelta(seconds=10)
    manager = TimedInputManager(
        {
            "dem": StubTimedArray(np.ones((1, 1), dtype=np.float32), start, end),
            key: StubTimedArray(np.full((1, 1), source_value, dtype=np.float32), start, end),
        },
        input_wse=False,
        end_time=end,
        mask=np.zeros((1, 1), dtype=bool),
    )

    updates, _ = manager.read_at(start)

    np.testing.assert_allclose(
        dict(updates)[key],
        np.full((1, 1), expected_value, dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("key", "mask", "source", "expected_error", "message"),
    [
        (
            "rain",
            np.array([[True, False], [False, False]], dtype=bool),
            np.array([[99.0, np.nan], [np.nan, np.nan]], dtype=np.float32),
            RuntimeWarning,
            r"input map <rain> contains only NULL/NaN cells inside the active domain",
        ),
        (
            "dem",
            np.ones((2, 2), dtype=bool),
            np.ones((2, 2), dtype=np.float32),
            NullError,
            r"active domain contains no cells for input map <dem>",
        ),
        (
            "rain",
            np.array([[True, False], [False, False]], dtype=bool),
            np.array([[np.nan, 2.0], [np.nan, np.nan]], dtype=np.float32),
            None,
            None,
        ),
    ],
)
def test_validation_uses_active_cells_only(
    key: str,
    mask: np.ndarray,
    source: np.ndarray,
    expected_error: type[Exception] | None,
    message: str | None,
) -> None:
    start = datetime(2000, 1, 1)
    end = start + timedelta(seconds=10)
    timed_arrays = {
        "dem": StubTimedArray(np.ones((2, 2), dtype=np.float32), start, end),
    }
    timed_arrays[key] = StubTimedArray(source, start, end)
    manager = TimedInputManager(
        timed_arrays,
        input_wse=False,
        end_time=end,
        mask=mask,
    )

    if expected_error is not None:
        with pytest.raises(expected_error, match=message):
            manager.read_at(start)
        return

    updates, _ = manager.read_at(start)
    accepted = dict(updates)["rain"]
    assert np.isnan(accepted[0, 0])
    assert accepted[0, 1] == pytest.approx(2.0 / (1000 * 3600))


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
