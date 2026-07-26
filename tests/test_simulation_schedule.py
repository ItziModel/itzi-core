from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from itzi_core.itzi_error import HotstartError
from itzi_core.simulation_schedule import SimulationSchedule


@pytest.fixture
def times() -> tuple[datetime, datetime]:
    start = datetime(2000, 1, 1)
    return start, start + timedelta(seconds=30)


def test_initial_deadlines_and_nextstep_are_normalized(times) -> None:
    start, end = times
    schedule = SimulationSchedule(start, end, timedelta(seconds=10), has_drainage=False)

    assert schedule.now == start
    assert schedule.dt == timedelta(milliseconds=1)
    assert schedule.deadline("drainage") == end
    assert schedule.nextstep == start


def test_step_selection_clamps_input_and_aligns_hydrology(times) -> None:
    start, end = times
    input_boundary = start + timedelta(seconds=5)
    schedule = SimulationSchedule(start, end, timedelta(seconds=40), has_drainage=False)
    schedule.set_deadline("input", input_boundary)
    schedule.set_deadline("hydrology", start + timedelta(seconds=20))

    step_end = schedule.select_step_end(start + timedelta(seconds=15))

    assert step_end == input_boundary
    assert schedule.dt == timedelta(seconds=5)
    assert schedule.deadline("hydrology") == input_boundary
    assert schedule.deadline("record") == end


def test_temporary_stop_limit_is_exception_safe_and_not_snapshotted(times) -> None:
    start, end = times
    schedule = SimulationSchedule(start, end, timedelta(seconds=10), has_drainage=False)
    target = start + timedelta(seconds=3)

    with pytest.raises(RuntimeError, match="expected"):
        with schedule.stop_at(target):
            assert schedule.select_step_end(start + timedelta(seconds=20)) == target
            assert "temp_end" not in schedule.snapshot_deadlines()
            raise RuntimeError("expected")

    assert schedule.select_step_end(start + timedelta(seconds=20)) == start + timedelta(seconds=10)


def test_restore_uses_legacy_input_fallback_and_ignores_temp_end(times) -> None:
    start, end = times
    restored_time = start + timedelta(seconds=10)
    schedule = SimulationSchedule(start, end, timedelta(seconds=10), has_drainage=False)

    schedule.restore(
        restored_time,
        timedelta(seconds=0.2),
        {
            "end": end,
            "hydrology": start + timedelta(seconds=20),
            "drainage": end,
            "record": start + timedelta(seconds=20),
            "temp_end": start + timedelta(seconds=12),
        },
    )

    assert schedule.now == restored_time
    assert schedule.dt == timedelta(seconds=0.2)
    assert schedule.deadline("input") == end
    assert "temp_end" not in schedule.snapshot_deadlines()
    assert schedule.nextstep == schedule.deadline("hydrology")


def test_restore_rejects_unknown_or_stale_deadlines(times) -> None:
    start, end = times
    schedule = SimulationSchedule(start, end, timedelta(seconds=10), has_drainage=False)
    deadlines = {
        "end": end,
        "input": end,
        "hydrology": start + timedelta(seconds=20),
        "drainage": end,
        "record": start + timedelta(seconds=20),
    }

    with pytest.raises(HotstartError, match="unknown"):
        schedule.restore(start, timedelta(seconds=1), {**deadlines, "other": end})
    with pytest.raises(HotstartError, match="precedes"):
        schedule.restore(
            start + timedelta(seconds=10),
            timedelta(seconds=1),
            {**deadlines, "input": start + timedelta(seconds=5)},
        )
    with pytest.raises(HotstartError, match="record deadline .*precedes"):
        schedule.restore(
            start + timedelta(seconds=10),
            timedelta(seconds=1),
            {**deadlines, "record": start + timedelta(seconds=5)},
        )
