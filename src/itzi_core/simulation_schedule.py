"""Simulation clock and named deadline management."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta
from types import MappingProxyType

from itzi_core.itzi_error import HotstartError


class SimulationSchedule:
    """Own the simulation clock, selected interval, and persistent deadlines."""

    _EVENTS = frozenset({"end", "input", "hydrology", "drainage", "record"})

    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        record_step: timedelta,
        has_drainage: bool,
    ) -> None:
        self._start_time = start_time
        self._end_time = end_time
        self._now = start_time
        # Preserve the legacy initial-report placeholder until the first interval is selected.
        self._dt = timedelta(milliseconds=1)
        self._deadlines = {
            "end": end_time,
            "input": end_time,
            "hydrology": start_time,
            "drainage": start_time if has_drainage else end_time,
            "record": start_time + record_step,
        }
        self._stop_limit: datetime | None = None

    @property
    def now(self) -> datetime:
        return self._now

    @property
    def start_time(self) -> datetime:
        return self._start_time

    @property
    def end_time(self) -> datetime:
        return self._end_time

    @property
    def dt(self) -> timedelta:
        return self._dt

    @property
    def deadlines(self) -> Mapping[str, datetime]:
        return MappingProxyType(self._deadlines)

    @property
    def nextstep(self) -> datetime:
        return min(self._deadlines.values())

    def deadline(self, event: str) -> datetime:
        self._validate_event(event)
        return self._deadlines[event]

    def set_deadline(self, event: str, when: datetime) -> None:
        self._validate_event(event)
        self._deadlines[event] = when

    def advance_event(self, event: str, interval: timedelta) -> None:
        self._validate_event(event)
        self._deadlines[event] += interval

    def select_step_end(self, surface_flow_end: datetime) -> datetime:
        """Choose the earliest future physical or scheduled boundary."""
        self._deadlines["record"] = min(self._deadlines["end"], self._deadlines["record"])
        self._deadlines["input"] = min(self._deadlines["end"], self._deadlines["input"])
        self._deadlines["hydrology"] = min(self._deadlines["hydrology"], self._deadlines["input"])

        candidates = [surface_flow_end]
        candidates.extend(when for when in self._deadlines.values() if when > self._now)
        if self._stop_limit is not None and self._stop_limit > self._now:
            candidates.append(self._stop_limit)
        step_end = min(candidates)
        self._dt = step_end - self._now
        return step_end

    def commit_step(self, step_end: datetime) -> None:
        self._now = step_end

    @contextmanager
    def stop_at(self, target: datetime) -> Iterator[None]:
        """Temporarily constrain interval selection without changing hotstart state."""
        previous_limit = self._stop_limit
        self._stop_limit = target if previous_limit is None else min(previous_limit, target)
        try:
            yield
        finally:
            self._stop_limit = previous_limit

    def snapshot_deadlines(self) -> dict[str, datetime]:
        return dict(self._deadlines)

    def restore(
        self,
        sim_time: datetime,
        dt: timedelta,
        deadlines: Mapping[str, datetime],
    ) -> None:
        """Restore persistent scheduler state from a version 1 hotstart payload."""
        restored = dict(deadlines)
        restored.pop("temp_end", None)
        required = {"end", "hydrology", "drainage", "record"}
        missing = required - restored.keys()
        if missing:
            raise HotstartError(
                f"Hotstart schedule missing deadlines: {', '.join(sorted(missing))}"
            )
        restored.setdefault("input", restored["end"])

        unknown = set(restored) - self._EVENTS
        if unknown:
            raise HotstartError(
                f"Hotstart schedule has unknown deadlines: {', '.join(sorted(unknown))}"
            )
        for event in self._EVENTS:
            if restored[event] < sim_time:
                raise HotstartError(
                    f"Hotstart {event} deadline {restored[event]} precedes simulation time {sim_time}"
                )

        self._now = sim_time
        self._dt = dt
        self._deadlines = {event: restored[event] for event in self._EVENTS}

    def _validate_event(self, event: str) -> None:
        if event not in self._EVENTS:
            raise ValueError(f"Unknown simulation event: {event}")
