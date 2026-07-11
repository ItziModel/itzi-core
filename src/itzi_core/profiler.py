"""
This module provides profiling capabilities.

It uses pyinstrument to profile the execution of simulations.
Profiling is activated by setting the environment variable ITZI_PROFILE=1.

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

import os
from pathlib import Path
from contextlib import contextmanager

# Attempt to import pyinstrument
try:
    from pyinstrument import Profiler

    PYINSTRUMENT_AVAILABLE = True
except ImportError:
    PYINSTRUMENT_AVAILABLE = False
    Profiler = None


@contextmanager
def profile_context(file_path: Path = None):
    """
    A context manager for profiling code blocks.

    If the environment variable ITZI_PROFILE is set to '1',
    this context manager will start the pyinstrument profiler
    before yielding control, and stop/print the results when
    the block exits.

    If ITZI_PROFILE is not set or pyinstrument is not available,
    this context manager does nothing.

    Usage:
        with profile_context():
            # Code to be profiled (or not)
            run_simulation()
    """
    profiler_active = os.environ.get("ITZI_PROFILE") == "1" and PYINSTRUMENT_AVAILABLE

    if profiler_active:
        profiler = Profiler()
        profiler.start()
    try:
        yield
    finally:
        if profiler_active:
            profiler.stop()
            if file_path:
                Path(file_path).write_text(profiler.output_text(unicode=False, color=False))
            else:
                print(profiler.output_text(unicode=True, color=True))
