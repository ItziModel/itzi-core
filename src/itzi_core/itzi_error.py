"""
Copyright (C) 2015-2026 Laurent G. Courty

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 2.1
of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.
"""


class NullError(RuntimeError):
    """Raised when null values is detected in simulation"""


class DtError(RuntimeError):
    """Error related to time-step calculation"""


class MassBalanceError(RuntimeError):
    """Raised when mass balance error exceeds threshold"""


class HotstartError(RuntimeError):
    """Raised when hotstart file operations fail."""
