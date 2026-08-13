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

import csv
import numbers
from datetime import datetime

from itzi_core.data_containers import MassBalanceData
from itzi_core.providers.base import MassBalanceOutputProvider


class CSVMassBalanceOutputProvider(MassBalanceOutputProvider):
    """Writes pre-calculated mass balance data to a CSV file."""

    def __init__(
        self,
        file_name: str,
    ):
        """Initialize the provider and create the output file with headers."""
        self.fields = list(MassBalanceData.model_fields.keys())
        self.file_name = self._set_file_name(file_name)
        self._create_file()

    def _set_file_name(self, file_name: str) -> str:
        """Generate output file name"""
        if not file_name:
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")  # noqa: DTZ005
            file_name = f"{timestamp}_stats.csv"
        return file_name

    def _create_file(self) -> None:
        """Create a csv file and write headers"""
        with open(self.file_name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()

    def log(self, report_data: MassBalanceData) -> None:
        """Writes a single line of data to the CSV file."""
        line_to_write = {}

        for key, value in report_data.model_dump().items():
            if value != value:  # noqa: PLR0124  # test for NaN
                line_to_write[key] = "-"
            elif isinstance(value, numbers.Real) and not isinstance(value, int):
                line_to_write[key] = f"{value:.17g}"
            else:
                line_to_write[key] = value

        with open(self.file_name, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(line_to_write)
