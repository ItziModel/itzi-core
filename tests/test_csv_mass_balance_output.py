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

import csv
from datetime import UTC, datetime

import pytest

from itzi_core.data_containers import MassBalanceData
from itzi_core.providers.base import MassBalanceOutputProvider
from itzi_core.providers.csv_mass_balance_output import CSVMassBalanceOutputProvider


@pytest.fixture
def provider_fixture(tmp_path):
    return {"file_name": str(tmp_path / "stats.csv")}


def test_init_with_custom_filename(provider_fixture):
    provider = CSVMassBalanceOutputProvider(
        file_name=provider_fixture["file_name"],
    )
    assert isinstance(provider, MassBalanceOutputProvider)
    assert provider.file_name == provider_fixture["file_name"]


def test_init_with_default_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    provider = CSVMassBalanceOutputProvider(
        file_name="",
    )
    assert provider.file_name.endswith("_stats.csv")


def test_log_absolute_time(provider_fixture):
    provider = CSVMassBalanceOutputProvider(
        file_name=provider_fixture["file_name"],
    )
    test_time = datetime.now(UTC)
    test_data = MassBalanceData(
        simulation_time=test_time,
        average_timestep=12.42345,
        timesteps=34,
        boundary_volume=123.456789,
        rainfall_volume=12.34567,
        infiltration_volume=-12.434567,
        inflow_volume=12.34567,
        losses_volume=-12.34567,
        drainage_network_volume=12.34567,
        domain_volume=12.34567,
        volume_change=12.34567,
        created_volume=12.34567,
        created_volume_ratio=0.123456,
        closure_residual=1.234567890123456e-10,
        relative_closure_error=9.876543210987654e-12,
    )

    provider.log(test_data)
    with open(provider_fixture["file_name"], "r") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == list(MassBalanceData.model_fields)
        row = next(reader)

    assert row["simulation_time"] == str(test_time)
    assert float(row["boundary_volume"]) == test_data.boundary_volume
    assert float(row["rainfall_volume"]) == test_data.rainfall_volume
    assert row["timesteps"] == "34"
    assert float(row["created_volume_ratio"]) == test_data.created_volume_ratio
    assert float(row["closure_residual"]) == test_data.closure_residual
    assert float(row["relative_closure_error"]) == test_data.relative_closure_error
