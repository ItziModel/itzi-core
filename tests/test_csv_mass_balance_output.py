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
        volume_error=12.34567,
        percent_error=0.123456,
    )

    provider.log(test_data)
    with open(provider_fixture["file_name"], "r") as f:
        lines = f.readlines()
        assert str(test_time) in lines[1]  # datetime formatting
        assert "123.456789" in lines[1]  # float formatting
        assert "12.345670" in lines[1]  # float formatting
        assert "34" in lines[1]  # int formatting
        assert "12.35%" in lines[1]  # percentage formatting
