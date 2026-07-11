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

import hashlib
import os
from datetime import datetime, timedelta
from collections import namedtuple

import pytest
import numpy as np

from itzi_core.array_definitions import ARRAY_DEFINITIONS, ArrayCategory


class Helpers:
    @staticmethod
    def get_rmse(model, ref):
        """return root mean square error"""
        return np.sqrt(np.mean((model - ref) ** 2))

    @staticmethod
    def get_nse(model, ref):
        """Nash-Sutcliffe Efficiency"""
        noise = np.mean((ref - model) ** 2)
        information = np.mean((ref - np.mean(ref)) ** 2)
        return 1 - (noise / information)

    @staticmethod
    def get_rsr(model, ref):
        """RMSE/StdDev ratio"""
        rmse = Helpers.get_rmse(model, ref)
        return rmse / np.std(ref)

    @staticmethod
    def roughness(timeseries):
        """Sum of the squared difference of
        the normalized differences.
        """
        f = timeseries.diff()
        normed_f = (f - f.mean()) / f.std()
        return (normed_f.diff() ** 2).sum()

    @staticmethod
    def sha256(file_path):
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    @staticmethod
    def md5(file_path):
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @staticmethod
    def make_input_map_names(**overrides) -> dict[str, str | None]:
        """Generate default input_map_names dict from ARRAY_DEFINITIONS.

        Keys set to None are inactive; keys set to a truthy string activate features.
        """
        names = {ad.key: None for ad in ARRAY_DEFINITIONS if ArrayCategory.INPUT in ad.category}
        names.update(overrides)
        return names

    @staticmethod
    def make_output_map_names(prefix: str, keys: list[str]) -> dict[str, str | None]:
        """Generate output_map_names dict from ARRAY_DEFINITIONS.

        Args:
            prefix: Prefix for output map names (e.g., "out_5by5")
            keys: List of output keys to activate

        Returns:
            Dict with all OUTPUT-category keys, activated ones set to "prefix_keyname"
        """
        names = {ad.key: None for ad in ARRAY_DEFINITIONS if ArrayCategory.OUTPUT in ad.category}
        for k in keys:
            names[k] = f"{prefix}_{k}"
        return names


@pytest.fixture(scope="session")
def helpers():
    return Helpers


@pytest.fixture(scope="session")
def test_data_path():
    """Path to the permanent test data directory."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, "test_data")


@pytest.fixture(scope="session")
def test_data_temp_path():
    """Directory where generated test data resides."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    temp_path = os.path.join(dir_path, "test_data_temp")
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    return temp_path


@pytest.fixture(
    scope="module", params=[datetime(year=2020, month=3, day=23, hour=10), timedelta(seconds=0)]
)
def sim_time(request):
    """Fixture for simulation time in parametrized tests.

    Provides both datetime and timedelta representations for testing
    CSV vector output with different time formats.
    """
    return request.param


Domain5by5Data = namedtuple(
    "Domain5by5Data",
    [
        "domain_data",
        "arr_dem_flat",
        "arr_dem_high",
        "arr_n",
        "arr_start_h",
        "arr_start_wse",
        "arr_mask",
        "arr_bctype",
        "arr_rain",
        "arr_inf",
        "arr_loss",
        "arr_inflow",
    ],
)


@pytest.fixture(scope="module")
def domain_5by5() -> Domain5by5Data:
    """Create a 5x5 domain with all base arrays."""
    rows, cols = 5, 5
    north, south, east, west = 50.0, 0.0, 50.0, 0.0

    from itzi_core.providers.domain_data import DomainData

    domain_data = DomainData(
        north=north, south=south, east=east, west=west, rows=rows, cols=cols, crs_wkt=""
    )

    arr_dem_flat = np.zeros(domain_data.shape, dtype=np.float32)
    arr_dem_high = np.full(domain_data.shape, 132.0, dtype=np.float32)
    arr_n = np.full(domain_data.shape, 0.05, dtype=np.float32)

    arr_start_h = np.zeros(domain_data.shape, dtype=np.float32)
    arr_start_h[2, 2] = 0.2

    arr_start_wse = np.zeros(domain_data.shape, dtype=np.float32)
    arr_start_wse[2, 2] = 132.2

    arr_mask = np.full(domain_data.shape, False, dtype=np.bool_)

    arr_bctype = np.zeros(domain_data.shape, dtype=np.float32)
    arr_bctype[0, :] = 2
    arr_bctype[4, :] = 2
    arr_bctype[:, 0] = 2
    arr_bctype[:, 4] = 2

    arr_rain = np.full(domain_data.shape, 10.0 / (1000 * 3600), dtype=np.float32)
    arr_inf = np.full(domain_data.shape, 2.0 / (1000 * 3600), dtype=np.float32)
    arr_loss = np.full(domain_data.shape, 1.5 / (1000 * 3600), dtype=np.float32)
    arr_inflow = np.full(domain_data.shape, 0.1, dtype=np.float32)

    return Domain5by5Data(
        domain_data=domain_data,
        arr_dem_flat=arr_dem_flat,
        arr_dem_high=arr_dem_high,
        arr_n=arr_n,
        arr_start_h=arr_start_h,
        arr_start_wse=arr_start_wse,
        arr_mask=arr_mask,
        arr_bctype=arr_bctype,
        arr_rain=arr_rain,
        arr_inf=arr_inf,
        arr_loss=arr_loss,
        arr_inflow=arr_inflow,
    )
