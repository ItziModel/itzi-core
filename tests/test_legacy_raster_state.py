"""Regression tests for legacy raster-state NPZ member names."""

import io

import numpy as np
import pytest

from itzi_core.rasterdomain import RasterDomain

LEGACY_MEMBER_NAMES = {
    "ground_elevation": "dem",
    "boundary_value": "bcval",
    "boundary_type": "bctype",
    "effective_precipitation": "eff_precip",
    "flow_depth_east": "hfe",
    "flow_depth_south": "hfs",
    "old_discharge_east": "qe",
    "old_discharge_south": "qs",
    "new_discharge_east": "qe_new",
    "new_discharge_south": "qs_new",
    "max_water_depth": "hmax",
    "flow_speed": "v",
    "flow_velocity_direction": "vdir",
    "max_flow_speed": "vmax",
    "drainage_inflow": "n_drain",
}


def test_load_state_migrates_legacy_npz_members() -> None:
    source = RasterDomain(
        dtype=np.float32,
        arr_mask=np.zeros((3, 3), dtype=bool),
        cell_shape=(1.0, 1.0),
    )
    for value, key in enumerate(LEGACY_MEMBER_NAMES, start=1):
        array = np.full(source.shape, value, dtype=source.dtypes[key])
        if key == "boundary_type":
            array %= 5
        source.update_array(key, array)

    saved_state = source.save_state()
    saved_state.seek(0)
    with np.load(saved_state, allow_pickle=False) as npz:
        members = {LEGACY_MEMBER_NAMES.get(key, key): npz[key] for key in npz.files}
    members["bctype"] = members["bctype"].astype(np.float64)
    legacy_state = io.BytesIO()
    np.savez(legacy_state, allow_pickle=False, **members)
    legacy_state.seek(0)

    restored = RasterDomain(
        dtype=np.float32,
        arr_mask=source.mask.copy(),
        cell_shape=(source.dx, source.dy),
    )
    with pytest.warns(DeprecationWarning, match="deprecated NPZ member names"):
        restored.load_state(legacy_state)

    for key in source.k_all:
        np.testing.assert_array_equal(restored.get_padded(key), source.get_padded(key))
