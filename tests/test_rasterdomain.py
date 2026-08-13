"""Tests for RasterDomain storage dtypes and boundary-type validation."""

import io

import numpy as np
import pytest

from itzi_core.itzi_error import HotstartError
from itzi_core.rasterdomain import RasterDomain


def make_domain(
    dtype: type[np.floating] = np.float32, mask: np.ndarray | None = None
) -> RasterDomain:
    if mask is None:
        mask = np.zeros((3, 3), dtype=bool)
    return RasterDomain(dtype=dtype, arr_mask=mask, cell_shape=(1.0, 1.0))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_bctype_is_the_only_integer_storage_array(dtype):
    domain = make_domain(dtype)

    assert domain.get_array("bctype").dtype == np.dtype(np.uint8)
    assert domain.get_padded("bctype").dtype == np.dtype(np.uint8)
    for key in domain.k_all - {"bctype"}:
        assert domain.get_array(key).dtype == np.dtype(dtype)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32, np.float64])
def test_bctype_accepts_exact_codes(dtype):
    domain = make_domain()
    values = np.array([[0, 1, 2], [3, 4, 0], [1, 2, 3]], dtype=dtype)

    domain.update_array("bctype", values)

    np.testing.assert_array_equal(domain.get_array("bctype"), values)
    assert domain.get_array("bctype").dtype == np.dtype(np.uint8)


def test_bctype_masks_nan_and_domain_mask_before_validation():
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    domain = make_domain(mask=mask)
    values = np.zeros((3, 3), dtype=np.float64)
    values[0, 0] = 99
    values[1, 1] = np.nan

    domain.update_array("bctype", values)

    assert domain.get_array("bctype")[0, 0] == 0
    assert domain.get_array("bctype")[1, 1] == 0
    assert domain.get_unmasked("bctype").dtype == np.dtype(np.float32)
    assert np.isnan(domain.get_unmasked("bctype")[0, 0])


@pytest.mark.parametrize("invalid_value", [2.5, np.inf, -1, 5])
def test_invalid_bctype_update_does_not_mutate_state(invalid_value):
    domain = make_domain()
    domain.update_array("bctype", np.ones(domain.shape, dtype=np.uint8))
    before = domain.get_array("bctype").copy()
    before_padded = domain.get_padded("bctype").copy()
    values = np.ones(domain.shape, dtype=np.float64)
    values[1, 1] = invalid_value

    with pytest.raises(ValueError, match="bctype"):
        domain.update_array("bctype", values)

    np.testing.assert_array_equal(domain.get_array("bctype"), before)
    np.testing.assert_array_equal(domain.get_padded("bctype"), before_padded)


def make_archive(domain: RasterDomain, bctype: np.ndarray) -> io.BytesIO:
    saved = domain.save_state()
    saved.seek(0)
    npz = np.load(saved, allow_pickle=False)
    arrays = {key: npz[key] for key in npz.files}
    arrays["bctype"] = bctype
    buffer = io.BytesIO()
    np.savez(buffer, allow_pickle=False, **arrays)
    buffer.seek(0)
    return buffer


def test_load_state_accepts_legacy_float_bctype_and_restores_uint8():
    source = make_domain(np.float32)
    source.update_array("bctype", np.arange(9, dtype=np.uint8).reshape(3, 3) % 5)
    saved = source.save_state()
    saved.seek(0)
    npz = np.load(saved, allow_pickle=False)
    legacy_bctype = npz["bctype"].astype(np.float64)

    restored = make_domain(np.float32)
    restored.load_state(make_archive(source, legacy_bctype))

    np.testing.assert_array_equal(restored.get_padded("bctype"), npz["bctype"])
    assert restored.get_array("bctype").dtype == np.dtype(np.uint8)


def test_invalid_legacy_bctype_does_not_partially_restore_state():
    source = make_domain(np.float32)
    source.update_array("bctype", np.zeros(source.shape, dtype=np.uint8))
    invalid = source.get_padded("bctype").astype(np.float32)
    invalid[2, 2] = 2.5

    restored = make_domain(np.float32)
    restored.update_array("water_depth", np.full(restored.shape, 7, dtype=np.float32))
    before_bctype = restored.get_padded("bctype").copy()
    before_depth = restored.get_padded("water_depth").copy()

    with pytest.raises(HotstartError, match="bctype"):
        restored.load_state(make_archive(source, invalid))

    np.testing.assert_array_equal(restored.get_padded("bctype"), before_bctype)
    np.testing.assert_array_equal(restored.get_padded("water_depth"), before_depth)
