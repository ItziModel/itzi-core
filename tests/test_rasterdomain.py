"""Tests for RasterDomain storage dtypes and boundary-type validation."""

import numpy as np
import pytest

from itzi_core.rasterdomain import RasterDomain


def make_domain(
    dtype: type[np.floating] = np.float32, mask: np.ndarray | None = None
) -> RasterDomain:
    if mask is None:
        mask = np.zeros((3, 3), dtype=bool)
    return RasterDomain(dtype=dtype, arr_mask=mask, cell_shape=(1.0, 1.0))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_boundary_type_is_the_only_integer_storage_array(dtype):
    domain = make_domain(dtype)

    assert domain.get_array("boundary_type").dtype == np.dtype(np.uint8)
    assert domain.get_padded("boundary_type").dtype == np.dtype(np.uint8)
    for key in domain.k_all - {"boundary_type"}:
        assert domain.get_array(key).dtype == np.dtype(dtype)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32, np.float64])
def test_boundary_type_accepts_exact_codes(dtype):
    domain = make_domain()
    values = np.array([[0, 1, 2], [3, 4, 0], [1, 2, 3]], dtype=dtype)

    domain.update_array("boundary_type", values)

    np.testing.assert_array_equal(domain.get_array("boundary_type"), values)
    assert domain.get_array("boundary_type").dtype == np.dtype(np.uint8)


def test_boundary_type_masks_nan_and_domain_mask_before_validation():
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    domain = make_domain(mask=mask)
    values = np.zeros((3, 3), dtype=np.float64)
    values[0, 0] = 99
    values[1, 1] = np.nan

    domain.update_array("boundary_type", values)

    assert domain.get_array("boundary_type")[0, 0] == 0
    assert domain.get_array("boundary_type")[1, 1] == 0
    assert domain.get_unmasked("boundary_type").dtype == np.dtype(np.float32)
    assert np.isnan(domain.get_unmasked("boundary_type")[0, 0])


@pytest.mark.parametrize("invalid_value", [2.5, np.inf, -1, 5])
def test_invalid_boundary_type_update_does_not_mutate_state(invalid_value):
    domain = make_domain()
    domain.update_array("boundary_type", np.ones(domain.shape, dtype=np.uint8))
    before = domain.get_array("boundary_type").copy()
    before_padded = domain.get_padded("boundary_type").copy()
    values = np.ones(domain.shape, dtype=np.float64)
    values[1, 1] = invalid_value

    with pytest.raises(ValueError, match="boundary_type"):
        domain.update_array("boundary_type", values)

    np.testing.assert_array_equal(domain.get_array("boundary_type"), before)
    np.testing.assert_array_equal(domain.get_padded("boundary_type"), before_padded)
