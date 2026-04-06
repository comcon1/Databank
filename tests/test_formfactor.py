"""
Test module test_formfactor.py

Contains tests of functions designed for processing of form-factor curve.

Test data is stored in `./ToyData/Simulations.2`

-------------------------------------------------------------------------------
NOTE: globally import of fairmd-lipids is **STRICTLY FORBIDDEN** because it
      breaks the substitution of global path folders
"""

import numpy as np

import pytest
import pytest_check as check

pytestmark = [pytest.mark.nodata, pytest.mark.min]


def test_get_mins_from_ffdata():
    from fairmd.lipids.analib.formfactor import get_mins_from_ffdata

    synt_data = np.zeros((1000, 3))
    synt_data[:, 0] = np.linspace(0, 1, 1000)
    synt_data[:, 1] = np.abs(np.sin(synt_data[:, 0] * 8))
    # sin(x*8) = 0
    # x1 = pi/8
    # x2 = pi/4
    p = get_mins_from_ffdata(synt_data)
    check.almost_equal(p[0], np.pi / 8, abs=1e-3)
    check.almost_equal(p[1], np.pi / 4, abs=1e-3)


def test_estimate_error_of_min():
    from fairmd.lipids.analib.formfactor import estimate_error_of_min

    # synthetic data
    synt_data = np.zeros((1000, 3))
    synt_data[:, 0] = np.linspace(0, 1, 1000)
    synt_data[:, 1] = np.abs(np.sin(synt_data[:, 0] * 8))
    # Case 0: no error
    # no error is handled as constant error 0.1
    err_noerr = estimate_error_of_min(synt_data[:, :2])
    err_err01 = estimate_error_of_min(synt_data)
    check.almost_equal(
        err_noerr,
        err_err01,
        abs=1e-3,
        msg="Error of minimum with no error is not estimated correctly",
    )
    # Case 1: constant error
    synt_data[:, 2] = 0.1
    # sin(8x) = 0
    # 8x = pi*n => x = pi*n/8
    #
    # Error must be estimated as following: we down the curve by the value of error and find intersections
    # with x axis. The error is the mean distance between the minimum and these intersections. In this
    # case, the intersections are defined by the equation:
    #
    # |sin(8x)|-0.1 = 0 =>> |sin(8x)| = 0.1 =>> sin(8x) = 0.1 or sin(8x) = -0.1, i.e.,
    #
    # x = +-arcsin(0.1)/8 + pi*n/8; so error should be arcsin(0.1)/8
    #
    err_comp = estimate_error_of_min(synt_data)
    check.almost_equal(
        err_comp,
        np.arcsin(0.1) / 8,
        abs=1e-3,
        msg="Error of minimum with constant error is not estimated correctly",
    )
