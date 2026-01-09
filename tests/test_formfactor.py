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
