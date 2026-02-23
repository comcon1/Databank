"""
Test quality module functions.

-------------------------------------------------------------------------------
NOTE: globally import of fairmd-lipids is **STRICTLY FORBIDDEN** because it
      breaks the substitution of global path folders
"""

from contextlib import contextmanager
import os
import shutil
import json

import numpy as np
import numpy.testing as npt
import pytest
import pytest_check as check

# run only on sim2 mocking data
pytestmark = [pytest.mark.nodata, pytest.mark.min]


def test_prob_op_within_trustinterval():
    from fairmd.lipids.quality import prob_op_within_trustinterval

    # single value test
    op_exp = -0.22
    exp_error = 0.02
    op_sim = -0.21
    op_sim_sd = 0.017
    p1 = prob_op_within_trustinterval(op_exp, exp_error, op_sim, op_sim_sd)
    check.almost_equal(0.5051486814438984, p1, abs=1e-6)

    # array test
    op_exp = np.array([-0.22, -0.18, -0.25])
    exp_error = np.array([0.02, 0.01, 0.03])
    op_sim = np.array([-0.21, -0.19, -0.23])
    op_sim_sd = np.array([0.017, 0.01, 0.02])

    p2 = prob_op_within_trustinterval(op_exp, exp_error, op_sim, op_sim_sd)
    npt.assert_allclose(p2, [0.505148, 0.352416, 0.526465], atol=1e-6)

    # array test with nans
    op_exp = np.array([-0.22, -0.18, -0.25])
    exp_error = np.array([0.02, 0.01, 0.03])
    op_sim = np.array([-0.21, -0.19, -0.23])
    op_sim_sd = np.array([0.017, 0.0, 0.02])

    p2 = prob_op_within_trustinterval(op_exp, exp_error, op_sim, op_sim_sd)
    npt.assert_allclose(p2, [0.505148, np.nan, 0.526465], atol=1e-6)
