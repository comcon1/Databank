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
    from fairmd.lipids.quality import prob_op_within_trustinterval, prob_S_in_g
    op_exp = -0.22
    exp_error = 0.02
    op_sim = -0.21
    op_sim_sd = 0.017
    p1 = prob_op_within_trustinterval(op_exp, exp_error, op_sim, op_sim_sd)
    p2 = prob_S_in_g(op_exp, exp_error, op_sim, op_sim_sd)
    check.almost_equal(p1, p2)
