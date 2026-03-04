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
    from fairmd.lipids.quality import QualityEvaluator

    # single value test
    op_exp = -0.22
    exp_error = 0.02
    op_sim = -0.21
    op_sim_sd = 0.017
    p1 = QualityEvaluator.prob_2_within_trustinterval(op_exp, exp_error, op_sim, op_sim_sd)
    check.almost_equal(0.5051486814438984, p1, abs=1e-6)

    # array test
    op_exp = np.array([-0.22, -0.18, -0.25])
    exp_error = np.array([0.02, 0.01, 0.03])
    op_sim = np.array([-0.21, -0.19, -0.23])
    op_sim_sd = np.array([0.017, 0.01, 0.02])

    p2 = QualityEvaluator.prob_2_within_trustinterval(op_exp, exp_error, op_sim, op_sim_sd)
    npt.assert_allclose(p2, [0.505148, 0.352416, 0.526465], atol=1e-6)

    # array test with nans
    op_exp = np.array([-0.22, -0.18, -0.25])
    exp_error = np.array([0.02, 0.01, 0.03])
    op_sim = np.array([-0.21, -0.19, -0.23])
    op_sim_sd = np.array([0.017, 0.0, 0.02])

    p2 = QualityEvaluator.prob_2_within_trustinterval(op_exp, exp_error, op_sim, op_sim_sd)
    npt.assert_allclose(p2, [0.505148, np.nan, 0.526465], atol=1e-6)


def test_calc_ff_quality_sin_exp_curve():
    from fairmd.lipids.quality import FFQualityEvaluator

    rng = np.random.default_rng()

    # Create FF-like data: |sin(x)| * exp(-x) over q-range typical for FF (0.1 to 3.0)
    q = np.linspace(0.005, 3.0, 1000)
    # Simulation data: minimum near q=0.5
    ffd_sim = np.column_stack([q, np.abs(np.sin(2 * np.pi * q)) * np.exp(-q)]).astype(float)

    # Test edge case: identical data
    quality_identical = FFQualityEvaluator.calc_ff_quality(ffd_sim, ffd_sim)
    check.almost_equal(0.0, quality_identical, abs=1e-6, msg="Identical data should yield zero quality score")

    # Test with noised
    noise_sim = ffd_sim.copy()
    noise_sim[:, 1] += 0.001 * rng.standard_normal(len(q))  # 5% noise

    quality_noisy_sim = FFQualityEvaluator.calc_ff_quality(noise_sim, ffd_sim)
    check.almost_equal(0.0, quality_noisy_sim, rel=1e-2, msg="Noisy simulation data should yield low quality score")

    # Experimental data: minimum shifted to q=0.3
    ffd_exp = np.column_stack([q, np.abs(np.sin(2 * np.pi * (q + 0.2))) * np.exp(-q)]).astype(float)

    quality = FFQualityEvaluator.calc_ff_quality(ffd_sim, ffd_exp)
    check.almost_equal(20, quality, rel=1e-2, msg="Shifted experimental data should yield +20 quality score")

    noise_exp = ffd_exp.copy()
    noise_exp[:, 1] += 0.001 * rng.standard_normal(len(q))  # 5% noise

    quality_noisy = FFQualityEvaluator.calc_ff_quality(noise_sim, noise_exp)
    check.almost_equal(
        20,
        quality_noisy,
        rel=1e-2,
        msg="Noisy simulation and experimental data should yield similar quality score to clean data",
    )
