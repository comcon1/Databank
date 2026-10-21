"""Tests for loading experiments used by experiment matching."""

import pytest

pytestmark = [pytest.mark.min, pytest.mark.sim2]


def test_experiments_are_readable_through_package_api():
    """ToyData experiments are discoverable through the public collection API."""
    from fairmd.lipids.experiment import ExperimentCollection

    ExperimentCollection.clear_instance()
    op_experiments = ExperimentCollection.load_from_data("OPExperiment")
    assert len(op_experiments) == 1
    assert op_experiments[0].exp_id == "toy_op"
    assert op_experiments[0].data["POPC"]["M_G1_M M_G1H1_M"][:2] == [-0.17, 0.02]

    ExperimentCollection.clear_instance()
    ff_experiments = ExperimentCollection.load_from_data("FFExperiment")
    assert len(ff_experiments) == 1
    assert ff_experiments[0].exp_id == "toy_ff"
    assert ff_experiments[0].data[1] == [0.05, 0.80, 0.04]
