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


def test_match_experiments_matching_criteria():
    """Matching criteria accept compatible simulation and experiment values."""
    from fairmd.lipids.bin.match_experiments import (
        _match_membrane_composition,
        _match_solution_composition,
        _match_temperature,
    )

    assert _match_membrane_composition({"POPC": 1.0}, {"POPC": 1.0})
    assert _match_solution_composition({"SOD": 0.10}, {"SOD": 0.12})
    assert _match_temperature(303.0, 303.15)
    assert not _match_temperature(303.0, 310.0)
