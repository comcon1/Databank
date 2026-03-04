#!/usr/bin/env python3
"""
Perform comparison of experiments and simulations.

The script compares according to **EXPERIMENT** field inside :ref:`the simulation README.yaml file <readmesimu>`.
In the standard protocol, it should be run *after* :ref:`fmdl_match_experiments <match_experiments_py>`.

**Usage:**

.. code-block:: console

    fmdl_evaluate_quality

No arguments are needed.
"""

import fairmd.lipids.quality as qq
from fairmd.lipids.experiment import ExperimentCollection


def _evaluate_op_qualities(simulations) -> int:
    counter = 0
    opexps = ExperimentCollection.load_from_data("OPExperiment")
    for simulation in simulations:
        evaluator = qq.OPQualityEvaluator(simulation, opexps)
        if evaluator.evaluate_one():
            evaluator.save_results()
            counter += 1
    return counter


def _evaluate_ff_qualities(simulations) -> int:
    counter = 0
    ffexps = ExperimentCollection.load_from_data("FFExperiment")
    for simulation in simulations:
        evaluator = qq.FFQualityEvaluator(simulation, ffexps)
        if evaluator.evaluate_one():
            evaluator.save_results()
            counter += 1
    return counter


def evaluate_quality():
    simulations = qq.QualSimulation.load_all_paired()

    evaluated_op_counter = _evaluate_op_qualities(simulations)
    evaluated_ff_counter = _evaluate_ff_qualities(simulations)

    print("The number of systems with evaluated order parameters:", evaluated_op_counter)
    print("The number of systems with evaluated form factors:", evaluated_ff_counter)


if __name__ == "__main__":
    evaluate_quality()
