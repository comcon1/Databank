#!/usr/bin/env python3
"""
Perform comparison of experiments and simulations.

The script compares according to **EXPERIMENT** field inside :ref:`the simulation README.yaml file <readmesimu>`.
In the standard protocol, it should be run *after* :ref:`fmdl_match_experiments <match_experiments_py>`.

**Usage:**

.. code-block:: console

    fmdl_evaluate_quality [--op-only | --ff-only] [ --idlist=414,536 ]

No arguments means that both OP and FF qualities are evaluated for everything.
"""

import argparse

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
        print("Evaluating Sim#%d:" % simulation["ID"])
        evaluator = qq.FFQualityEvaluator(simulation, ffexps)
        if evaluator.evaluate_one():
            evaluator.save_results()
            counter += 1
    return counter


def evaluate_quality_impl(id_list: list[int] | None = None, *, do_op: bool = True, do_ff: bool = True) -> None:
    simulations = qq.QualSimulation.load_all_paired(id_list)

    evaluated_op_counter = _evaluate_op_qualities(simulations) if do_op else 0
    evaluated_ff_counter = _evaluate_ff_qualities(simulations) if do_ff else 0

    print("The number of systems with evaluated order parameters:", evaluated_op_counter)
    print("The number of systems with evaluated form factors:", evaluated_ff_counter)


def evaluate_quality():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--op-only",
        action="store_true",
        help="Evaluate order parameters only (skip form factors)",
    )
    parser.add_argument(
        "--ff-only",
        action="store_true",
        help="Evaluate form factors only (skip order parameters)",
    )
    parser.add_argument(
        "--idlist",
        type=lambda s: [int(x) for x in s.split(",")],
        default=None,
        help="Comma-separated list of IDs to evaluate (e.g. 1,2,3)",
    )
    args = parser.parse_args()
    if args.op_only and args.ff_only:
        msg = "Incompatible arguments. Please choose one."
        raise RuntimeError(msg)
    evaluate_quality_impl(
        id_list=args.idlist,
        do_op=not args.ff_only,
        do_ff=not args.op_only,
    )


if __name__ == "__main__":
    evaluate_quality()
