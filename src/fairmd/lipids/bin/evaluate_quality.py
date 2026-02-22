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

import json
import os

import numpy as np

import fairmd.lipids.quality as qq
from fairmd.lipids import FMDL_SIMU_PATH
from fairmd.lipids.auxiliary import CompactJSONEncoder
from fairmd.lipids.experiment import ExperimentCollection


def _round_quality_values(obj: dict | list, ndigits: int = 4) -> dict | list:
    """Round all floating-point values in a nested dict/list structure."""
    stack = [obj]

    while stack:
        current = stack.pop()

        if isinstance(current, dict):
            for k, v in current.items():
                if isinstance(v, float):
                    current[k] = round(v, ndigits)
                elif isinstance(v, (dict, list)):
                    stack.append(v)

        elif isinstance(current, list):
            for i, v in enumerate(current):
                if isinstance(v, float):
                    current[i] = round(v, ndigits)
                elif isinstance(v, (dict, list)):
                    stack.append(v)

    return obj


def _evaluate_op_qualities(simulations) -> int:
    counter = 0
    opexps = ExperimentCollection.load_from_data("OPExperiment")
    for simulation in simulations:
        wdir = os.path.join(FMDL_SIMU_PATH, simulation["path"])

        # Order Parameters
        system_quality = {}
        for lipid1 in simulation.lipids:
            print(f"\nEvaluating order parameter quality of simulation data in {simulation['path']}")

            OP_data_lipid = simulation.op_data[lipid1]

            # go through file paths in simulation.readme['EXPERIMENT']
            fragment_qual_dict = {}
            data_dict = {}

            for expid in simulation["EXPERIMENT"]["ORDERPARAMETER"][lipid1]:
                print(
                    f"Evaluating {lipid1} lipid using experimental data from {expid}",
                )
                OP_qual_data = {}
                exp_op_data = opexps.loc(expid)
                exp_error = 0.02

                for key, op_array_ in OP_data_lipid.items():
                    OP_array = op_array_.copy()
                    try:
                        OP_exp = exp_op_data[key][0][0]
                    except KeyError:
                        continue
                    else:
                        if not np.isnan(OP_exp):
                            OP_sim = OP_array[0]
                            op_sim_STEM = OP_array[2]
                            # changing to use shitness(TM) scale.
                            # This code needs to be cleaned
                            op_quality = qq.prob_S_in_g(OP_exp, exp_error, OP_sim, op_sim_STEM)
                            OP_array.append(OP_exp)
                            OP_array.append(exp_error)  # hardcoded!!! 0.02 for all exps
                            OP_array.append(op_quality)
                    OP_qual_data[key] = OP_array

                # save qualities of simulation-vs-experiment into a dictionary
                data_dict[expid] = OP_qual_data

                # calculate quality for molecule fragments headgroup, sn-1, sn-2
                fragments = qq.get_fragments(simulation.content[lipid1].mapping_dict)
                fragment_qual_dict[expid] = qq.fragmentQuality(fragments, exp_op_data, OP_data_lipid)

            try:
                fragment_quality_output = qq.fragmentQualityAvg(lipid1, fragment_qual_dict, fragments)
            except Exception:
                print("no fragment quality")
                fragment_quality_output = {}

            try:
                system_quality[lipid1] = fragment_quality_output
            except Exception:
                print("no system quality")
                system_quality[lipid1] = {}

            fragment_quality_file = os.path.join(wdir, lipid1 + "_FragmentQuality.json")

            FGout = False
            for FG in fragment_quality_output:
                if np.isnan(fragment_quality_output[FG]):
                    continue
                if fragment_quality_output[FG] > 0:
                    FGout = True
            if FGout:
                # write fragment qualities into a file for a molecule
                _round_quality_values(fragment_quality_output)

                with open(fragment_quality_file, "w") as f:
                    json.dump(fragment_quality_output, f)

            # write into the OrderParameters_quality.json quality data file
            outfile1 = os.path.join(wdir, lipid1 + "_OrderParameters_quality.json")
            try:
                _round_quality_values(data_dict)
                with open(outfile1, "w") as f:
                    json.dump(data_dict, f, cls=CompactJSONEncoder)
            except Exception:
                pass

        system_qual_output = qq.systemQuality(system_quality, simulation)
        # make system quality file

        outfile2 = os.path.join(wdir, "SYSTEM_quality.json")
        SQout = any(v > 0 for v in system_qual_output.values())

        if SQout:
            _round_quality_values(system_qual_output)
            with open(outfile2, "w") as f:
                json.dump(system_qual_output, f)
            print("Order parameter quality evaluated for " + simulation.idx_path)
            counter += 1
            print()
    return counter


def _evaluate_ff_qualities(simulations) -> int:
    counter = 0
    ffexps = ExperimentCollection.load_from_data("FFExperiment")
    for simulation in simulations:
        wdir = os.path.join(FMDL_SIMU_PATH, simulation["path"])
        if "FORMFACTOR" not in simulation.get("EXPERIMENT", {}):
            continue
        if simulation.ff_data is None:
            continue
        results_ff = {}
        for expid in simulation["EXPERIMENT"]["FORMFACTOR"]:
            exp_ff_data = np.array(ffexps.loc(expid).data)
            results_ff[expid] = qq.get_ffq_scaling(simulation.ff_data, exp_ff_data)

        # TODO: handle multiple FF experiments better
        # currently, just pick the best one
        best_ep = None
        for exp_path, ffq_scf in results_ff.items():
            if ffq_scf is None:
                continue
            if best_ep is None or ffq_scf[0] < results_ff[best_ep][0]:
                best_ep = exp_path
        if best_ep is not None:
            print(f"Form factor quality for experiment data from {best_ep}:")
            print("Distance =", results_ff[best_ep][0], "; scaling factor =", results_ff[best_ep][1])

            print("Form factor quality evaluated for ", wdir)
            outfile3 = os.path.join(wdir, "FormFactorQuality.json")

            ff_quality = list(results_ff[best_ep])
            _round_quality_values(ff_quality)

            with open(outfile3, "w") as f:
                json.dump(ff_quality, f)

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
