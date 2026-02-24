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

import fairmd.lipids.analib.formfactor as ff
import fairmd.lipids.quality as qq
from fairmd.lipids import FMDL_SIMU_PATH
from fairmd.lipids.auxiliary import CompactJSONEncoder, mollib
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

        system_quality = {}
        for lipid1 in simulation.lipids:
            md_lipid_ops = simulation.op_data[lipid1]

            fragment_qual_dict = {}
            data_dict = {}

            for expid in simulation["EXPERIMENT"]["ORDERPARAMETER"].get(lipid1, []):
                print(f"OP quality of simulation data in {simulation['path']}")
                print(
                    f".. evaluating {lipid1} lipid using experimental data from {expid}",
                )
                OP_qual_data = {}
                exp_lipid_ops = opexps.loc(expid).data[lipid1]
                exp_error = 0.02  # TODO: hardcoded error value, should be taken from experiment data when available

                for key, op_array_ in md_lipid_ops.items():
                    OP_array = op_array_.copy()
                    if key not in exp_lipid_ops:
                        continue
                    OP_exp_val = exp_lipid_ops[key][0]
                    if not np.isnan(OP_exp_val):
                        op_quality = qq.prob_op_within_trustinterval(
                            op_exp=OP_exp_val,
                            exp_error=exp_error,
                            op_sim=OP_array[0],
                            op_sim_sd=OP_array[2],
                        )
                        OP_array += [OP_exp_val, exp_error, op_quality]
                    OP_qual_data[key] = OP_array

                # save qualities of simulation-vs-experiment into a dictionary
                data_dict[expid] = OP_qual_data

                # calculate quality for molecule fragments headgroup, sn-1, sn-2
                # TODO: bb is merged into headgroup. But sn-s do not.. Cryptic rule.
                # TODO: What to do with other types of lipids?
                fragments = mollib.get_fragments(simulation.content[lipid1].mapping_dict)
                fragment_qual_dict[expid] = qq.fragment_quality(fragments, exp_lipid_ops, md_lipid_ops)

            try:
                fragment_quality_output = qq.fragment_quality_unite_multexp(lipid1, fragment_qual_dict, fragments)
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
            print("Order parameter quality evaluated for " + simulation["path"])
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
            results_ff[expid] = [
                qq.calc_ff_quality(simulation.ff_data, exp_ff_data),
                ff.calc_ff_scaling_distance(exp_ff_data, simulation.ff_data)[0],
            ]

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
