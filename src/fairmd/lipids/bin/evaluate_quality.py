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
        for lipidname, lipid in simulation.lipids.items():
            md_lipid_ops = simulation.op_data[lipidname]

            # TODO: bb is merged into headgroup. But sn-s do not.. Cryptic rule.
            # TODO: What to do with other types of lipids?
            fragments = mollib.get_fragments(lipid.mapping_dict)
            fragment_qual_perexp = {}
            lipid_quality_perexp = {}
            lipid_quality_perexp = {}

            for expid in simulation["EXPERIMENT"]["ORDERPARAMETER"].get(lipidname, []):
                print(f"OP quality of simulation data in {simulation['path']}")
                print(
                    f".. evaluating {lipidname} lipid using experimental data from {expid}",
                )
                exp_lipid_ops = opexps.loc(expid).data[lipidname]
                # save qualities of simulation-vs-experiment into a dictionary
                lipid_quality_perexp[expid] = qq.atomic_quality(exp_lipid_ops, md_lipid_ops)

                # calculate quality for molecule fragments headgroup, sn-1, sn-2
                _frq = qq.atomic2fragment_quality(lipid_quality_perexp[expid], fragments)
                _frw = qq.get_fragments_coverage(fragments, lipid_quality_perexp[expid])
                # dot product [ qualities * weights ]
                fragment_qual_perexp[expid] = {k: _frq[k] * _frw[k] for k in fragments}

            # Experiment-merged fragment quality for the lipid
            fragment_quality_merged = qq.fragment_quality_unite_multexp(fragment_qual_perexp)
            system_quality[lipidname] = fragment_quality_merged

            # Write FQ for the lipid
            fragment_quality_file = os.path.join(wdir, lipidname + "_FragmentQuality.json")
            FGout = False
            for FG in fragment_quality_merged:
                if np.isnan(fragment_quality_merged[FG]):
                    continue
                if fragment_quality_merged[FG] > 0:
                    FGout = True
            if FGout:
                # write fragment qualities into a file for a molecule
                _round_quality_values(fragment_quality_merged)
                with open(fragment_quality_file, "w") as f:
                    json.dump(fragment_quality_merged, f)

            # write into the OrderParameters_quality.json quality data file
            outfile1 = os.path.join(wdir, lipidname + "_OrderParameters_quality.json")
            _round_quality_values(lipid_quality_perexp)
            with open(outfile1, "w") as f:
                json.dump(lipid_quality_perexp, f, cls=CompactJSONEncoder)

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
