#!/usr/bin/env python3
r"""
Match simulations with experiments in the databank.

Script goes through all simulations and experiments in the databank and finds
pairs of simulations and experiments that match in composition, temperature and
other conditions. The found pairs are written into the simulation :ref:`README.yaml <readmesimu>`
files and into a log file.

**Usage:**

.. code-block:: console

    fmdl_match_experiments

No arguments are needed.
TODO: check if EXPERIMENT section changed and trigger the action!
"""

import logging
import os
from typing import IO

import numpy as np
import yaml
from tqdm import tqdm

from fairmd.lipids import FMDL_SIMU_PATH
from fairmd.lipids.core import System, initialize_databank
from fairmd.lipids.experiment import Experiment, ExperimentCollection

logger = logging.getLogger("__name__")

# TODO: REMOVE IT COMPLETELY!!!
ions_list = ["POT", "SOD", "CLA", "CAL"]  # should contain names of all ions

LIP_CONC_REL_THRESHOLD = 0.15  # relative acceptable error for determination
# of the hydration in ssNMR


# TODO: derive from Simulation (if not to remove at all!)
class SearchSystem:
    system: dict
    idx_path: str

    def __init__(self, readme):
        self.system: System = readme
        self.idx_path = readme["path"]

    def total_lipid_conc(self):
        c_water = 55.5
        n_water = self.system["COMPOSITION"]["SOL"]["COUNT"]
        try:
            if (n_water / self.system.n_lipids) > 25:
                tot_lipid_c = "full hydration"
            else:
                tot_lipid_c = (self.system.n_lipids * c_water) / n_water
        except ZeroDivisionError:
            logger.warning("Division by zero when determining lipid concentration!")
            print(self.system)
        return tot_lipid_c


##################


def load_simulations() -> list[SearchSystem]:
    """Generate the list of Simulation objects. Go through all README.yaml files."""
    systems = initialize_databank()
    simulations: list[SearchSystem] = []

    for system in systems:
        # conditions of exclusions
        try:
            if system["WARNINGS"]["NOWATER"]:
                continue
        except (KeyError, TypeError):
            pass

        simulations.append(SearchSystem(system))

    return simulations


def load_experiments(exp_type: str, all_experiments: ExperimentCollection) -> list[Experiment]:
    """Filter experiments from the collection by experiment type."""
    print(f"Filtering for {exp_type} experiments...")
    return [exp for exp in all_experiments if exp.exptype == exp_type]


def find_pairs_and_change_sims(experiments: list[Experiment], simulations: list[SearchSystem]):
    pairs = []
    for simulation in tqdm(simulations, desc="Simulation"):
        if simulation.system["ID"] == 755:
            continue
        sim_lipids_set = simulation.system.lipids.keys()
        sim_lipids_mf = simulation.system.membrane_composition(basis="molar")
        sim_ions_set = simulation.system.solubles.keys()
        sim_ions_mf = simulation.system.solution_composition(basis="molar")
        sim_tlc = simulation.total_lipid_conc()
        if sim_tlc == "full hydration":
            sim_tlc = 55.5 / 40
        t_sim = simulation.system["TEMPERATURE"]

        for experiment in experiments:
            # compare molar fractions
            # TODO: BAD! use relative threshold instead!
            if set(sim_lipids_set) != set(experiment.lipids.keys()):
                continue
            exp_mf = experiment.membrane_composition(basis="molar")
            abs_tolerance_molfraction = 0.03
            membrane_composition_ok = True
            for key in sim_lipids_mf:
                if np.abs(exp_mf[key] - sim_lipids_mf[key]) > abs_tolerance_molfraction:
                    membrane_composition_ok = False
                    break
            if not membrane_composition_ok:
                continue

            # compare ion concentrations
            # BAD! use relative threshold instead!
            # BAD! use logarithmic scale instead!
            abs_tolerance_solutionconc = 0.05
            if set(sim_ions_set) != set(experiment.solubles.keys()):
                continue
            exp_ions_mf = experiment.solution_composition(basis="molar")
            solution_composition_ok = True
            for key in sim_ions_set:
                if (exp_ions_mf[key] - sim_ions_mf[key]) < abs_tolerance_solutionconc:
                    solution_composition_ok = False
                    break
            if not solution_composition_ok:
                continue

            exp_tlc = experiment.readme["TOTAL_LIPID_CONCENTRATION"]
            if exp_tlc == "full hydration":
                exp_tlc = 55.5 / 40

            if (
                not (exp_tlc < 55.5 / 25 and sim_tlc < 55.5 / 25)  # both not full hydration
                and np.abs(exp_tlc / sim_tlc - 1) > LIP_CONC_REL_THRESHOLD
            ):
                continue

            t_exp = experiment.readme["TEMPERATURE"]
            abs_tolerance_t = 2.5
            if np.abs(t_exp - t_sim) > abs_tolerance_t:
                continue

            # !we found the match!
            pairs.append([simulation, experiment])

            # Add path to experiment into simulation README.yaml
            # many experiment entries can match to same simulation
            if experiment.exptype == "OrderParameters":
                for lipid in experiment.data:
                    if lipid not in simulation.system["EXPERIMENT"]["ORDERPARAMETER"]:
                        simulation.system["EXPERIMENT"]["ORDERPARAMETER"][lipid] = []
                    simulation.system["EXPERIMENT"]["ORDERPARAMETER"][lipid].append(experiment.exp_id)
            elif experiment.exptype == "FormFactors":
                simulation.system["EXPERIMENT"]["FORMFACTOR"].append(experiment.exp_id)

        # sorting experiment lists to keep experimental order strict
        cur_exp = simulation.system["EXPERIMENT"]
        cur_exp["FORMFACTOR"].sort()
        for _lipid in cur_exp["ORDERPARAMETER"]:
            cur_exp["ORDERPARAMETER"][_lipid].sort()

    return pairs


def log_pairs(pairs, fd: IO[str]) -> None:
    """
    Write found correspondences into log file.

    pairs: [(Simulation, Experiment), ...]
    fd: file descriptor for writting into
    """
    for p in pairs:
        sim: SearchSystem = p[0]
        exp: Experiment = p[1]

        sysn = sim.system["SYSTEM"]
        simp = sim.idx_path

        expp = exp.path
        expd = exp.readme.get("ARTICLE_DOI", "[no article DOI]")

        fd.write(f"""
--------------------------------------------------------------------------------
Simulation:
 - {sysn}
 - {simp}
Experiment:
 - {expd}
 - {expp}""")
        # end for
    fd.write("""
--------------------------------------------------------------------------------
    \n""")


def match_experiments() -> None:
    """Do main program work. Not for exporting."""
    simulations = load_simulations()

    # clear all EXPERIMENT sections in all simulations
    for simulation in simulations:
        simulation.system["EXPERIMENT"] = {}
        simulation.system["EXPERIMENT"]["ORDERPARAMETER"] = {}
        simulation.system["EXPERIMENT"]["FORMFACTOR"] = []
        for lipid in simulation.system.lipids:
            simulation.system["EXPERIMENT"]["ORDERPARAMETER"][lipid] = []

    # Pair each simulation with an experiment with the closest matching temperature
    # and composition
    with open("search-databank-pairs.log", "w") as logf:
        print("Scanning simulation-experiment pairs among order parameter experiments.")
        exps = ExperimentCollection.load_from_data("OPExperiment")
        print(f"{len(exps)} OP experiments loaded.")
        pairs_op = find_pairs_and_change_sims(exps, simulations)
        logf.write("=== OP PAIRS ===\n")
        log_pairs(pairs_op, logf)

        exps = ExperimentCollection.load_from_data("FFExperiment")
        print(f"{len(exps)} FF experiments loaded.")
        print("Scanning simulation-experiment pairs among form factor experiments.")
        pairs_ff = find_pairs_and_change_sims(exps, simulations)
        logf.write("=== FF PAIRS ===\n")
        log_pairs(pairs_ff, logf)

    # save changed simulations
    for simulation in tqdm(simulations, "Saving READMEs"):
        outfile_dict = os.path.join(FMDL_SIMU_PATH, simulation.idx_path, "README.yaml")
        with open(outfile_dict, "w") as f:
            if "path" in simulation.system:
                del simulation.system["path"]
            yaml.dump(simulation.system.readme, f, sort_keys=False, allow_unicode=True)

    print("Found order parameter data for " + str(len(pairs_op)) + " pairs")
    print("Found form factor data for " + str(len(pairs_ff)) + " pairs")


if __name__ == "__main__":
    match_experiments()
