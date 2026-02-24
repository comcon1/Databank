"""
Module `quality` accumulate functions required for major QualityEvaluation script.
TODO: add tests
TODO: remove code duplication and commented code
"""

import warnings

import numpy as np
import numpy.typing as npt
import scipy.stats

from fairmd.lipids.analib.formfactor import get_mins_from_ffdata
from fairmd.lipids.api import get_FF, get_OP
from fairmd.lipids.core import System, initialize_databank


class QualSimulation(System):
    def __init__(self, s: System):
        super().__init__(s)
        self.op_data = get_OP(s)
        try:
            self.ff_data = get_FF(s)
        except FileNotFoundError:
            msg = f"System #{s['ID']} doesn't have computed FF data."
            warnings.warn(msg, stacklevel=2)
            self.ff_data = None

    @staticmethod
    def load_all_paired() -> list["QualSimulation"]:
        """Load simulations with experimental pairings."""
        systems = initialize_databank()

        simulations = []
        for system in systems:
            experiments = system.get("EXPERIMENT", {})
            if any(experiments.values()):  # if experiments is not empty
                s = QualSimulation(system)
                simulations.append(s)
        print(f"Loaded {len(simulations)} that has some experimental pairings.")

        return simulations


# Order parameters
def prob_op_within_trustinterval(
    op_exp: npt.ArrayLike,
    exp_error: npt.ArrayLike,
    op_sim: npt.ArrayLike,
    op_sim_sd: npt.ArrayLike,
) -> npt.ArrayLike:
    """
    Compute the quality value from experimental and simulation OP data.

    Probability is computed using Eq. (3) [10.1038/s41467-024-45189-z].

    NOTE: Computing the probability taking into account small values using
    scipy.special.log1p is not required if sd is above 1e-5, which is
    currently the case for all OP data.

    Args:
        OP_exp: Experimental OP value (all float/arrays)
        exp_error: Experimental error
        OP_sim: Simulated OP value
        op_sim_sd: Standard deviation from simulation

    Returns
    -------
        float/ndarray: probability value(s) or nans
    """
    # normal distribution N(s, OP_sim, op_sim_sd)
    a = op_exp - exp_error
    b = op_exp + exp_error

    a_rel = (op_sim - a) / op_sim_sd
    b_rel = (op_sim - b) / op_sim_sd

    p_b = scipy.stats.t.sf(b_rel, df=1, loc=0, scale=1)
    p_a = scipy.stats.t.sf(a_rel, df=1, loc=0, scale=1)

    return p_b - p_a


def get_fragments_coverage(fragments: dict, q_data: dict) -> dict:
    """
    Calculate the coverage in data of each fragment.

    The coverage is implemented as the percent of non-nans.

    :param fragments: Dictionary of type {fragment: lists of unames}.
    :param q_data: Dictionary of type {op_uname: quality value}.

    :return: Dictionary of type {fragment: percentage of evaluated OPs}
    """
    frag_percentage = dict.fromkeys(fragments, 0)

    for frg_name, frg_atoms in fragments.items():
        frg_atoms_set = set(frg_atoms)
        nan_mask = [np.isnan(v) for k, v in q_data.items() if k.split(" ")[0] in frg_atoms_set]
        frag_percentage[frg_name] = 1 - sum(nan_mask) / len(nan_mask) if len(nan_mask) > 0 else 0

    return frag_percentage


def atomic_quality(exp_op_data: dict, sim_op_data: dict):
    """
    Calculate quality for a molecule (times their weights in exp data).

    :param exp_op_data: dictionary of type {op_uname: [op_value]}.
    :param sim_op_data: dictionary of type {op_uname: [op_value, op_sigma, op_sd]}.

    :return: dictionary of type {"nC nH": quality value}.
    """
    exp_error = 0.02  # TODO: hardcoded error value, should be taken from experiment data when available

    # union of keys in exp_op_data and sim_op_data
    all_keys = sorted(set(exp_op_data.keys()) | set(sim_op_data.keys()))
    res_dict = {}

    for key in all_keys:
        if key not in exp_op_data or key not in sim_op_data:
            res_dict[key] = np.nan
            continue
        q = prob_op_within_trustinterval(
            op_exp=exp_op_data[key][0],
            exp_error=exp_error,
            op_sim=sim_op_data[key][0],
            op_sim_sd=sim_op_data[key][2],
        )
        res_dict[key] = q

    return res_dict


def atomic2fragment_quality(atomic_qual_dict: dict, fragments: dict):
    fqdict = dict.fromkeys(fragments.keys())
    for frg_name, frg_atoms in fragments.items():
        q_list = [v for k, v in atomic_qual_dict.items() if k.split(" ")[0] in frg_atoms]
        fqdict[frg_name] = np.nanmean(q_list) if len(q_list) > 0 else np.nan
    return fqdict


def fragment_quality_unite_multexp(
    lipid,
    fragment_qual_dict: dict,
    fragments: dict,
):
    """
    Condition fragment qualities.

    The second-layer function.

    :param lipid: ...
    :param fragment_qual_perexp: dictionary of type {expid: {fragment: quality value}}.
    :param fragments: dictionary of type {fragment:lists of unames}.

    :return: dictionary of type {fragment: average quality value}.
    """
    sums_dict = {}
    for exp_frag_qual in fragment_qual_dict.values():
        for frag_name, frag_qual in exp_frag_qual.items():
            sums_dict.setdefault(frag_name, []).append(frag_qual)

    avg_total_quality = {frag_name: np.nanmean(frag_vals) for frag_name, frag_vals in sums_dict.items()}
    avg_total_quality["total"] = np.nanmean(list(avg_total_quality.values()))

    return avg_total_quality


# fragments is different for each lipid ---> need to make individual dictionaries
def systemQuality(system_fragment_qualities, simulation: QualSimulation):
    system_dict = {}
    lipid_dict = {}
    w_nan = []

    for lipid in system_fragment_qualities:
        # copy keys to new dictionary
        lipid_dict = dict.fromkeys(system_fragment_qualities[lipid].keys(), 0)

        w = simulation.membrane_composition(basis="molar")[lipid]

        for key, value in system_fragment_qualities[lipid].items():
            if not np.isnan(value):
                lipid_dict[key] += w * value
            else:
                # save 1 - w of a lipid into a list if the fragment quality is nan
                w_nan.append(1 - w)

        system_dict[lipid] = lipid_dict

    system_quality = {}

    headgroup = 0
    tails = 0
    total = 0

    for lipid_key in system_dict:
        for key, value in system_dict[lipid_key].items():
            if key == "total":
                total += value
            elif key == "headgroup":
                headgroup += value
            elif key == "sn-1" or key == "sn-2":
                tails += value / 2
            else:
                tails += value

    if np.prod(w_nan) > 0:
        # multiply all elements of w_nan and divide the sum by the product
        system_quality["headgroup"] = headgroup * np.prod(w_nan)
        system_quality["tails"] = tails * np.prod(w_nan)
        system_quality["total"] = total * np.prod(w_nan)
    else:
        system_quality["headgroup"] = headgroup
        system_quality["tails"] = tails
        system_quality["total"] = total

    print("system_quality")
    print(system_quality)

    return system_quality


def calc_ff_quality(ffd_sim: np.ndarray, ffd_exp: np.ndarray) -> float:
    """
    Calculate form factor quality.

    Quality calculation is performed as defined by Kučerka et al. 2010, doi:10.1007/s00232-010-9254-5

    :param ffd_sim: Simulation FF data (float 2D list)
    :param ffd_exp: Experiment FF data (float 2D list)

    :return: Quality value.
    """
    sim_min = get_mins_from_ffdata(ffd_sim)
    exp_min = get_mins_from_ffdata(ffd_exp)

    return np.abs(sim_min[0] - exp_min[0]) * 100
