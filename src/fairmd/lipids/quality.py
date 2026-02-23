"""
Module `quality` accumulate functions required for major QualityEvaluation script.
TODO: add tests
TODO: remove code duplication and commented code
"""

import decimal as dc
import re
import warnings

import numpy as np
import scipy.stats
from scipy.special import log1p, expm1

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
def prob_S_in_g(OP_exp: float, exp_error: float, OP_sim: float, op_sim_sd: float) -> float:
    """Compute the quality value from experimental and simulation OP data.

    Args:
        OP_exp (float): Experimental OP value
        exp_error (float): Experimental error
        OP_sim (float): Simulated OP value
        op_sim_sd (float): Standard deviation from simulation

    Returns
    -------
        float: single-OP quality value or NaN
    """
    # normal distribution N(s, OP_sim, op_sim_sd)
    a = OP_exp - exp_error
    b = OP_exp + exp_error

    a_rel = (OP_sim - a) / op_sim_sd
    b_rel = (OP_sim - b) / op_sim_sd
    p_s = scipy.stats.t.sf(b_rel, df=1, loc=0, scale=1) - scipy.stats.t.sf(a_rel, df=1, loc=0, scale=1)

    if np.isnan(p_s):
        return p_s

    # this is an attempt to deal with precision, max set manually to 70
    dc.getcontext().prec = 70
    _ = -dc.Decimal(p_s).log10()

    return float(p_s)


def prob_op_within_trustinterval(op_exp: float, exp_error: float, op_sim: float, op_sim_sd: float) -> float:
    a = op_exp - exp_error
    b = op_exp + exp_error

    a_rel = (op_sim - a) / op_sim_sd
    b_rel = (op_sim - b) / op_sim_sd

    a_logprob = scipy.stats.t.logsf(a_rel, df=1, loc=0, scale=1)
    b_logprob = scipy.stats.t.logsf(b_rel, df=1, loc=0, scale=1)

    # log(b - a) = log(exp(logb) - exp(loga)) = logb + log(1 - exp(loga - logb))
    log_ab = b_logprob + log1p(-expm1(a_logprob - b_logprob))

    if np.isnan(log_ab):
        return np.nan

    return -log_ab / np.log(10)


# quality of molecule fragments
def get_fragments(mapping_dict: dict):
    fragments = {}

    for key_m, value in mapping_dict.items():
        try:
            key_f = value["FRAGMENT"]
        except KeyError:
            key_f = "n/d"
        fragments.setdefault(key_f, []).append(key_m)

    # merge glycerol backbone fragment into headgroup fragment
    if "glycerol backbone" in fragments and "headgroup" in fragments:
        fragments["headgroup"] += fragments["glycerol backbone"]
        fragments.pop("glycerol backbone")

    return fragments


def filterCH(fragment_key, fragments):
    re_CH = re.compile(r"M_([GC0-9]*[A-Z0-9]*C[0-9]*H[0-9]*)*([GC0-9]*H[0-9]*)*_M")
    filtered = list(filter(re_CH.match, fragments[fragment_key]))
    return filtered


def checkForCH(fragment_key, fragments):
    filtered = filterCH(fragment_key, fragments)
    return bool(filtered)


def evaluated_percentage(fragments, exp_op_data):
    # C-H bonds only???

    frag_percentage = dict.fromkeys(fragments, 0)

    for fragment_key in fragments.keys():  # go through fragments
        count_value = 0
        fragment_size = 0
        for key, value in exp_op_data.items():
            if key.split(" ")[0] in fragments[fragment_key]:  # check if atom belongs to the fragment
                fragment_size += 1
                if not np.isnan(value[0][0]):
                    count_value += 1
        if fragment_size != 0:
            frag_percentage[fragment_key] = count_value / fragment_size
        else:
            frag_percentage[fragment_key] = 0

    print("experiment data availability percentage")
    print(frag_percentage)

    return frag_percentage


def fragmentQuality(fragments, exp_op_data, sim_op_data):
    # depends on the experiment file what fragments are in this dictionary
    p_F = evaluated_percentage(fragments, exp_op_data)
    exp_error = 0.02

    # empty dictionary with fragment names as keys
    fragment_quality = dict.fromkeys(fragments.keys())

    for fragment_key in fragments.keys():
        E_sum = 0
        AV_sum = 0
        try:
            _ = p_F[fragment_key]
        except KeyError:
            fragment_quality[fragment_key] = np.nan
            continue
        else:
            if p_F[fragment_key] != 0:
                for key_exp, value_exp in exp_op_data.items():
                    if key_exp.split()[0] in fragments[fragment_key] and not np.isnan(value_exp[0][0]):
                        OP_exp = value_exp[0][0]
                        try:
                            OP_sim = sim_op_data[key_exp][0]
                        except (KeyError, TypeError):
                            continue
                        else:
                            op_sim_STEM = sim_op_data[key_exp][2]

                            # change here if you want to use shitness(TM) scale for
                            # fragments. Warning big numbers will dominate
                            # TODO: remove commented
                            # if OP_exp != float("NaN"):
                            QE = prob_S_in_g(OP_exp, exp_error, OP_sim, op_sim_STEM)
                            # print(OP_exp, OP_sim ,QE)
                            # print(QE, 10**(-QE))

                            # print('prob_S')
                            # print(QE)
                            #  if QE >0:
                            #   if QE == float("NaN"):
                            #    E_sum = E_sum
                            #    if QE == float("inf"): #'Infinity' or QE == 'inf':
                            #         E_sum += 300
                            #         AV_sum += 1
                            #    else:
                            #        print(QE)
                            #        E_sum += prob_S_in_g(OP_exp, exp_error, OP_sim,
                            #                               op_sim_STEM)
                            #        AV_sum += 1
                            E_sum += QE
                            AV_sum += 1
                if AV_sum > 0:
                    E_F = (E_sum / AV_sum) * p_F[fragment_key]
                    fragment_quality[fragment_key] = E_F
                else:
                    fragment_quality[fragment_key] = np.nan
            else:
                fragment_quality[fragment_key] = np.nan

    print("fragment quality ", fragment_quality)
    return fragment_quality


def fragmentQualityAvg(
    lipid,
    fragment_qual_dict,
    fragments,
):  # handles one lipid at a time
    sums_dict = {}

    for doi in fragment_qual_dict.keys():
        for key_fragment in fragment_qual_dict[doi].keys():
            f_value = fragment_qual_dict[doi][key_fragment]
            sums_dict.setdefault(key_fragment, []).append(f_value)

    avg_total_quality = {}

    for key_fragment in sums_dict:
        # remove nan values
        to_be_summed = [x for x in sums_dict[key_fragment] if not np.isnan(x)]
        if to_be_summed:
            avg_value = sum(to_be_summed) / len(to_be_summed)
        else:
            avg_value = np.nan
        avg_total_quality.setdefault(key_fragment, avg_value)

    # if average fragment quality exists for all fragments that contain CH bonds then
    # calculate total quality over all fragment quality averages
    if [
        x
        for x in avg_total_quality
        if (checkForCH(x, fragments) and not np.isnan(avg_total_quality[x])) or (not checkForCH(x, fragments))
    ]:
        list_values = [x for x in avg_total_quality.values() if not np.isnan(x)]
        avg_total_quality["total"] = sum(list_values) / len(list_values)
    else:
        avg_total_quality["total"] = np.nan

    print("fragment avg")
    print(avg_total_quality)

    return avg_total_quality


# fragments is different for each lipid ---> need to make individual dictionaries
def systemQuality(system_fragment_qualities, simulation):
    system_dict = {}
    lipid_dict = {}
    w_nan = []

    for lipid in system_fragment_qualities:
        # copy keys to new dictionary
        lipid_dict = dict.fromkeys(system_fragment_qualities[lipid].keys(), 0)

        w = simulation.molar_fraction(lipid)

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
