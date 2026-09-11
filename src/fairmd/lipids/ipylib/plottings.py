"""
@DRAFT
Network communication. Downloading files. Checking links etc.
"""

import re
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np

from fairmd.lipids.auxiliary.opconvertor import build_nice_OPdict
from fairmd.lipids.analib.formfactor import calc_ff_scaling_distance
from fairmd.lipids.api import get_FF, get_OP, get_quality
from fairmd.lipids.experiment import ExperimentCollection
from fairmd.lipids.molecules import Lipid


def plotFormFactor(  # noqa: N802
    exp_form_factor,
    k,
    legend,
    plot_color,
):
    """:meta private:"""
    x_vals = []
    y_vals = []
    for i in exp_form_factor:
        x_vals.append(i[0])
        y_vals.append(k * i[1])
    plt.plot(x_vals, y_vals, label=legend, color=plot_color, linewidth=4.0)
    plt.xlabel(r"$q_{z} [Å^{-1}]$", size=20)
    plt.ylabel(r"$|F(q_{z})|$", size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xlim([0, 0.69])
    plt.ylim([-10, 250])
    plt.legend(loc="upper right")
    plt.savefig("FormFactor.pdf")


def plotOrderParameters(OPsim, OPexp, lipid_name):  # noqa
    """:meta private:"""

    def _build_registry_rows(op_data, lipid_obj):
        formatted = build_nice_OPdict(op_data, lipid_obj)
        for fragment in formatted:
            for row in formatted[fragment]:
                row["ERR"] = 0.0 if row["STD"] is None else float(row["STD"])
        return formatted

    def _group_by_carbon(registry_rows, fragments):
        grouped = {}
        for fragment in fragments:
            for row in registry_rows.get(fragment, []):
                carbon = row["C"]
                grouped.setdefault(carbon, {"OP": [], "ERR": []})
                grouped[carbon]["OP"].append(float(row["OP"]))
                grouped[carbon]["ERR"].append(float(row["ERR"]))
        reduced = {}
        for carbon, values in grouped.items():
            reduced[carbon] = {
                "OP": float(np.mean(values["OP"])),
                "ERR": float(np.mean(values["ERR"])),
            }
        return reduced

    def _collect_aligned_points(sim_grouped, exp_grouped, xpos_getter):
        x_vals = []
        y_sim_vals = []
        y_sim_errs = []
        y_exp_vals = []
        x_exp_vals = []

        aligned = []
        for carbon in set(sim_grouped).intersection(exp_grouped):
            xpos = xpos_getter(carbon)
            if xpos is None:
                continue
            aligned.append((xpos, carbon))

        aligned.sort(key=lambda x: x[0])

        for xpos, carbon in aligned:
            x_vals.append(xpos)
            y_sim_vals.append(sim_grouped[carbon]["OP"])
            y_sim_errs.append(sim_grouped[carbon]["ERR"])
            y_exp_vals.append(exp_grouped[carbon]["OP"])
            x_exp_vals.append(xpos)

        return x_vals, y_sim_vals, y_sim_errs, y_exp_vals, x_exp_vals

    lipid_obj = Lipid(lipid_name)
    sim_rows = _build_registry_rows(OPsim, lipid_obj)
    exp_rows = _build_registry_rows(OPexp, lipid_obj)

    xValuesHG = []  # noqa: N806
    xValuesSN1 = []  # noqa: N806
    xValuesSN2 = []  # noqa: N806

    yValuesHGsim = []  # noqa: N806
    yValuesSN1sim = []  # noqa: N806
    yValuesSN2sim = []  # noqa: N806
    yValuesHGsimERR = []  # noqa: N806
    yValuesSN1simERR = []  # noqa: N806
    yValuesSN2simERR = []  # noqa: N806
    yValuesHGexp = []  # noqa: N806
    yValuesSN1exp = []  # noqa: N806
    yValuesSN2exp = []  # noqa: N806
    xValuesHGexp = []  # noqa: N806
    xValuesSN1exp = []  # noqa: N806
    xValuesSN2exp = []  # noqa: N806

    sim_sn1_grouped = _group_by_carbon(sim_rows, ["sn-1"])
    exp_sn1_grouped = _group_by_carbon(exp_rows, ["sn-1"])
    sim_sn2_grouped = _group_by_carbon(sim_rows, ["sn-2"])
    exp_sn2_grouped = _group_by_carbon(exp_rows, ["sn-2"])
    sim_hg_grouped = _group_by_carbon(sim_rows, ["headgroup", "glycerol backbone"])
    exp_hg_grouped = _group_by_carbon(exp_rows, ["headgroup", "glycerol backbone"])

    hg_positions = {
        "γ": 1,
        "β": 2,
        "α": 3,
        "g1": 4,
        "g2": 5,
        "g3": 6,
    }

    xValuesSN1, yValuesSN1sim, yValuesSN1simERR, yValuesSN1exp, xValuesSN1exp = _collect_aligned_points(
        sim_sn1_grouped,
        exp_sn1_grouped,
        lambda c: int(c),
    )
    xValuesSN2, yValuesSN2sim, yValuesSN2simERR, yValuesSN2exp, xValuesSN2exp = _collect_aligned_points(
        sim_sn2_grouped,
        exp_sn2_grouped,
        lambda c: int(c),
    )
    xValuesHG, yValuesHGsim, yValuesHGsimERR, yValuesHGexp, xValuesHGexp = _collect_aligned_points(
        sim_hg_grouped,
        exp_hg_grouped,
        lambda c: hg_positions.get(c),
    )
    plt.rc("font", size=15)
    if xValuesHG:
        plt.errorbar(
            xValuesHGexp,
            yValuesHGexp,
            yerr=0.02,
            fmt=".",
            color="black",
            markersize=25,
        )
        plt.errorbar(
            xValuesHG,
            yValuesHGsim,
            yerr=yValuesHGsimERR,
            fmt=".",
            color="red",
            markersize=20,
        )
        my_xticks = ["\u03b3", "\u03b2", "\u03b1", "$g_{1}$", "$g_{2}$", "$g_{3}$"]
        plt.xticks([1, 2, 3, 4, 5, 6], my_xticks, size=20)
        plt.yticks(size=20)
        plt.ylabel(r"$S_{CH}$", size=25)
        plt.savefig("HG.pdf")
        plt.show()

    if xValuesSN1:
        plt.text(2, -0.04, "sn-1", fontsize=25)
        plt.xticks(np.arange(min(xValuesSN1), max(xValuesSN1) + 1, 2.0))
        plt.plot(xValuesSN1, yValuesSN1sim, color="red")
        plt.plot(xValuesSN1exp, yValuesSN1exp, color="black")
        plt.errorbar(
            xValuesSN1,
            yValuesSN1sim,
            yerr=yValuesSN1simERR,
            fmt=".",
            color="red",
            markersize=25,
        )
        plt.errorbar(
            xValuesSN1exp,
            yValuesSN1exp,
            yerr=0.02,
            fmt=".",
            color="black",
            markersize=20,
        )
        plt.ylabel(r"$S_{CH}$", size=25)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.savefig("sn-1.pdf")
        plt.show()

    if xValuesSN2:
        plt.text(2, -0.04, "sn-2", fontsize=25)
        plt.xticks(np.arange(min(xValuesSN2), max(xValuesSN2) + 1, 2.0))
        plt.plot(xValuesSN2, yValuesSN2sim, color="red")
        plt.plot(xValuesSN2exp, yValuesSN2exp, color="black")
        plt.errorbar(
            xValuesSN2,
            yValuesSN2sim,
            yValuesSN2simERR,
            fmt=".",
            color="red",
            markersize=25,
        )
        plt.errorbar(
            xValuesSN2exp,
            yValuesSN2exp,
            yerr=0.02,
            fmt=".",
            color="black",
            markersize=20,
        )
        plt.xlabel("Carbon", size=25)
        plt.ylabel(r"$S_{CH}$", size=25)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.savefig("sn-2.pdf")
        plt.show()


def plotGenOrderParameter(OPsim, OPexp, lipid_name):  # noqa: N802
    """Generic OP plotter driven by fragment naming registry.

    Builds registry-formatted OP dictionaries for simulation and experiment
    and renders one panel per fragment shared by both datasets.
    """

    def _build_registry_rows(op_data, lipid_obj):
        formatted = build_nice_OPdict(op_data, lipid_obj)
        for fragment in formatted:
            for row in formatted[fragment]:
                row["ERR"] = 0.0 if row["STD"] is None else float(row["STD"])
        return formatted

    def _group_fragment_rows(rows):
        grouped = {}
        for row in rows:
            carbon = row["C"]
            grouped.setdefault(carbon, {"OP": [], "ERR": []})
            grouped[carbon]["OP"].append(float(row["OP"]))
            grouped[carbon]["ERR"].append(float(row["ERR"]))
        reduced = {}
        for carbon, values in grouped.items():
            reduced[carbon] = {
                "OP": float(np.mean(values["OP"])),
                "ERR": float(np.mean(values["ERR"])),
            }
        return reduced

    def _default_xpos(carbon):
        if re.fullmatch(r"[0-9]+", carbon):
            return int(carbon)
        label_positions = {
            "γ": 1,
            "β": 2,
            "α": 3,
            "g1": 4,
            "g2": 5,
            "g3": 6,
            "g1'": 7,
            "g2'": 8,
            "g3'": 9,
        }
        return label_positions.get(carbon)

    def _format_xtick(carbon):
        if re.fullmatch(r"g([0-9]+)", carbon):
            idx = re.fullmatch(r"g([0-9]+)", carbon).group(1)
            return f"$g_{{{idx}}}$"
        if re.fullmatch(r"g([0-9]+)'", carbon):
            idx = re.fullmatch(r"g([0-9]+)'", carbon).group(1)
            return f"$g_{{{idx}}}'$"
        return carbon

    def _sanitize_fname(name):
        return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")

    lipid_obj = Lipid(lipid_name)
    sim_rows = _build_registry_rows(OPsim, lipid_obj)
    exp_rows = _build_registry_rows(OPexp, lipid_obj)

    common_fragments = sorted(set(sim_rows).intersection(exp_rows))
    for fragment in common_fragments:
        sim_grouped = _group_fragment_rows(sim_rows.get(fragment, []))
        exp_grouped = _group_fragment_rows(exp_rows.get(fragment, []))

        aligned = []
        for carbon in set(sim_grouped).intersection(exp_grouped):
            xpos = _default_xpos(carbon)
            if xpos is None:
                continue
            aligned.append((xpos, carbon))
        aligned.sort(key=lambda x: x[0])
        if not aligned:
            continue

        x_vals = [x for x, _ in aligned]
        y_sim = [sim_grouped[c]["OP"] for _, c in aligned]
        y_sim_err = [sim_grouped[c]["ERR"] for _, c in aligned]
        y_exp = [exp_grouped[c]["OP"] for _, c in aligned]

        plt.rc("font", size=15)
        plt.plot(x_vals, y_sim, color="red")
        plt.plot(x_vals, y_exp, color="black")
        plt.errorbar(
            x_vals,
            y_sim,
            yerr=y_sim_err,
            fmt=".",
            color="red",
            markersize=25,
        )
        plt.errorbar(
            x_vals,
            y_exp,
            yerr=0.02,
            fmt=".",
            color="black",
            markersize=20,
        )

        labels = [c for _, c in aligned]
        if all(re.fullmatch(r"[0-9]+", c) for c in labels):
            plt.xticks(np.arange(min(x_vals), max(x_vals) + 1, 2.0))
        else:
            plt.xticks(x_vals, [_format_xtick(c) for c in labels], size=20)

        plt.text(min(x_vals), -0.04, fragment, fontsize=25)
        plt.ylabel(r"$S_{CH}$", size=25)
        plt.xlabel("Carbon", size=25)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.savefig(f"{_sanitize_fname(fragment)}.pdf")
        plt.show()


def _as_id_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(_as_id_list(item))
        return out
    if isinstance(value, Iterable):
        out = []
        for item in value:
            out.extend(_as_id_list(item))
        return out
    return [str(value)]


def _load_op_experiment_dict(lipid: str, exp_ids: list[str]) -> dict:
    op_exp_flat = {}
    ExperimentCollection.clear_instance()
    op_collection = ExperimentCollection.load_from_data("OPExperiment")
    for exp_id in exp_ids:
        exp = op_collection.get(exp_id)
        if exp is None:
            continue
        lipid_data = exp.data.get(lipid)
        if not lipid_data:
            continue
        op_exp_flat.update(lipid_data)
    return op_exp_flat


def _ff_to_curve(ff_data):
    if ff_data is None:
        return None
    if isinstance(ff_data, np.ndarray):
        if ff_data.ndim != 2 or ff_data.shape[1] < 2:
            return None
        return ff_data.tolist()
    if isinstance(ff_data, list):
        return ff_data if ff_data else None
    if isinstance(ff_data, dict):
        q_vals = ff_data.get("q")
        i_vals = ff_data.get("I")
        if q_vals is None or i_vals is None:
            return None
        n_vals = min(len(q_vals), len(i_vals))
        return [[float(q_vals[i]), float(i_vals[i])] for i in range(n_vals)]
    return None


def _load_first_ff_experiment_curve(exp_ids: list[str]):
    ExperimentCollection.clear_instance()
    ff_collection = ExperimentCollection.load_from_data("FFExperiment")
    for exp_id in exp_ids:
        exp = ff_collection.get(exp_id)
        if exp is None:
            continue
        curve = _ff_to_curve(exp.data)
        if curve is not None:
            return exp_id, curve
    return None, None


def plotSimulation(system, lipid: str):  # noqa: N802
    """
    Creates plots of form factor and C-H bond order parameters for the selected
    ``lipid`` from a simulation given by system.

    :param system: FAIRMD Lipids ID number of the simulation
    :param lipid: universal molecul name of the lipid

    """
    print("DOI: ", system.get("DOI", "N/A"))

    ff_quality = np.nan
    try:
        ff_quality = get_quality(system, experiment="FF")
        if not np.isnan(ff_quality):
            print("Form factor quality: ", ff_quality)
    except Exception:
        print("Force field quality not found")

    try:
        op_sim = get_OP(system).get(lipid) or {}
    except Exception:
        op_sim = {}

    ff_ids = _as_id_list(system.get("EXPERIMENT", {}).get("FORMFACTOR", []))
    op_ids = _as_id_list(system.get("EXPERIMENT", {}).get("ORDERPARAMETER", {}).get(lipid, []))
    op_exp = _load_op_experiment_dict(lipid, op_ids)
    ff_exp_id, ff_exp = _load_first_ff_experiment_curve(ff_ids)

    try:
        ff_sim_raw = get_FF(system)
        ff_sim = _ff_to_curve(ff_sim_raw)
        if ff_sim is not None:
            plotFormFactor(ff_sim, 1, "Simulation", "red")
        if ff_exp is not None:
            ff_scale = 1.0
            if ff_sim is not None:
                try:
                    ff_scale = float(calc_ff_scaling_distance(np.array(ff_exp), np.array(ff_sim))[0])
                    if ff_exp_id is not None:
                        print(f"Using FF scale {ff_scale:.4g} from experiment {ff_exp_id}.")
                except Exception:
                    print("Could not compute FF scale; using 1.0.")
            plotFormFactor(ff_exp, ff_scale, "Experiment", "black")
        plt.show()
    except Exception:
        plt.show()
        print("Form factor plotting failed")

    if op_sim and op_exp:
        plotGenOrderParameter(op_sim, op_exp, lipid)
    elif not op_sim:
        print(f"Simulation OP data missing for {lipid}.")
    else:
        print(f"No matched OP experiment data found for {lipid}.")
