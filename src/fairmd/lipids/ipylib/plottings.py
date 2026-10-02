"""Implementation of plotting routines for IPython/Jupyter notebooks."""

import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from fairmd.lipids.api import get_OP
from fairmd.lipids.auxiliary.opconvertor import build_nice_OPdict
from fairmd.lipids.core import System
from fairmd.lipids.experiment import ExperimentCollection
from fairmd.lipids.molecules import Lipid

from .plotff import plot_simulation_FF

# Define a default plotting style for OP
fmdl_plot_style = {
    "font.size": 13,
    "figure.figsize": (8.5, 5.5),
    "figure.dpi": 180,
    "label_size": 16,
    "tick_size": 13,
    "tick_width": 1.2,
    "tick_length": 6,
    "simulation": {
        "fmt": "s",
        "label": "Simulation",
        "color": "red",
        "markersize": 9,
        "markeredgecolor": "black",
        "markeredgewidth": 0.7,
        "capsize": 2,
    },
    "experimental": {
        "fmt": "o",
        "label": "Experimental",
        "color": "blue",
        "markersize": 9,
        "markeredgecolor": "black",
        "markeredgewidth": 0.7,
        "capsize": 2,
    },
}


def plotGenOrderParameter(op_sim: dict, op_exp: dict, lipid_name: str):  # noqa: N802
    """Plot generic OP data by using fragment naming registry.

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
    sim_rows = _build_registry_rows(op_sim, lipid_obj)
    exp_rows = _build_registry_rows(op_exp, lipid_obj)

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
        plt.title(f"{lipid_name}: {fragment}", size=20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.savefig(f"{_sanitize_fname(fragment)}.pdf")
        plt.show()


def plot_regular_phospholipid_op(op_sim: dict, op_exp: dict | None, lipid_obj: Lipid) -> tuple[Figure, Figure, Figure]:
    """Plot simulation and experimental order parameters by fragment."""

    def _prep_df(op_dict: dict) -> dict:
        dfdic = build_nice_OPdict(op_dict, lipid_obj)
        # unite glcerol + head into head
        dfdic["head"] = dfdic.get("glycerol backbone", []) + dfdic.get("headgroup", [])
        dfdic.pop("glycerol backbone", None)
        dfdic.pop("headgroup", None)
        for frag in dfdic:
            _df = pd.DataFrame(dfdic[frag])
            if _df.empty:
                dfdic[frag] = None
                continue
            _df.index = _df.index.astype(int)
            _df = _df.sort_index()
            dfdic[frag] = _df
        return dfdic

    sim_nice_opdict = _prep_df(op_sim)
    exp_nice_opdict = _prep_df(op_exp) if op_exp else {}

    plt.rcParams.update({"font.size": fmdl_plot_style["font.size"]})
    fig_list = []
    for frag in sim_nice_opdict:
        simdf = sim_nice_opdict[frag]
        expdf = exp_nice_opdict.get(frag, None)
        figure, axis = plt.subplots(
            figsize=fmdl_plot_style["figure.figsize"],
            dpi=fmdl_plot_style["figure.dpi"],
        )
        axis.set_title(f"{lipid_obj.name} : {frag}", fontsize=fmdl_plot_style["label_size"])
        axis.errorbar(
            simdf["C"],
            simdf["OP"],
            yerr=simdf["STD"],
            **fmdl_plot_style["simulation"],
        )
        if expdf is not None:
            axis.errorbar(
                expdf["C"],
                expdf["OP"],
                yerr=expdf["STD"],
                **fmdl_plot_style["experimental"],
            )
        axis.set_xticks(simdf.C)
        axis.set_xlabel("Carbon", fontsize=fmdl_plot_style["label_size"])
        axis.set_ylabel(r"$S_{CH}$", fontsize=fmdl_plot_style["label_size"])
        axis.tick_params(
            axis="both",
            which="major",
            labelsize=fmdl_plot_style["tick_size"],
            width=fmdl_plot_style["tick_width"],
            length=fmdl_plot_style["tick_length"],
        )
        figure.tight_layout()
        fig_list.append(figure)
    return tuple(fig_list)


def plot_simulation_OP(system: System, lipid: str) -> tuple[Figure, Figure, Figure]:  # noqa: N802
    """Plot simulated and experimental C-H bond order parameters."""
    op_sim = get_OP(system).get(lipid)
    if op_sim is None:
        msg = f"Order parameter data not found for {lipid}"
        raise FileNotFoundError(msg)

    op_exp = {}
    op_experiments = ExperimentCollection.load_from_data("OPExperiment")
    for exp_op_id in system["EXPERIMENT"]["ORDERPARAMETER"][lipid]:
        experiment = op_experiments.get(exp_op_id)
        if experiment is not None:
            op_exp.update(experiment.data.get(lipid, {}))

    return plot_regular_phospholipid_op(op_sim, op_exp, system.lipids[lipid])


def plotSimulation(system: System, lipid: str) -> None:  # noqa: N802
    """Plot form factors and order parameters (deprecated)."""
    warnings.warn(
        "plotSimulation is deprecated; use plot_simulation_FF and plot_simulation_OP instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    plot_simulation_FF(system)
    plot_simulation_OP(system, lipid)
