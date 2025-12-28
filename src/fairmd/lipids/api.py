"""
API functions for analyzing the FAIRMD Lipids Databank.

Functions are organized into few groups:
1. Functions that extract computed properties:
    - get_OP
    - get_thickness
    - get_eqtimes
2. Functions that extract post-processed properties:
    - get_mean_ApL
    - get_total_area
    - get_formfactor_mins
3. Auxiliary functions for better interface with MDAnalysis
    - mda_gen_selection_mols
"""

import json
import logging
import math
import os
import sys
import warnings
from collections.abc import Container

import MDAnalysis as mda
import numpy as np

from fairmd.lipids import FMDL_SIMU_PATH
from fairmd.lipids.core import System
from fairmd.lipids.databankio import download_resource_from_uri, resolve_file_url
from fairmd.lipids.molecules import Molecule, lipids_set
from fairmd.lipids.SchemaValidation.engines import get_struc_top_traj_fnames

logger = logging.getLogger(__name__)


def get_thickness(system: System) -> float:
    """
    Get thickness for a simulation defined with ``system`` from the ``thickness.json``.

    :param system: FAIRMD Lipids dictionary defining a simulation.

    :return: membrane thickess (nm) or raise exception
    """
    thickness_path = os.path.join(FMDL_SIMU_PATH, system["path"], "thickness.json")
    try:
        with open(thickness_path) as f:
            thickness = json.load(f)
        thickness_v = float(thickness)
    except FileNotFoundError:
        print("No thickness information for system#{}.".format(system["ID"]), file=sys.stderr)
        raise
    except ValueError:
        print("Thickness information for system#{} is invalid.".format(system["ID"]), file=sys.stderr)
        raise
    else:
        return thickness_v


def get_eqtimes(system: System) -> dict:
    """
    Return relative equilibration time for each lipid of ``system``.

    :param system: Simulation object.

    :return: dictionary of relative equilibration times for each lipid
    """
    eq_times_path = os.path.join(FMDL_SIMU_PATH, system["path"], "eq_times.json")

    try:
        with open(eq_times_path) as f:
            eq_time_dict = json.load(f)
    except FileNotFoundError:
        print("No thickness information for system#{}.".format(system["ID"]), file=sys.stderr)
        raise
    except json.JSONDecodeError:
        print("Equilibration times information for system#{} is invalid.".format(system["ID"]), file=sys.stderr)
        raise

    return eq_time_dict


def get_OP(system: System) -> dict:  # noqa: N802 (API name)
    """
    Return a dictionary with the order parameter data for each lipid in ``system``.

    :param system: NMRlipids databank dictionary defining a simulation.

    :return: dictionary contaning, for each lipid, the order parameter data:
             average OP, standard deviation, and standard error of mean. Contains
             None if ``LipidNameOrderParameters.json`` missing.
    """
    sim_op_data = {}  # order parameter data for each type of lipid
    for mol in system["COMPOSITION"]:
        if mol not in lipids_set:
            continue
        fname = os.path.join(
            FMDL_SIMU_PATH,
            system["path"],
            mol + "OrderParameters.json",
        )
        if not os.path.isfile(fname):
            warnings.warn(f"{fname} not found for {system['ID']}", stacklevel=2)
            sim_op_data[mol] = None
            continue
        op_data = {}
        try:
            with open(fname) as json_file:
                op_data = json.load(json_file)
        except json.JSONDecodeError:
            print(
                f"Order parameter data in {fname} is invalid for {system['ID']}",
                file=sys.stderr,
            )
            raise
        sim_op_data[mol] = op_data
    return sim_op_data


def get_mean_ApL(system: System) -> float:  # noqa: N802 (API name)
    """
    Calculate average area per lipid for a system.

    :param system: FAIRMD Lipids dictionary defining a simulation.

    :return: area per lipid (Å^2)
    """
    path = os.path.join(FMDL_SIMU_PATH, system["path"], "apl.json")
    if not os.path.isfile(path):
        msg = "apl.json not found from" + path
        raise FileNotFoundError(msg)
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("Area per lipid data for system #{} in {} is invalid.".format(system["ID"], path), file=sys.stderr)
        raise
    vals = np.array(list(data.values()))
    return vals.mean()


def get_total_area(system: System) -> float:
    """
    Return area of the membrane in the simulation box.

    :param system: a system dictionary

    :return: area of the system (Å^2)
    """
    apl = get_mean_ApL(system)
    return system.n_lipids * apl / 2


def get_formfactor_mins(system: System) -> list:
    """
    Return list of minima of form factor of ``system``.

    :param system: a system dictionary

    :return: list of form factor minima or raise exception
    """
    form_factor_path = os.path.join(FMDL_SIMU_PATH, system["path"], "FormFactor.json")
    if not os.path.isfile(form_factor_path):
        msg = "{} not found for system #{}".format(form_factor_path, system["ID"])
        raise FileNotFoundError(msg)
    with open(form_factor_path) as f:
        form_factor = json.load(f)
    iprev = form_factor[0][1]
    iprev_d = 0
    min_x = []
    for i in form_factor:
        i_d = i[1] - iprev
        if i_d > 0 and iprev_d < 0 and i[0] > 0.1:
            min_x.append(i[0])
        iprev_d = i[1] - iprev
        iprev = i[1]

    return min_x


def mda_gen_selection_mols(system: System, molecules: Container[Molecule] | None = None) -> str:
    """
    Return a MDAnalysis selection string covering all the molecules (default None means "lipids").

    :param system: FAIRMD Lipids dictionary defining a simulation.
    :param molecules: container of molecule objects to be included in the selection.

    :return: a string using MDAnalysis notation that can used to select all lipids from
             the ``system``.
    """
    res_set = set()
    molecules = system.lipids.values() if molecules is None else molecules
    for key, mol in system.content.items():
        if mol in molecules:
            try:
                for atom in mol.mapping_dict:
                    res_set.add(mol.mapping_dict[atom]["RESIDUE"])
            except (KeyError, TypeError):
                res_set.add(system["COMPOSITION"][key]["NAME"])
    sorted_res = sorted(res_set)
    return "resname " + " or resname ".join(sorted_res)


class UniverseConstructError(Exception):
    """Specific error for UniverseConstructor"""

    pass


class UniverseConstructor:
    """
    Class operating with downloading and constructing Universe for the Databank `System`.

    To use this class, one instantinate it with a particular system, and then download.

    :: code
        s = systems.loc(120)
        uc = UniverseConstructor(s)
        uc.download_mddata()

    After this, the pointer `uc.path` will show which files are available to work with.
    """

    def __init__(self, s: System):
        self._s = s
        self._paths = {
            "struc": None,
            "top": None,
            "traj": None,
            "energy": None,
        }

    @property
    def system(self):
        return self._s

    @property
    def paths(self):
        """Return dicts of absolute paths of downloaded files: struc, traj, top, energy."""
        return self._paths

    def download_mddata(self, skip_traj=False) -> None:
        """
        Download all the files. Previously downloaded are skipped.

        :param skip_traj: Download only TOP&struc for further constructing single-frame universe
        """
        gpath = os.path.join(FMDL_SIMU_PATH, self._s["path"])
        struc, top, trj = get_struc_top_traj_fnames(self._s)

        def _resolve_dwnld(fname):
            fpath = os.path.join(gpath, fname)
            if self._s["DOI"] == "localhost":
                if not os.path.isfile(fpath):
                    msg = f"File {fpath} must be predownloaded for {self._s}"
                    raise FileNotFoundError(msg)
                return fpath
            if os.path.isfile(fpath):
                # do not download if exists
                return fpath
            url = resolve_file_url(self._s["DOI"], fname)
            _ = download_resource_from_uri(url, fpath)
            return fpath

        if struc is not None:
            self._paths["struc"] = _resolve_dwnld(struc)
        if top is not None:
            self._paths["top"] = _resolve_dwnld(top)
        if trj is not None and not skip_traj:
            self._paths["traj"] = _resolve_dwnld(trj)

    def clear_mddata(self) -> None:
        """Clear downloaded MD data. For DOI=localhost, do nothing."""
        if self._s["DOI"] == "localhost":
            for k in self._paths:
                self._paths[k] = None
            return
        for k, v in self._paths.items():
            if v is None:
                continue
            print(f"Clearing {k}-file..", end="", flush=True)
            os.remove(v)
            print("OK")
            self._paths[k] = None

    def build_universe(self) -> mda.Universe:
        """Build MDAnalysis Universe.

        Replaces outdated `system2MDanalysisUniverse`."""
        if not any(self._paths.values()):
            msg = "You **MUST** run `download_mddata` before `build_universe`"
            raise UniverseConstructError(msg)

        if self._paths["top"] is None:
            if self._paths["traj"] is None:
                u = mda.Universe(self._paths["struc"])
            else:
                u = mda.Universe(self._paths["struc"], self._paths["traj"])
        else:
            try:
                if self._paths["traj"] is None:
                    if self._paths["struc"] is None:
                        u = mda.Universe(self._paths["top"])
                    else:
                        u = mda.Universe(self._paths["top"], self._paths["struc"])
                else:
                    u = mda.Universe(self._paths["top"], self._paths["traj"])
            except IOError as e:
                print(
                    f"We got exception.. == \n{e}\n == ..and assume that TOPOLOGY is file is corrupted", file=sys.stderr
                )
                if self._paths["struc"] is None:
                    raise UniverseConstructError("TOPOLOGY is corrupted, and no STRUCTURE is given") from e
                if self._paths["traj"] is None:
                    u = mda.Universe(self._paths["struc"])
                else:
                    u = mda.Universe(self._paths["struc"], self._paths["traj"])
        return u


def calc_angle(atoms, com):
    """
    calculates the angle between the vector and z-axis in degrees
    no PBC check!
    Calculates the center of mass of the selected atoms to invert bottom leaflet vector
    """
    vec = atoms[1].position - atoms[0].position
    d = math.sqrt(np.square(vec).sum())
    cos = vec[2] / d
    # values for the bottom leaflet are inverted so that
    # they have the same nomenclature as the top leaflet
    cos *= math.copysign(1.0, atoms[0].position[2] - com)
    try:
        angle = math.degrees(math.acos(cos))
    except ValueError:
        if abs(cos) >= 1.0:
            print(f"Cosine is too large = {cos} --> truncating it to +/-1.0")
            cos = math.copysign(1.0, cos)
            angle = math.degrees(math.acos(cos))
    return angle


def read_trj_PN_angles(  # noqa: N802 (API name)
    molname: str,
    atom1: str,
    atom2: str,
    mda_universe: mda.Universe,
):
    """
    Calculate the P-N vector angles with respect to membrane normal from the
    simulation defined by the MDAnalysis universe.

    :param molname: residue name of the molecule for which the P-N vector angle will
                    be calculated
    :param atom1: name of the P atom in the simulation
    :param atom2: name of the N atom in the simulation
    :param MDAuniverse: MDAnalysis universe of the simulation to be analyzed

    :return: tuple (angles of all molecules as a function of time,
                    time averages for each molecule,
                    the average angle over time and molecules,
                    the error of the mean calculated over molecules)
    """
    mol = mda_universe
    selection = mol.select_atoms(
        "resname " + molname + " and (name " + atom1 + ")",
        "resname " + molname + " and (name " + atom2 + ")",
    ).atoms.split("residue")
    com = mol.select_atoms(
        "resname " + molname + " and (name " + atom1 + " or name " + atom2 + ")",
    ).center_of_mass()

    n_res = len(selection)
    n_frames = len(mol.trajectory)
    angles = np.zeros((n_res, n_frames))

    res_aver_angles = [0] * n_res
    res_std_error = [0] * n_res
    j = 0

    for _ in mol.trajectory:
        for i in range(n_res):
            residue = selection[i]
            angles[i, j] = calc_angle(residue, com[2])
        j = j + 1
    for i in range(n_res):
        res_aver_angles[i] = sum(angles[i, :]) / n_frames
        res_std_error[i] = np.std(angles[i, :])

    total_average = sum(res_aver_angles) / n_res
    total_std_error = np.std(res_aver_angles) / np.sqrt(n_res)

    return angles, res_aver_angles, total_average, total_std_error


# -------------------------------------- SEPARATED PART (??) ----------------------
