"""
Library contains all API functions and many functions used in building and
analyzing the FAIRMD Lipids
"""

import json
import logging
import math
import os
import subprocess
import warnings

import MDAnalysis as mda
import numpy as np

from fairmd.lipids import FMDL_SIMU_PATH
from fairmd.lipids.core import System
from fairmd.lipids.databankio import download_resource_from_uri, resolve_file_url
from fairmd.lipids.molecules import lipids_set
from fairmd.lipids.SchemaValidation.engines import get_struc_top_traj_fnames

logger = logging.getLogger(__name__)


def CalcAreaPerMolecule(system) -> None | float:  # noqa: N802 (API name)
    """
    Calculate average area per lipid for a system.

    It is using the ``apl.json`` file where area per lipid as a function of time
    calculated by the ``calcAPL.py`` is stored.

    :param system: FAIRMD Lipids dictionary defining a simulation.

    :return: area per lipid (Å^2)
    """
    path = os.path.join(FMDL_SIMU_PATH, system["path"], "apl.json")
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        print("apl.json not found from" + path)
        return None
    sum_apl = 0
    sum_ind = 0
    for j in data.values():
        sum_apl += j
        sum_ind += 1
    return sum_apl / sum_ind


def GetThickness(system):  # noqa: N802 (API name)
    """
    Gets thickness for a simulation defined with ``system`` from the ``thickness.json``
    file where thickness calculated by the ``calc_thickness.py`` is stored.

    :param system: FAIRMD Lipids dictionary defining a simulation.

    :return: membrane thickess (nm) or None
    """
    thickness_path = os.path.join(FMDL_SIMU_PATH, system["path"], "thickness.json")
    try:
        with open(thickness_path) as f:
            thickness = json.load(f)
        return thickness
    except Exception:
        return None


def ShowEquilibrationTimes(system: System):  # noqa: N802 (API name)
    """
    Prints relative equilibration time for each lipid within a simulation defined
    by ``system``. Relative equilibration times are calculated with
    ``NMRPCA_timerelax.py`` and stored in ``eq_times.json`` files.

    :param system: FAIRMD Lipids dictionary defining a simulation.
    """
    warnings.warn(
        "This function is deprecated. Use GetEquilibrationTimes instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    eq_times_path = os.path.join(FMDL_SIMU_PATH, system["path"], "eq_times.json")

    try:
        with open(eq_times_path) as f:
            eq_time_dict = json.load(f)
    except Exception as err:
        raise FileNotFoundError(f"eq_times.json not found for {system['ID']}") from err

    for i in eq_time_dict:
        print(i + ":", eq_time_dict[i])


def GetEquilibrationTimes(system: System):  # noqa: N802 (API name)
    """
    Returns relative equilibration time for each lipid within a simulation defined
    by ``system``. Relative equilibration times are calculated with
    ``NMRPCA_timerelax.py`` and stored in ``eq_times.json`` files.

    :param system: FAIRMD Lipids dictionary defining a simulation.

    :return: dictionary of relative equilibration times for each lipid
    """
    eq_times_path = os.path.join(FMDL_SIMU_PATH, system["path"], "eq_times.json")

    try:
        with open(eq_times_path) as f:
            eq_time_dict = json.load(f)
    except Exception as err:
        raise FileNotFoundError(f"eq_times.json not found for {system['ID']}") from err

    return eq_time_dict


def GetOP(system):  # noqa: N802 (API name)
    """
    Returns a dictionary containing the order parameter data time for each lipid in the
    ``system``, stored in ``LipidNameOrderParameters.json`` files.

    :param system: NMRlipids databank dictionary defining a simulation.

    :return: dictionary contaning, for each lipid, the order parameter data: average OP, standard deviation,
     and standard error of mean. Contains None if ``LipidNameOrderParameters.json`` missing.
    """
    SimOPdata = {}  # order parameter data for each type of lipid
    for mol in system["COMPOSITION"]:
        if mol not in lipids_set:
            continue
        fname = os.path.join(
            FMDL_SIMU_PATH,
            system["path"],
            mol + "OrderParameters.json",
        )
        OPdata = {}
        try:
            with open(fname) as json_file:
                OPdata = json.load(json_file)
        except FileNotFoundError:
            missingName = mol + "OrderParameters.json"
            warnings.warn(f"{missingName} not found for {system['ID']}", stacklevel=2)

            OPdata = None

        SimOPdata[mol] = OPdata
    return SimOPdata


def GetNlipids(system: System):  # noqa: N802 (API name)
    """
    Returns the total number of lipids in a simulation defined by ``system``.

    :param system: FAIRMD Lipids dictionary defining a simulation.

    :return: the total number of lipids in the ``system``.
    """
    n_lipid = 0
    for molecule in system["COMPOSITION"]:
        if molecule in lipids_set:
            n_lipid += np.sum(system["COMPOSITION"][molecule]["COUNT"])
    return n_lipid


def getLipids(system: System, molecules=lipids_set):  # noqa: N802 (API name)
    """
    Returns a string using MDAnalysis notation that can used to select all lipids from
    the ``system``.

    :param system: FAIRMD Lipids dictionary defining a simulation.

    :return: a string using MDAnalysis notation that can used to select all lipids from
             the ``system``.
    """
    res_set = set()
    for key, mol in system.content.items():
        if key in molecules:
            try:
                for atom in mol.mapping_dict:
                    res_set.add(mol.mapping_dict[atom]["RESIDUE"])
            except (KeyError, TypeError):
                res_set.add(system["COMPOSITION"][key]["NAME"])

    lipids = "resname " + " or resname ".join(sorted(list(res_set)))

    return lipids


def getAtoms(system: System, lipid: str):  # noqa: N802 (API name)
    """
    Return system specific atom names of a lipid

    :param system: System simulation object
    :param lipid: universal lipid name

    :return: string of system specific atom names
    """
    atoms = ""
    mdict = system.content[lipid].mapping_dict
    for key in mdict:
        atoms = atoms + " " + mdict[key]["ATOMNAME"]

    return atoms


def calc_angle(atoms, com):
    """
    :meta private:
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


def calc_z_dim(gro):
    """
    :meta private:
    Returns the simulation box dimension in z-direction from coordinate file.

    :param gro: coordinate in ``gro``, ``pdb`` or corresponding format.

    :return: size of box z-direction.
    """
    u = mda.Universe(gro)
    z = u.dimensions[2]
    return z


def system2MDanalysisUniverse(system):  # noqa: N802 (API name)
    """
    Takes the ``system`` dictionary as an input, downloads the required files to
    the FAIRMD Lipids directory and retuns MDAnalysis universe corressponding
    the ``system``.

    :param system: FAIRMD Lipids dictionary describing the simulation.

    :return: MDAnalysis universe
    """
    system_path = os.path.join(FMDL_SIMU_PATH, system["path"])
    doi = system.get("DOI")
    skip_downloading: bool = doi == "localhost"
    if skip_downloading:
        print("NOTE: The system with 'localhost' DOI should be downloaded by the user.")

    try:
        struc, top, trj = get_struc_top_traj_fnames(system)
        trj_name = os.path.join(system_path, trj)
        if struc is None:
            struc_name = None
        else:
            struc_name = os.path.join(system_path, struc)
        if top is None:
            top_name = None
        else:
            top_name = os.path.join(system_path, top)
    except Exception:
        logger.exception(f"Error getting structure/topology/trajectory filenames for system {system['ID']}.")
        raise

    # downloading trajectory (obligatory)
    if skip_downloading:
        if not os.path.isfile(trj_name):
            msg = (f"Trajectory should be downloaded [{trj_name}] by user",)
            raise FileNotFoundError(msg)
    else:
        trj_url = resolve_file_url(doi, trj)
        if not os.path.isfile(trj_name):
            print(
                "Downloading trajectory with the size of ",
                system["TRAJECTORY_SIZE"],
                " to ",
                system["path"],
            )
            _ = download_resource_from_uri(trj_url, trj_name)

    # downloading topology (if exists)
    if top is not None:
        if skip_downloading:
            if not os.path.isfile(top_name):
                msg = f"TPR should be downloaded [{top_name}]"
                raise FileNotFoundError(msg)
        else:
            top_url = resolve_file_url(doi, top)
            if not os.path.isfile(top_name):
                _ = download_resource_from_uri(top_url, top_name)

    # downloading structure (if exists)
    if struc is not None:
        if skip_downloading:
            if not os.path.isfile(struc_name):
                msg = f"GRO should be downloaded [{struc_name}]"
                raise FileNotFoundError(msg)
        else:
            struc_url = resolve_file_url(doi, struc)
            if not os.path.isfile(struc_name):
                _ = download_resource_from_uri(struc_url, struc_name)

    made_from_top = False
    try:
        u = mda.Universe(top_name, trj_name)
        made_from_top = True
    except Exception as e:
        logger.warning(f"Couldn't make Universe from {top_name} and {trj_name}.")
        logger.warning(str(e))

    if not made_from_top and struc is not None:
        made_from_struc = False
        try:
            u = mda.Universe(struc_name, trj_name)
            made_from_struc = True
        except Exception as e:
            logger.warning(f"Couldn't make Universe from {struc_name} and {trj_name}.")
            logger.warning(str(e))

        if not made_from_struc:
            if system["SOFTWARE"].upper() == "GROMACS":
                # rewrite struc_fname!
                struc_fname = os.path.join(system_path, "conf.gro")

                print(
                    "Generating conf.gro because MDAnalysis cannot (probably!) read tpr version",
                )
                if (
                    "WARNINGS" in system
                    and "GROMACS_VERSION" in system["WARNINGS"]
                    and system["WARNINGS"]["GROMACS_VERSION"] == "gromacs3"
                ):
                    command = ["editconf", "-f", top_name, "-o", struc_fname]
                else:
                    command = ["gmx", "trjconv", "-s", top_name, "-f", trj_name, "-dump", "0", "-o", struc_fname]
                try:
                    subprocess.run(command, input="System\n", text=True, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(
                        f"Command 'echo System | {' '.join(command)}' failed with error: {e.stderr}",
                    ) from e
                # the last try!
                u = mda.Universe(struc_fname, trj_name)
            else:
                raise RuntimeError("There is no way to build up your system!")

    return u


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

def calcArea(system) -> float:  # noqa: N802 (API name)
    """
    Return area of the calculated based on the area per lipid stored in the databank.

    :param system: a system dictionary

    :return: area of the system (Å^2)
    """
    APL = CalcAreaPerMolecule(system)  # noqa: N806
    n_lipid = 0
    for molecule in system["COMPOSITION"]:
        if molecule in lipids_set:
            n_lipid += np.sum(system["COMPOSITION"][molecule]["COUNT"])
    print(n_lipid, APL)
    return n_lipid * APL / 2


def GetFormFactorMin(system):  # noqa: N802 (API name)
    """
    Return list of minima of form factor of ``system``.

    :param system: a system dictionary

    :return: list of form factor minima
    """
    form_factor_path = os.path.join(FMDL_SIMU_PATH, system["path"], "FormFactor.json")
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


def averageOrderParameters(system):  # noqa: N802 (API name)
    """
    Returns average order paramaters of *sn*-1 and *sn*-2 acyl chains based on universal
    atom names. The names starting with M_G1C will be assigned to sn-1 and names
    starting M_G2C to *sn*-2.

    :parameters system: a system dictionary

    :return: average of *sn*-1 and *sn*-2 order parameters
    """
    path = os.path.join(FMDL_SIMU_PATH, system["path"])

    sn1sum = 0
    sn1count = 0
    sn2sum = 0
    sn2count = 0

    for lipid in system["COMPOSITION"]:
        if lipid in lipids_set and "CHOL" not in lipid:
            OP_path_sim = os.path.join(  # noqa: N806
                path,
                lipid + "OrderParameters.json",
            )
            with open(OP_path_sim) as json_file:
                OP_sim = json.load(json_file)  # noqa: N806

            for key in OP_sim:
                if "M_G1C" in key:
                    sn1sum += float(OP_sim[key][0][0])
                    sn1count += 1
                elif "M_G2C" in key:
                    sn2sum += float(OP_sim[key][0][0])
                    sn2count += 1

    return sn1sum / sn1count, sn2sum / sn2count


def calcLipidFraction(system, lipid):  # noqa: N802 (API name)
    """
    Return the number fraction of ``lipid`` with respect to total number of lipids.

    :param system: a system dictionary
    :param lipid: universal molecule name of lipid

    :return: number fraction of ``lipid`` with respect total number of lipids
    """
    n_lipid_tot = 0
    for molecule in system["COMPOSITION"]:
        if molecule in lipids_set:
            n_lipid_tot += np.sum(system["COMPOSITION"][molecule]["COUNT"])

    n_lipid = 0
    for molecule in system["COMPOSITION"]:
        if lipid in molecule:
            n_lipid += np.sum(system["COMPOSITION"][molecule]["COUNT"])

    return n_lipid / n_lipid_tot


def getHydrationLevel(system) -> float:  # noqa: N802 (API name)
    """
    Return hydration level of the system.

    Hydration level is defined as the number of water molecules divided by number of lipid molecules.

    :param system: a system dictionary

    :return: number of water molecules divided by number of lipid molecules
    """
    n_lipid = 0
    for molecule in system["COMPOSITION"]:
        if molecule in lipids_set:
            n_lipid += np.sum(system["COMPOSITION"][molecule]["COUNT"])
    n_water = system["COMPOSITION"]["SOL"]["COUNT"]
    return n_water / n_lipid
