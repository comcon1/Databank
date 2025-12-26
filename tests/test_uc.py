"""
`test_recompute_api.py` tests those parts of API, which require building MDAnalysis
universe and from-trajectories computations.

Test folder: ToyData/Simulations.1

NOTE: globally import of fairmd-lipids is **STRICTLY FORBIDDEN** because it
      breaks the substitution of global path folders
"""

import copy
import os
import sys
import tempfile

import pytest
import pytest_check as check
# Global fixtures
# ----------------------------------------------------------------

pytestmark = [pytest.mark.sim1]


@pytest.fixture(scope="module")
def systems():
    from fairmd.lipids import FMDL_DATA_PATH, FMDL_SIMU_PATH
    from fairmd.lipids.core import initialize_databank

    if os.path.isfile(os.path.join(FMDL_DATA_PATH, ".notest")):
        pytest.exit("Test are corrupted. I see '.notest' file in the data folder.")
    s = initialize_databank()
    print(f"Loaded: {len(s)} systems")
    yield s


def test_uc_cleardata(systems, capsys):
    """Test if UC can clear predownloaded raw data"""
    from fairmd.lipids.api import UniverseConstructor

    s = systems.loc(566)
    uc = UniverseConstructor(s)

    # mocking that we downloaded struc-file
    tf = tempfile.NamedTemporaryFile(delete=False)
    tf.close()
    uc._paths["struc"] = tf.name

    uc.clear_mddata()
    check.is_in("struc-file", capsys.readouterr().out)
    check.is_none(uc.paths["struc"])
    check.is_false(os.path.isfile(tf.name))


def test_uc_skipdwnl_ifexists(systems):
    from fairmd.lipids import FMDL_SIMU_PATH
    from fairmd.lipids.api import UniverseConstructor

    s = systems.loc(566)
    uc = UniverseConstructor(s)
    for _sfx in ["tpr", "gro", "xtc"]:
        with open(os.path.join(FMDL_SIMU_PATH, s["path"], f"566.{_sfx}"), "w") as fd:
            fd.write(' ')
    uc.download_mddata()
    check.equal(os.stat(uc.paths["struc"]).st_size, 1)
    check.equal(os.stat(uc.paths["top"]).st_size, 1)
    check.equal(os.stat(uc.paths["traj"]).st_size, 1)
    # TEARDOWN
    uc.clear_mddata()


@pytest.fixture(scope="function")
def uconstructor566(systems):
    from fairmd.lipids.api import UniverseConstructor

    s = systems.loc(566)
    uc = UniverseConstructor(s)
    check.equal(uc.system["ID"], 566)
    yield uc
    # TEARDOWN
    uc.clear_mddata()


@pytest.fixture(scope="function")
def uconstructor566nogro(systems):
    from fairmd.lipids.api import UniverseConstructor

    s = copy.deepcopy(systems.loc(566))
    del(s._store["GRO"])
    uc = UniverseConstructor(s)
    yield uc
    # TEARDOWN
    uc.clear_mddata()


def test_download_mddata_localhost(systems, capsys):
    from fairmd.lipids.api import UniverseConstructor

    s = systems.loc(243)
    uc = UniverseConstructor(s)
    # paths structure
    check.is_in("struc", uc.paths)
    check.is_in("top", uc.paths)
    check.is_in("traj", uc.paths)
    check.is_in("energy", uc.paths)

    uc.download_mddata()
    # md.gro is not downloaded
    check.is_true("md.gro" not in capsys.readouterr().err)
    # paths are filled in
    check.is_in(s["GRO"][0][0], uc.paths["struc"])
    check.is_true(os.path.isfile(uc.paths["struc"]))
    _store_fpath = uc.paths["struc"]
    uc.clear_mddata()
    check.is_none(uc.paths["struc"])
    check.is_true(os.path.isfile(_store_fpath)) # DOI-localhost systems are not really cleaned


def test_download_mddata_gromacs_gro_xtc_tpr(uconstructor566, capsys):
    uc = uconstructor566
    s = uc.system
    uc.download_mddata(skip_traj=True)
    assert("566.gro" in capsys.readouterr().err)

    # structure file is correct
    check.is_in(s["path"], uc.paths["struc"])
    check.is_in(s["GRO"][0][0], uc.paths["struc"])
    check.is_true(os.path.isfile(uc.paths["struc"]))

    # topology file is correct
    check.is_in(s["path"], uc.paths["top"])
    check.is_in(s["TPR"][0][0], uc.paths["top"])
    check.is_true(os.path.isfile(uc.paths["top"]))

    # traj is None for skip_traj=True
    check.is_none(uc.paths["traj"])


def test_download_mddata_gromacs_tpr_xtc(uconstructor566nogro):
    uc = uconstructor566nogro
    s = uc.system
    uc.download_mddata()
    check.is_none(uc.paths["struc"])


