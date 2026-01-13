"""Test the Experiment classes."""

import os
import pytest
import pytest_check as check
import yaml
import json


# run only on sim2 mocking data
pytestmark = [pytest.mark.exp]

@pytest.fixture(scope="module")
def dummy_experiments(tmp_path_factory):
    """Create a dummy experiment directory structure."""
    exp_path = tmp_path_factory.mktemp("databank_exp")
    
    # OP Experiment
    op_exp_path = exp_path / "OrderParameters" / "exp1"
    op_exp_path.mkdir(parents=True)
    readme_op = {
        "MOLAR_FRACTIONS": {"POPC": 1.0},
        "TEMPERATURE": 303,
        "ION_CONCENTRATIONS": {},
        "COUNTER_IONS": [],
        "TOTAL_LIPID_CONCENTRATION": "full hydration",
        "ARTICLE_DOI": "10.1021/acs.jpcb.5b01208",
    }
    with open(op_exp_path / "README.yaml", "w") as f:
        yaml.dump(readme_op, f)
    
    op_data = {"C2": {"S": 0.45}, "C3": {"S": 0.44}}
    with open(op_exp_path / "POPC_Order_Parameters.json", "w") as f:
        json.dump(op_data, f)

    # FF Experiment
    ff_exp_path = exp_path / "FormFactors" / "exp2"
    ff_exp_path.mkdir(parents=True)
    readme_ff = {
        "MOLAR_FRACTIONS": {"DOPC": 1.0},
        "TEMPERATURE": 300,
        "ION_CONCENTRATIONS": {},
        "COUNTER_IONS": [],
        "TOTAL_LIPID_CONCENTRATION": "full hydration",
        "ARTICLE_DOI": "10.1016/j.bbamem.2016.02.016",
    }
    with open(ff_exp_path / "README.yaml", "w") as f:
        yaml.dump(readme_ff, f)

    ff_data = {"q": [0.1, 0.2], "I": [1.0, 0.5]}
    with open(ff_exp_path / "system_FormFactor.json", "w") as f:
        json.dump(ff_data, f)
        
    # Set the global experiment path to our temporary directory
    original_path = FMDL_EXP_PATH
    FMDL_EXP_PATH = str(exp_path)

    yield str(exp_path)

    # Teardown: restore original path
    FMDL_EXP_PATH = original_path


def test_load_experiments(dummy_experiments):
    """Test loading of experiments."""
    from fairmd.lipids.experiment import ExperimentCollection, OPExperiment, FFExperiment
    from fairmd.lipids.api import FMDL_EXP_PATH, lipids_set
    from fairmd.lipids.core import System

    collection = ExperimentCollection.load_from_data()
    check.equal(len(collection), 2, "Should load two experiments.")

    exp1 = collection.get("exp1")
    exp2 = collection.get("exp2")

    check.is_not_none(exp1)
    check.is_not_none(exp2)

    check.is_instance(exp1, OPExperiment)
    check.is_instance(exp2, FFExperiment)

def test_op_experiment_properties(dummy_experiments):
    """Test properties of OPExperiment."""
    from fairmd.lipids.experiment import ExperimentCollection, OPExperiment, FFExperiment
    from fairmd.lipids.api import FMDL_EXP_PATH, lipids_set
    from fairmd.lipids.core import System
    collection = ExperimentCollection.load_from_data()
    exp = collection.get("exp1")

    check.equal(exp.exp_id, "exp1")
    check.equal(exp.exptype, "OrderParameters")
    check.equal(exp.molname, "POPC")
    check.equal(exp.readme["TEMPERATURE"], 303)
    check.equal(exp.data["POPC"]["C2"]["S"], 0.45)
    check.is_in("POPC", exp.get_lipids(lipids_set))

def test_ff_experiment_properties(dummy_experiments):
    """Test properties of FFExperiment."""
    from fairmd.lipids.experiment import ExperimentCollection, OPExperiment, FFExperiment
    from fairmd.lipids.api import FMDL_EXP_PATH, lipids_set
    from fairmd.lipids.core import System
    collection = ExperimentCollection[FFExperiment].load_from_data()
    exp = collection.get("exp2")

    check.equal(exp.exp_id, "exp2")
    check.equal(exp.exptype, "FormFactors")
    check.equal(exp.molname, "system")
    check.equal(exp.readme["TEMPERATURE"], 300)
    check.equal(exp.data["q"], [0.1, 0.2])
    check.is_in("DOPC", exp.get_lipids(lipids_set))

def test_collection_methods(dummy_experiments):
    """Test methods of ExperimentCollection."""
    from fairmd.lipids.experiment import ExperimentCollection, OPExperiment, FFExperiment
    from fairmd.lipids.api import FMDL_EXP_PATH, lipids_set
    from fairmd.lipids.core import System
    
    with pytest.raises(TypeError):
        ExperimentCollection.load_from_data()
    
    exp1 = collection.get("exp1")
    check.is_in(exp1, collection)
    check.is_in("exp1", collection)
    check.is_not_in("nonexistent", collection)

    check.equal(collection.loc("exp1"), exp1)
    with pytest.raises(KeyError):
        collection.loc("nonexistent")

    collection.discard("exp1")
    check.equal(len(collection), 1)
    check.is_not_in("exp1", collection)

    # Test adding back
    with pytest.raises(NotImplementedError):
        collection.add("exp1")
    
    collection.add(exp1)
    check.equal(len(collection), 2)
    check.is_in("exp1", collection)



