"""
:module: experiment.py

:description: Module for handling experimental data entries in the databank.
"""
import json
import os
from abc import ABC, abstractmethod
from typing import Any

import yaml

from fairmd.lipids.core import Collection
from fairmd.lipids.api import FMDL_EXP_PATH, lipids_set


class Experiment(ABC):
    """
    Abstract base class representing an experimental dataset in the databank.
    """

    _exp_id: str
    _metadata: dict | None = None
    _data: dict | None = None

    def __init__(self, exp_id: str, path: str):
        """
        Initialize the Experiment object.

        :param exp_id: The unique identifier for the experiment.
        :param path: The absolute path to the experiment's directory.
        """
        self._exp_id = exp_id
        self._path = path
        self._populate_meta_data()

    def _get_path(self) -> str:
        """
        Return the absolute path to the experiment's directory.
        """
        return self._path

    def _populate_meta_data(self) -> None:
        """
        Populate metadata from the README.yaml file.
        """
        self._metadata = {}
        meta_path = os.path.join(self._get_path(), "README.yaml")
        if os.path.isfile(meta_path):
            with open(meta_path) as yaml_file:
                self._metadata = yaml.load(yaml_file, Loader=yaml.FullLoader)
        else:
            msg = f"Metadata file (README.yaml) not found for experiment '{self._exp_id}'."
            raise FileNotFoundError(msg)

    @property
    def metadata(self) -> dict:
        """Access the experiment's metadata."""
        if self._metadata is None:
            self._populate_meta_data()
        return self._metadata

    @property
    def readme(self) -> dict:
        """Provides access to the experiment's metadata (for backward compatibility)."""
        return self.metadata

    @property
    def exp_id(self) -> str:
        """The unique identifier of the experiment."""
        return self._exp_id

    @property
    def path(self) -> str:
        """The absolute path to the experiment's directory."""
        return self._path

    @property
    def dataPath(self) -> str:
        """The absolute path to the experiment's directory (for backward compatibility)."""
        return self._path

    @property
    @abstractmethod
    def data(self) -> dict:
        """Provide access to the experiment's data."""

    @property
    @abstractmethod
    def molname(self) -> str:
        """The molecule name associated with the experiment data."""

    @property
    @abstractmethod
    def exptype(self) -> str:
        """The type of the experiment."""

    def get_lipids(self, molecules=lipids_set) -> list[str]:
        """Get lipids from molar fractions."""
        return [k for k in self.metadata.get("MOLAR_FRACTIONS", {}) if k in molecules]

    def get_ions(self, ions: list[str]) -> list[str]:
        """Get ions from ion concentrations and counter ions."""
        exp_ions: list[str] = []
        for key in ions:
            if self.metadata.get("ION_CONCENTRATIONS", {}).get(key, 0) != 0:
                exp_ions.append(key)
            if key in self.metadata.get("COUNTER_IONS", []):
                exp_ions.append(key)
        return list(set(exp_ions))

    def __getitem__(self, key: str):
        return self.metadata[key]

    def __repr__(self):
        return f"{type(self).__name__}(id='{self.exp_id}')"

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.exp_id == other.exp_id

    def __hash__(self):
        return hash(self.exp_id)


class OPExperiment(Experiment):
    """Represents an order parameter experiment."""

    @property
    def data(self) -> dict:
        if self._data is None:
            self._data = {}
            for fname in os.listdir(self._get_path()):
                if fname.endswith("_Order_Parameters.json"):
                    molecule_name = fname.replace("_Order_Parameters.json", "")
                    fpath = os.path.join(self._get_path(), fname)
                    with open(fpath) as json_file:
                        self._data[molecule_name] = json.load(json_file)
        return self._data

    @property
    def molname(self) -> str:
        """The molecule name is derived from the first found data file."""
        for fname in os.listdir(self._get_path()):
            if fname.endswith("_Order_Parameters.json"):
                return fname.replace("_Order_Parameters.json", "")
        return ""

    @property
    def exptype(self) -> str:
        return "OrderParameters"


class FFExperiment(Experiment):
    """Represents a form factor experiment."""

    @property
    def data(self) -> dict:
        if self._data is None:
            self._data = {}
            for fname in os.listdir(self._get_path()):
                if fname.endswith("_FormFactor.json"):
                    fpath = os.path.join(self._get_path(), fname)
                    with open(fpath) as json_file:
                        self._data = json.load(json_file)
                    break  # Assuming one form factor file per experiment
        return self._data

    @property
    def molname(self) -> str:
        return "system"

    @property
    def exptype(self) -> str:
        return "FormFactors"


class ExperimentCollection(Collection[Experiment]):
    """A collection of experiments."""

    def _test_item_type(self, item: Any) -> bool:
        return isinstance(item, Experiment)

    def _create_item(self, name: str) -> Experiment:
        raise NotImplementedError("Cannot create an experiment by name. Use load_from_data.")

    def _get_item_id(self, item: Experiment) -> str:
        return item.exp_id

    @staticmethod
    def load_from_data() -> "ExperimentCollection":
        """Load experiment data from the designated directory."""
        collection = ExperimentCollection()
        exp_types = {
            "OrderParameters": OPExperiment,
            "FormFactors": FFExperiment,
        }

        for exp_type, exp_cls in exp_types.items():
            path = os.path.join(FMDL_EXP_PATH, exp_type)
            if not os.path.isdir(path):
                continue
            for subdir, _, files in os.walk(path):
                if "README.yaml" in files:
                    # exp_id is the directory name relative to the exp_type directory
                    exp_id = os.path.relpath(subdir, path)
                    try:
                        exp = exp_cls(exp_id, subdir)
                        collection.add(exp)
                    except FileNotFoundError:
                        # Log this error?
                        pass
        return collection

    def loc(self, exp_id: str) -> Experiment:
        """Locate an experiment by its path-ID."""
        exp = self.get(exp_id)
        if exp:
            return exp
        msg = f"Experiment with path-ID '{exp_id}' not found in the collection."
        raise KeyError(msg)
