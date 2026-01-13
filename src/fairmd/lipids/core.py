"""
Core databank class and system initialization function.

Imported by :mod:`fairmd.lipids.databankLibrary` by default.
Can be imported without additional libraries to scan Databank system file tree!
"""

import os
import sys
import typing
from abc import ABC, abstractmethod
from collections.abc import MutableMapping, MutableSet, Sequence
from typing import Any, Generic, TypeVar

import yaml

from fairmd.lipids import FMDL_DATA_PATH, FMDL_SIMU_PATH
from fairmd.lipids.molecules import Lipid, Molecule, NonLipid, lipids_set, molecules_set


class System(MutableMapping):
    """
    Main Databank single object.

    It is an extension of a dictionary with additional functionality.
    """

    def __init__(self, data: dict | typing.Mapping) -> None:
        """
        Initialize the container for storing simulation record.

        :param data: README-type dictionary.
        :raises TypeError: If `data` is neither a `dict` nor another mapping type.
        :raises ValueError: If a molecule key in the "COMPOSITION" data does not
                            belong to the predefined set of lipids or molecules.
        """
        self._store: dict = {}
        if isinstance(data, dict):
            self._store.update(data)
        elif isinstance(data, typing.Mapping):
            self._store.update(dict(data))
        else:
            expect_type_msg = "Expected dict or Mapping"
            raise TypeError(expect_type_msg)

        self._content = {}
        for k, v in self["COMPOSITION"].items():
            mol = None
            if k in lipids_set:
                mol = Lipid(k)
            elif k in molecules_set:
                mol = NonLipid(k)
            else:
                mol_not_found_msg = f"Molecule {k} is not in the set of lipids or molecules."
                raise ValueError(mol_not_found_msg)
            mol.register_mapping(v["MAPPING"])
            self._content[k] = mol

    def __getitem__(self, key: str):  # noqa: ANN204
        return self._store[key]

    def __setitem__(self, key: str, value) -> None:
        self._store[key] = value

    def __delitem__(self, key: str) -> None:
        del self._store[key]

    def __iter__(self) -> typing.Iterator:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    @property
    def readme(self) -> dict:
        """Get the README dictionary of the system in true dict format.

        :return: dict-type README (dict)
        """
        return self._store

    @property
    def content(self) -> dict[str, Molecule]:
        """Returns dictionary of molecule objects."""
        return self._content

    @property
    def lipids(self) -> dict[str, Lipid]:
        """Returns dictionary of lipid molecule objects."""
        return {k: v for k, v in self._content.items() if k in lipids_set}

    @property
    def n_lipids(self) -> int:
        """Returns total number of lipid molecules in the system."""
        total = 0
        for k, v in self["COMPOSITION"].items():
            if k in lipids_set:
                total += sum(v["COUNT"])
        return total

    def membrane_composition(self, basis: typing.Literal["molar", "mass"] = "molar") -> dict[str, float]:
        """Return the composition of the membrane in system.

        :param which: Type of composition to return. Options are:
                      - "molar": compute molar fraction
                      - "mass": compute mass fraction
        :return: dictionary (universal molecule name -> value)
        """
        if basis not in ["molar", "mass"]:
            msg = "Basis must be 'molar' or 'mass'"
            raise ValueError(msg)
        comp: dict[str, float] = {}
        for k, v in self["COMPOSITION"].items():
            if k not in lipids_set:
                continue
            count = sum(v["COUNT"])
            comp[k] = count
        n_lipids = self.n_lipids
        if basis == "molar":
            for k in comp:
                comp[k] /= n_lipids
        else:  # (basis == "mass")
            total_mass = 0.0
            for k in comp:  # noqa: PLC0206 (modify dict while iterating)
                mol: Lipid = self._content[k]
                mw = mol.metadata["bioschema_properties"]["molecularWeight"]
                comp[k] *= mw
                total_mass += comp[k]
            for k in comp:
                comp[k] /= total_mass
        return comp

    def get_hydration(self, basis: typing.Literal["number", "mass"] = "number") -> float:
        """Get system hydration."""
        if basis not in ["number", "mass"]:
            msg = "Basis must be 'molar' or 'mass'"
            raise ValueError(msg)
        if basis == "number":
            if "SOL" not in self["COMPOSITION"]:
                msg = "Cannot compute hydration for implicit water (system #{}).".format(self["ID"])
                raise ValueError(msg)
            hyval = self["COMPOSITION"]["SOL"]["COUNT"] / self.n_lipids
        else:  # basis == "mass"
            msg = "Mass hydration is not implemented yet."
            raise NotImplementedError(msg)
        return hyval

    def __repr__(self) -> str:
        return f"System({self._store['ID']}): {self._store['path']}"


T = TypeVar("T")


class CollectionSingleton(MutableSet[T], Generic[T], ABC):
    """A generic, mutable set collection for databank items."""

    def __init__(self):
        """Initialize the Empty Collection."""
        self._items: list[T] = list()
        self._ids: set = set()

    @abstractmethod
    def _test_item_type(self, item: Any) -> bool:
        """Test if an item is of the proper type for the collection."""

    @abstractmethod
    def _get_item_id(self, item: T) -> str:
        """Get the unique identifier of an item."""

    def __contains__(self, item: Any) -> bool:
        """Check if an item is in the set by instance or by ID."""
        return (
            (self._test_item_type(item) and item in self._items)
            or (item in self._ids)
        )

    def __iter__(self):
        return iter(self._items)
    
    def __len__(self) -> int:
        return len(self._items)
    
    def __getitem__(self, index: int) -> T:
        return self._items[index]

    def add(self, item: T) -> None:
        """Add an item to the set."""
        if self._test_item_type(item):
            self._items.append(item)
            id = self._get_item_id(item)
            if isinstance(id, str):
                id = id.upper()
            if id in self._ids:
                msg = f"Item with ID '{id}' already exists in {type(self).__name__}."
                raise KeyError(msg)
            self._ids.add(id)
        else:
            msg = f"Only proper instances can be added to {type(self).__name__}."
            raise TypeError(msg)

    def discard(self, id: str|int) -> None:
        """Remove an item from the set without raising an error if it does not exist."""
        raise NotImplementedError("This method should be implemented for non-set.")
        # if isinstance(id, str):
        #     id = id.upper()
        # item_to_remove = self.get(id)
        # if item_to_remove:
        #     # self._items.discard(item_to_remove) MUST BE IMPLEMENTED FOR LIST
        #     self._ids.discard(id)

    def get(self, key: str | int, default: Any = None) -> T | None:
        """Get an item by its ID (case-insensitive)."""
        if isinstance(key, str):
            key = key.upper()
        if key in self._ids:
            for item in self._items:
                comparison_id = self._get_item_id(item)
                if isinstance(comparison_id, str):
                    comparison_id = comparison_id.upper()
                if comparison_id == key:
                    return item
        return default

    def __repr__(self) -> str:
        return f"{type(self).__name__}({sorted(list(self._ids))})"

    @property
    def ids(self) -> set:
        """The set of unique identifiers for all items in the collection."""
        return self._ids

    @staticmethod
    def load_from_data() -> "CollectionSingleton":
        """Load collection data from the designated directory."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    # TODO: schedule for removing
    def loc(self, key: str | int) -> T:
        """Locate an item by its unique identifier."""
        return self.get(key)


class SystemsCollection(CollectionSingleton[System]):
    """Immutable collection of system dicts. Can be accessed by ID using loc()."""

    def _get_item_id(self, item: System) -> int:
        return item["ID"]

    def _test_item_type(self, item: Any) -> bool:
        return isinstance(item, System)

    @staticmethod
    def load_from_data() -> "SystemsCollection":
        """Load systems data from the designated directory."""
        print("Simulations are initialized from the folder:", os.path.realpath(FMDL_SIMU_PATH))
        systems = SystemsCollection()
        for subdir, _dirs, files in os.walk(FMDL_SIMU_PATH):
            for filename in files:
                filepath = os.path.join(subdir, filename)
                if filename == "README.yaml":
                    ydict = {}
                    try:
                        with open(filepath) as yaml_file:
                            ydict.update(yaml.load(yaml_file, Loader=yaml.FullLoader))
                    except (FileNotFoundError, PermissionError) as e:
                        sys.stderr.write(f"""
!!README LOAD ERROR!!
Problems while loading on of the files required for the system: {e}
System path: {subdir}
System: {ydict!s}\n""")
                    try:
                        content = System(ydict)
                    except Exception as e:
                        sys.stderr.write(f"""
!!README LOAD ERROR!!
Unexpected error: {e}
System: {ydict!s}\n""")
                    else:
                        content["path"] = os.path.relpath(subdir, FMDL_SIMU_PATH)
                        systems.add(content)
        return systems


def initialize_databank() -> SystemsCollection:
    """
    Returns Simulation collection (an alias).

    :return: list of dictionaries that contain the content of README.yaml files for
             each system.
    """
    return SystemsCollection.load_from_data()


# TODO: is not used at all in the project!!
def print_README(system: str | typing.Mapping) -> None:  # noqa: N802
    """
    Print the content of ``system`` dictionary in human readable format.

    :param system: FAIRMD Lipids dictionary defining a simulation or "example".
    """
    if system == "example":
        current_folder = os.path.dirname(os.path.realpath(__file__))
        readme_path = os.path.join(current_folder, "SchemaValidation", "Schema", "READMEexplanations.yaml")
        with open(readme_path) as file:
            readme_file = yaml.safe_load(file)
    else:
        readme_file = system

    for key in readme_file:
        print("\033[1m" + key + ":" + "\033[0m")
        print(" ", readme_file[key])
