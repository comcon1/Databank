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

from fairmd.lipids import FMDL_SIMU_PATH
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


class SystemsCollection(Sequence[System]):
    """Immutable collection of system dicts. Can be accessed by ID using loc()."""

    def __init__(self, iterable: typing.Sequence[System] = []) -> None:
        self._data = iterable
        self.__get_index_byid()

    def __get_index_byid(self) -> None:
        self._idx: dict = {}
        for i in range(len(self)):
            if "ID" in self[i]:
                self._idx[self[i]["ID"]] = i

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self) -> int:
        return len(self._data)

    def loc(self, sid: int) -> System:
        """Locate system by its ID.

        :param sid: System ID
        :return: System object with ID `sid`
        """
        return self._data[self._idx[sid]]


class Databank:
    """
    Representation of all simulation in the NMR lipids databank.

    `path` should be the local location of `{FMDL_DATA_PATH}/Simulations/` in
    the FAIRMD Lipids folder. Example usage to loop over systems:

    .. code-block:: python

        path = 'BilayerData/Simulations/'
        db_data = databank(path)
        systems = db_data.get_systems()

        for system in systems:
            print(system)
    """

    def __init__(self) -> None:
        self.path = FMDL_SIMU_PATH
        __systems = self.__load_systems__()
        self._systems: SystemsCollection = SystemsCollection(__systems)
        print("Databank initialized from the folder:", os.path.realpath(self.path))

    def __load_systems__(self) -> list[System]:
        systems: list[System] = []
        rpath = os.path.realpath(self.path)
        for subdir, _dirs, files in os.walk(rpath):
            for filename in files:
                filepath = os.path.join(subdir, filename)
                if filename == "README.yaml":
                    ydict = {}
                    try:
                        with open(filepath) as yaml_file:
                            ydict.update(yaml.load(yaml_file, Loader=yaml.FullLoader))
                        content = System(ydict)
                    except (FileNotFoundError, PermissionError) as e:
                        sys.stderr.write(f"""
!!README LOAD ERROR!!
Problems while loading on of the files required for the system: {e}
System path: {subdir}
System: {ydict!s}\n""")
                    except Exception as e:
                        sys.stderr.write(f"""
!!README LOAD ERROR!!
Unexpected error: {e}
System: {ydict!s}\n""")
                    else:
                        relpath = os.path.relpath(filepath, rpath)
                        content["path"] = relpath[:-11]
                        systems.append(content)
        return systems

    def get_systems(self) -> SystemsCollection:
        """List all systems in the FAIRMD Lipids."""
        return self._systems


def initialize_databank() -> SystemsCollection:
    """
    Intialize the FAIRMD Lipids.

    :return: list of dictionaries that contain the content of README.yaml files for
             each system.
    """
    db_data = Databank()
    return db_data.get_systems()


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


T = TypeVar("T")


class Collection(MutableSet[T], Generic[T], ABC):
    """A generic, mutable set collection for databank items."""

    def __init__(self, *args: T):
        """Initialize the Collection with optional initial elements."""
        self._items: set[T] = set()
        self._ids: set[str] = set()
        for arg in args:
            self.add(arg)

    @abstractmethod
    def _test_item_type(self, item: Any) -> bool:
        """Test if an item is of the proper type for the collection."""

    @abstractmethod
    def _create_item(self, name: str) -> T:
        """Construct an item of the proper type from a name."""

    @abstractmethod
    def _get_item_id(self, item: T) -> str:
        """Get the unique identifier of an item."""

    def __contains__(self, item: Any) -> bool:
        """Check if an item is in the set by instance or by ID."""
        return (self._test_item_type(item) and item in self._items) or (
            isinstance(item, str) and item.upper() in self._ids
        )

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, item: T | str) -> None:
        """
        Add an item to the set.

        If the item is a string, `_create_item` is used to construct the object.
        """
        if self._test_item_type(item):
            self._items.add(item)
            self._ids.add(self._get_item_id(item).upper())
        elif isinstance(item, str):
            new_item = self._create_item(item)
            self._items.add(new_item)
            self._ids.add(self._get_item_id(new_item).upper())
        else:
            msg = f"Only proper instances or strings can be added to {type(self).__name__}."
            raise TypeError(msg)

    def discard(self, item: T | str) -> None:
        """Remove an item from the set without raising an error if it does not exist."""
        item_id_to_discard = None
        if self._test_item_type(item):
            item_id_to_discard = self._get_item_id(item).upper()
        elif isinstance(item, str):
            item_id_to_discard = item.upper()

        if item_id_to_discard is None or item_id_to_discard not in self._ids:
            return

        item_to_remove = self.get(item_id_to_discard)
        if item_to_remove:
            self._items.discard(item_to_remove)
            self._ids.discard(item_id_to_discard)


    def get(self, key: str, default: Any = None) -> T | None:
        """Get an item by its ID (case-insensitive)."""
        key_upper = key.upper()
        if key_upper in self._ids:
            for item in self._items:
                if self._get_item_id(item).upper() == key_upper:
                    return item
        return default

    def __repr__(self) -> str:
        return f"{type(self).__name__}({sorted(list(self._ids))})"

    @property
    def ids(self) -> set[str]:
        """The set of unique identifiers for all items in the collection."""
        return self._ids
