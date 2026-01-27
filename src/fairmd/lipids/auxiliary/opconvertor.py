"""
Auxiliary functions for converting OP data.

Helps to build a nicely formatted OP dictionary from raw OP data.
Fragmentation is handled.
"""

from fairmd.lipids.core import System
from fairmd.lipids.molecules import Lipid


def build_nice_OPdict(src: dict, lipid: Lipid) -> dict:
    """Build nicely formatted OP dictionary from raw OP data.

    Handles fragmentation of lipids.

    :param src: raw OP data (dict)
    :param lipid: Lipid object
    :return: nicely formatted OP dictionary
    """
    nice_OPdict: dict = {}
    return nice_OPdict
