"""
Test Order Parameter handling functionality.

Currently focused on testing opconvertor realted code because it is important for
OP data visualization.

Test data is stored in `./ToyData/Simulations.2`

-------------------------------------------------------------------------------
NOTE: globally import of fairmd-lipids is **STRICTLY FORBIDDEN** because it
      breaks the substitution of global path folders
"""

import copy
import os
import sys
import warnings
import numpy as np
import pytest
import pytest_check as check

# run only on sim2 mocking data
pytestmark = [pytest.mark.sim2, pytest.mark.min]


class TestBuildNiceOPdict:
    @pytest.fixture
    def systems(self):
        from fairmd.lipids.core import initialize_databank

        return initialize_databank()

    def test_build_nice_OPdict(self, systems):
        from fairmd.lipids.auxiliary.opconvertor import build_nice_OPdict
        from fairmd.lipids.api import get_OP

        sys = systems.loc(281)
        opdata = get_OP(sys)
        rdict = build_nice_OPdict(opdata["POPC"], sys.lipids["POPC"])
        assert isinstance(rdict, dict)  # dict expected
        assert "sn-1" in rdict
        assert "sn-2" in rdict  # fragments at the top level
