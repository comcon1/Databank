.. _plottings:

Plotting functions
==================

The functions that help visualizing data in the FAIRMD Lipids are described here. These
functions are located in the :mod:`fairmd.lipids.ipylib` submodule. Currently, the module
offers a function to plot simulation against experimental data for form factor and order
parameter.

Plotting form factors
---------------------

.. code-block:: bash

   from fairmd.lipids.core import initialize_databank
   from fairmd.lipids.ipylib import plot_simulation_FF

   ss = initialize_databank()
   f = plot_simulation_FF(ss.loc(914))
   f.savefig("form_factor_914.png", dpi=300)


:func:`fairmd.lipids.ipylib.plot_simulation_FF` will produce a figure with the simulated and
experimental form factors together. Experimental data is automatically rescaled before plotting by
the scaling factor computed by function
:func:`fairmd.lipids.analib.formfactor.calc_ff_scaling_distance`. Currently, the function plots only
first experiment if there are multiple experiments associated.

.. image:: _static/images/914ff.png
   :alt: Simulated and experimental form factors for databank entry 914
   :align: center
