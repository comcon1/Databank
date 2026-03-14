.. _addingExpData:

Adding experimental data
========================

Experimental data is stored in :ref:`BilayerData/experiments <dbstructure_exp>` folder. C-H bond order parameters from NMR are in ``OrderParameters`` subfolder and X-ray scattering form factors in ``FormFactors`` subfolder.
The keys of these dictionaries are summarized in the :ref:`Experiment metadata description <readmeexp>`.

Steps to add experimental data
------------------------------

#. Fork the BilayerData repository in the GitHub interface and add+checkout the branch with an explicit
   naming, e.g., ``add-exp-Smidth-2023-dpps``. Work only in this branch. You can add several experiments into
   one branch, it's also fine.

#. Create and fill the ``README.yaml`` file of your data. This is not as simple as it seems,
   please read the :ref:`experiment metadata guidelines <addexp_guidelines>` if you are not feeling familiar.

#. Copy this README file data into a appropriate directory named as described above.

#. If you have order parameter data, create a file named ``{lipidname}_OrderParameters.json``
   where ``{lipidname}`` is the universal name of the lipid from which the data is measured
   from. The first two columns of this file should define the atom pair with universal
   atom names, third column has the experimental order parameter value, and fourth
   column has the experimental error. If the experimental error is not known, set it to 0.02.
   Store the created ``{lipidname}_OrderParameters.json`` file into the appropriate folder
   with the ``README.yaml`` file. Create ``json`` version from ``dat`` file by running

   .. code-block:: bash

        python data_to_json.py path-to-dat.dat

   in folder `experiments/OrderParameters <https://github.com/NMRLipids/BilayerData/tree/main/experiments/OrderParameters>`_. You can see previously added experiments for examples.

#. If you have X-ray scattering form factor data, store the form factor into appropriate
   folders in ASCII format where first column in x-axis values (Å\ :sup:`-1`), second
   column is y-axis value, and third is the error. Then create ``json`` file by runnig

   .. code-block:: bash

      python data_to_json.py ascii-file.txt

   in folder [experiments/FormFactors](https://github.com/NMRLipids/BilayerData/tree/main/experiments/FormFactors).
   Please see previously added experiments for examples.

#. Adding experiments can lead to recalculation of quality of some simulations and rankings.
   So, after the addition, you may want to run

   .. code-block:: bash

      fmdl_match_experiments
      fmdl_evaluate_quality
      fmdl_make_ranking

   This is not obligatory. We will do it anyway at the CI/CD stage; however, it's useful to check that
   your experiments are properly paired and quality is recalculated.

#. Submit the files to your branch and make a pull-request.

.. _addexp_guidelines:

Guidelines to fill experiment metadata
--------------------------------------

Every field of the metadata file is :ref:`explained here <readmeexp>`.
Our experience indicates that reading the original paper and filling the fields is not very straightforward,
because experimentators often use very different names to name the same things. Here, we will go throug the
most strange points.

#. **Hydration**

   We use water mass % for hydration, i.e., 10 mg in 100 ml water is ``10/(10+100)*100 (%)``. Experimentators could
   use watever - v/w %, lipid:water ratio, molar concentration. They should be converted to water mass %.

#. **Reagent sources**

   We care about reagent sources. Lipids can be bought synthetic or isolated from crude lipid extracts. They can be
   bought isolated, but be almost pure. Lipids can be bought partly deuterated and can be synthetized locally in
   the paper from synthetic lipids. Note, that the project **do not distinguish deuteration**, so whatever experimentat
   or simulation with deuterated lipid is added, we use the entity of usual lipid. However, we should mention
   deuteration in reagent sources. Lipid synthesis is usually described in the paper and it is often quite large, so
   we do not copy the whole synthesis methodology to the metadata, but we mention ``POPE: ethanolamine group is
   alpha-deuterated, synthesized according to (Vanco, 1983) from POPA (Avanti Polar Lipids)``.

   Solubles, which we explicitly add, we also mention in sources, and for water too, e.g.:

   .. code-block:: yaml

      REAGENT_SOURCES:
         TOCL: Avanti Polar Lipids
         water: Type I Milli-Q, degassed by argon bubbling
         TMACl: Sigma-Aldrich

   We do mention water as well. It can be Type I, distillated, degassed, deuterium-depletet.
   It's important for reproducibility and we appriciate a data curator mentioning it.

#. **Additional molecules**

   In the ideal world, all what we have in the sample vial, we should have also in the simulation box.
   However, this goal is not achivable. Some samples has additives, which is hard to simulate because of
   lacking force fields or because it's too diluted to add into a small simulation box. Molecules, which
   is hard to simulate but which exist in the experiment comes into the category ``ADDITIONAL_MOLECULES``.
   Typical examples are buffers, antioxidants, chelators like EDTA, etc.

.. toctree::
   :maxdepth: 1

   ../schemas/experiment_metadata.md

