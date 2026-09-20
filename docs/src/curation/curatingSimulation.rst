Simulation Curation
===================

Curation is a bit different for manually and automatically added simulations. We will consider both
cases in the following sections.

Curation of automatically added simulations
-------------------------------------------

Simulation addition is an automated process in the FAIRMD Lipids Databank. The user can submit
their simulations to Zenodo, and fill the form in the upload portal. Then the system comes as a branch
in NMRLipids/UserData repository and is pull-requested to the BilayerData repository.

First, the PR contains only a single file, called `info.yaml`, which contains the initial metadata of
the simulation. Then, a first validation starts:

**Validate Info File**: the workflow runs system addition with 50Mb restriction, so it doesn't
process the whole trajectory but fails if there are problems with the info file. The responsibility of
curator is to guide the user to fix the problems or fix them by themselves.

Typical problems are:
- improperly selected mapping file
- missing molecules in the composition
- wrong residue names

These problems are unignorable, because they will lead to problems in the processing of the simulations.

If these probelems cannot be solved, the curator should inform the contributor that problems must be
solved at the deposition stage and reject the pull request.