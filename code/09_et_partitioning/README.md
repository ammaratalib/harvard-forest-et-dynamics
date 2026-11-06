# ET partitioning revision analysis

This directory contains the reproducible reviewer-response analysis. Original
manuscript products are retained; revision outputs are written under
`outputs/et_partitioning`.

Workflow:

1. The standardized manuscript-consistent subdaily inputs are supplied under
   `data/processed/et_partitioning`.
2. `02_run_nelson_tea.py`: partition ET with the Nelson et al. (2018) TEA method.
3. `03_plot_tea_partition.py`: summarize growing- and nongrowing-season T/ET.
4. `04_compare_et_vs_t_conductance.py`: calculate paired flux-gradient
   conductance from total ET and TEA transpiration and compare annual trends.
5. `05_make_figure3.py`: regenerate final Figure 3 while changing only the
   erroneous HEM latent-heat values; EMS remains unchanged.
6. `07_soil_moisture_conductance_sensitivity.py`: test whether conductance
   trends persist after accounting for upper-soil moisture.

The conductance comparison uses one common screened sample, the same energy-
closure factor for the paired calculations, and physically consistent latent
heat in J kg-1. It does not overwrite the original manuscript conductance files.

## Audited method details

`02_run_nelson_tea.py` follows the released Nelson et al. (2018) TEA predictor
definitions and flags. Direct regression tests against the authors' source code
give zero numerical difference for potential radiation and CSWI, and differences
below `2e-15` for DWCI and the diurnal-centroid predictor. EMS is hourly
(1992--2023) and HEM is half-hourly (2006--2022); WUE conversion uses the actual
timestep at each site. Missing random-forest predictors are excluded rather than
filled within the TEA script. Predicted evaporation is retained as `E = ET - T`,
including negative subdaily residuals, as in the released algorithm.

The available precipitation product is daily rather than subdaily. Each daily
total is therefore applied as one conservative pulse at the first timestep of
the day for the CSWI calculation. This approximation and the fraction of
negative subdaily E should be disclosed if the partitioning is reported.

`05_make_figure3.py` includes regression guards that reproduce the
archived HEM conductance before changing lambda, enforce a physically plausible
latent-heat range, and use the original Mann-Kendall test with tie and continuity
corrections. The archived HEM lambda was approximately 3.149 MJ kg-1; the
corrected median is approximately 2.454 MJ kg-1. EMS is read unchanged from its
final processed Figure 3 table.

