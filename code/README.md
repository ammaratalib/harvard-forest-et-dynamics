# Code inventory

The numbered directories follow the analysis from public tower observations to the revised manuscript results.

| Stage | Main scripts | Purpose |
|---|---|---|
| 01 | `01_prepare_ems_flux.py`, `02_prepare_hem_flux.py` | Prepare EMS and HEM source variables |
| 02 | `03_gap_fill_fluxes.R`, `04`–`07` site scripts | Gap-fill subdaily and daily flux products |
| 03 | `phenology_source_workflow.py` | Process LAI observations and growing-season timing |
| 04 | `08_calculate_conductance_current.R`, `10`–`11` site scripts | Calculate and filter flux-gradient conductance |
| 05 | `14_calculate_priestley_taylor_pet.py`, `15_forest_jarvis_penman_monteith_pet.py` | Calculate alternative and final PET estimates |
| 06 | `15_analyze_meteorology.py` | Analyze meteorological and ET trends |
| 07 | `01`–`04` XGBoost site scripts | Fit original daily and subdaily EMS/HEM models |
| 08 | `16_master_manuscript_analysis_figures.py` | Assemble original manuscript analyses and figures |
| 09 | `01`–`08` revision scripts | Prepare TEA inputs, partition ET, compare conductance, and test soil moisture |
| 10 | `01_xgboost_blocked_sensitivity.py` | Evaluate complete groups of years, HEM disturbance periods, predictor groups, GPP/NEE, and soil moisture |

Stages 01–08 preserve the original working analysis and contain some workstation-specific paths that must be replaced when rerunning them elsewhere. Stages 09–10 were prepared for the manuscript revision and use the repository structure directly.

