# Analysis workflow

## 1. Obtain the public observations and analysis-ready tables

The public source observations are linked in `data_sources.md`. The repository supplies the derived subdaily and daily site tables used by the final manuscript analyses under `data/processed`.

## 2. Determine growing seasons and calculate conductance

Growing-season dates and compact flux-gradient conductance tables are under `data/processed/conductance`.

## 3. Calculate potential evapotranspiration

`code/05_pet/15_forest_jarvis_penman_monteith_pet.py` estimates fixed site-specific conductance parameters during rain-free, moisture-sufficient periods, removes soil-water limitation for the PET calculation, and calculates annual and seasonal PET. Derived products are stored under `outputs/forest_jarvis_pet`.

## 4. Partition ET and repeat the conductance analysis

Scripts in `code/09_et_partitioning` estimate transpiration and evaporation from the supplied standardized TEA inputs, compare ET-based and transpiration-based conductance, and test upper-soil-moisture sensitivity. Annual summary tables and the revised conductance figure are under `outputs/et_partitioning` and `outputs/soil_moisture`.

## 5. Evaluate XGBoost models

`code/10_xgboost_reviewer_analysis/01_xgboost_blocked_sensitivity.py` evaluates complete groups of years, compares a common HEM model with period-specific models, excludes related predictor groups, and tests GPP/NEE and soil moisture. Outputs are under `outputs/xgboost_reviewer_analysis`.

## 6. Review manuscript figures

Stable final and revision-specific figures are stored under `outputs/figures` and the analysis-specific output directories.

