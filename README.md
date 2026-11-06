# Harvard Forest evapotranspiration dynamics

## Project overview

This repository contains analysis-ready data, code, and derived results for the manuscript:

> **Multidecadal evapotranspiration measurements in a temperate forest demonstrate a sustained impact of warmer winters and a transient effect of biotic disturbance**

The study uses long-term eddy-covariance measurements from two Harvard Forest sites in Massachusetts, USA:

| Site | AmeriFlux ID | Forest and study role | Analysis period |
|---|---|---|---|
| EMS tower | US-Ha1 | Hardwood-dominated forest used to evaluate long-term meteorological effects | 1993–2023 |
| Hemlock tower | US-Ha2 | Eastern hemlock forest affected by hemlock woolly adelgid and subsequent deciduous replacement | 2005–2023 |

The analyses quantify long-term and seasonal evapotranspiration (ET), canopy conductance, potential evapotranspiration (PET), ET partitioning, soil-moisture sensitivity, and the predictive contributions of meteorological and ecosystem variables.

## Main analyses

```text
Public tower and meteorological observations
                    ↓
   Analysis-ready gap-filled site tables
                    ↓
      Daily ET and growing-season timing
                    ↓
 ┌──────────────────┼────────────────────┐
 ↓                  ↓                    ↓
Conductance   Jarvis–Penman–Monteith   XGBoost ET models
analysis      potential ET             and sensitivity tests
 ↓                  ↓                    ↓
ET partitioning     PET versus ET       Year-group validation,
and soil-moisture   trends              disturbance periods,
sensitivity                             and predictor groups
 └──────────────────┴────────────────────┘
                    ↓
          Manuscript figures and tables
```

The revision-specific analyses include:

- partitioning ET into transpiration and evaporation with the Transpiration Estimation Algorithm (TEA);
- comparing ET-based conductance with transpiration-based canopy stomatal conductance;
- testing conductance trends after accounting for upper-soil moisture;
- calculating PET with fixed, site-specific Jarvis–Stewart conductance parameters and soil-water limitation removed; and
- evaluating XGBoost models using complete groups of years, HEM disturbance periods, predictor-group exclusion, and soil-moisture sensitivity.

## Repository structure

```text
harvard-forest-et-dynamics/
├── code/
│   ├── 05_pet/                       # Revised PET analysis
│   ├── 09_et_partitioning/           # TEA and soil-moisture analyses
│   └── 10_xgboost_reviewer_analysis/ # Temporal validation and sensitivity tests
├── data/
│   ├── source/                       # Small source files used directly
│   └── processed/                    # Compact analysis-ready site tables
├── outputs/
│   ├── et_partitioning/
│   ├── forest_jarvis_pet/
│   ├── soil_moisture/
│   ├── xgboost_reviewer_analysis/
│   └── figures/
├── documentation/
└── environment/
```

## Included datasets

Only files directly connected to the manuscript are included. Raw AmeriFlux and Harvard Forest archive downloads are not duplicated because the authoritative records are already publicly available through the data repositories cited below. The repository includes the derived gap-filled site tables and standardized TEA inputs needed by the final analyses.

| File or directory | Description |
|---|---|
| `data/processed/daily/EMS_daily.csv` | Daily EMS meteorology, carbon fluxes, and ET used in manuscript analyses |
| `data/processed/daily/HEM_daily.csv` | Daily HEM meteorology, carbon fluxes, and ET used in manuscript analyses |
| `data/processed/flux/gaps_ML_EMS.csv` | Derived subdaily EMS table used by the final PET and XGBoost analyses |
| `data/processed/flux/gaps_ML_HEM_2023.csv` | Derived subdaily HEM table used by the final PET and XGBoost analyses |
| `data/processed/et_partitioning/` | Standardized subdaily EMS and HEM inputs used by TEA |
| `data/processed/conductance/` | Conductance estimates and growing-season dates |
| `data/processed/soil_moisture/` | Daily upper-soil-moisture data used in sensitivity analyses |
| `data/source/climate/hf300-05-daily-m.csv` | Harvard Forest daily precipitation record |
| `data/source/phenology/` | EMS and HEM field phenology/LAI source tables |
| `outputs/et_partitioning/` | Annual TEA summaries and conductance comparisons |
| `outputs/forest_jarvis_pet/` | Revised PET series, fitted parameters, and trend summaries |
| `outputs/xgboost_reviewer_analysis/` | Validation, disturbance-period, predictor-group, and soil-moisture results |

The included derived tables preserve the values used in the revised manuscript. Recreating the preprocessing steps from the original tower observations requires downloading the cited public datasets.

## Core scripts

| Analysis | Script | Purpose |
|---|---|---|
| Revised PET | `code/05_pet/15_forest_jarvis_penman_monteith_pet.py` | Calculate fixed-parameter Jarvis–Penman–Monteith PET |
| ET partitioning | `code/09_et_partitioning/02_run_nelson_tea.py` | Partition ET into transpiration and evaporation |
| Conductance comparison | `code/09_et_partitioning/04_compare_et_vs_t_conductance.py` | Compare ET-based and transpiration-based conductance |
| Soil-moisture sensitivity | `code/09_et_partitioning/07_soil_moisture_conductance_sensitivity.py` | Test whether conductance trends persist after accounting for upper-soil moisture |
| XGBoost revision analysis | `code/10_xgboost_reviewer_analysis/01_xgboost_blocked_sensitivity.py` | Evaluate complete groups of years, HEM disturbance periods, predictor groups, GPP/NEE, and soil moisture |

## Software

The analyses were run with Python 3.11 and R. Install the principal Python dependencies with:

```bash
python -m pip install -r environment/requirements.txt
```

The R workflow uses `REddyProc` together with standard data-manipulation and plotting packages. Exact versions may differ from the environment in which the original preprocessing was performed.

## Reproducing the revision analyses

Run commands from the repository root. The XGBoost script accepts an explicit repository path:

```bash
python code/10_xgboost_reviewer_analysis/01_xgboost_blocked_sensitivity.py --repo-root .
```

The TEA, conductance, and PET scripts are organized in their analysis order under `code/09_et_partitioning/` and `code/05_pet/`. See [`documentation/data_sources.md`](documentation/data_sources.md) for the public source archives and [`documentation/workflow.md`](documentation/workflow.md) for the order of analysis. Historical preprocessing scripts are not included because the authoritative raw data are public and those scripts contain workstation-specific exploratory code that is not needed to reproduce the revised analyses from the supplied analysis-ready tables.

## Data sources

- AmeriFlux US-Ha1: [https://doi.org/10.17190/AMF/1246059](https://doi.org/10.17190/AMF/1246059)
- AmeriFlux US-Ha2: [https://doi.org/10.17190/AMF/1246060](https://doi.org/10.17190/AMF/1246060)
- Harvard Forest EMS tower record (HF004): [https://doi.org/10.6073/pasta/56c6fe02a07e8a8aaff44a43a9d9a6a5](https://doi.org/10.6073/pasta/56c6fe02a07e8a8aaff44a43a9d9a6a5)
- Harvard Forest soil moisture (HF206): [https://doi.org/10.6073/pasta/43f37233c7c9ee1513d674ea2a0510c7](https://doi.org/10.6073/pasta/43f37233c7c9ee1513d674ea2a0510c7)
- Harvard Forest daily meteorology: [https://doi.org/10.6073/pasta/a7eb36231cbed30cea58b77af62945f1](https://doi.org/10.6073/pasta/a7eb36231cbed30cea58b77af62945f1)

Users of the source data should follow the citation and data-use requirements of AmeriFlux, Harvard Forest, and the Environmental Data Initiative.

## Citation

If you use this repository, please cite the associated manuscript and the original data products listed above:

> Talib, A., Munger, J. W., and Matthes, J. H. *Multidecadal evapotranspiration measurements in a temperate forest demonstrate a sustained impact of warmer winters and a transient effect of biotic disturbance*. Water Resources Research, manuscript under revision.

## Contact

**Ammara Talib**  
Yale School of the Environment, Yale University

