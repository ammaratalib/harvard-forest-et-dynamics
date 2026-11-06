#!/usr/bin/env python3
"""Reviewer-focused validation and sensitivity analysis for the Harvard Forest ET models.

This script does not replace or modify the original gap-filling models. It uses the
processed daily data, the saved optimized settings in the original daily XGBoost
scripts, and temporally blocked validation to evaluate:

1. Baseline predictive performance with complete multi-year periods withheld.
2. HEM performance before, during, and after HWA-related disturbance.
3. Common HEM models versus models trained within each vegetation period.
4. Sensitivity to removal of correlated predictor groups and carbon variables.
5. Sensitivity to adding measured upper-soil moisture over the available overlap.

Outputs are written to outputs/xgboost_reviewer_analysis unless --output-dir is set.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


MODEL_PARAMETERS = {
    # Saved optimized settings in the original daily-model scripts. The
    # optimization calls are commented out there because tuning was performed
    # once on the chronological training portion and the selected values were
    # then retained.
    "EMS": {
        "objective": "reg:squarederror",
        "booster": "gbtree",
        "n_estimators": 195,
        "max_depth": 12,
        "learning_rate": 0.5506300797492677,
        "min_child_weight": 0.20645034912406007,
        "subsample": 0.5088196895966189,
        "colsample_bytree": 0.5594401417766359,
        "reg_alpha": 24.023171492098253,
        "reg_lambda": 13.270921107926412,
        "tree_method": "hist",
        "random_state": 0,
        "n_jobs": -1,
        "verbosity": 0,
    },
    "HEM": {
        "objective": "reg:squarederror",
        "booster": "gbtree",
        "n_estimators": 373,
        "max_depth": 14,
        "learning_rate": 0.98,
        "min_child_weight": 2.43,
        "subsample": 0.705,
        "colsample_bytree": 0.33,
        "reg_alpha": 16.28,
        "reg_lambda": 77.27,
        "tree_method": "hist",
        "random_state": 0,
        "n_jobs": -1,
        "verbosity": 0,
    },
}


FEATURE_GROUPS = {
    "Radiation and energy": ["NETRAD", "PAR", "Rg", "H"],
    "Carbon and canopy activity": ["GPP", "NEE"],
    "Atmospheric conditions": ["VPD", "Tair", "PA"],
    "Turbulence": ["WS", "WD", "Ustar"],
    "Seasonal and temporal": ["Year", "DoY", "Month_cos", "sos", "eos"],
    "Soil temperature": ["SoilTemp"],
}

BASE_FEATURES = [feature for features in FEATURE_GROUPS.values() for feature in features]

DISPLAY_GROUP_ORDER = [
    "Radiation and energy",
    "Carbon and canopy activity",
    "Atmospheric conditions",
    "Turbulence",
    "Seasonal and temporal",
    "Soil temperature",
]

GROUP_DISPLAY_LABELS = {
    "Radiation and energy": "Radiation and energy fluxes",
    "Carbon and canopy activity": "Carbon exchange",
    "Atmospheric conditions": "Atmospheric conditions",
    "Turbulence": "Turbulence",
    "Seasonal and temporal": "Time and phenology",
    "Soil temperature": "Soil temperature",
}

SITE_COLORS = {"EMS": "#ef7f78", "HEM": "#2b8a8f"}
PERIOD_LABELS = {
    "all": "All years",
    "pre": "Pre-disturbance",
    "transition": "Transition",
    "post": "Post-disturbance",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Path to harvard-forest-et-dynamics. Auto-detected when script is in the repo.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory.",
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=10,
        help="Grouped-permutation repeats for the soil-moisture sensitivity models.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Rebuild Figure R2 from the saved selected-model sensitivity table.",
    )
    return parser.parse_args()


def detect_repo_root(explicit: Path | None) -> Path:
    if explicit is not None:
        root = explicit.expanduser().resolve()
        if not (root / "data" / "processed").exists():
            raise FileNotFoundError(f"Processed-data directory not found under {root}")
        return root

    script = Path(globals().get("__file__", Path.cwd())).resolve()
    for candidate in [script.parent, *script.parents]:
        if (candidate / "data" / "processed").exists():
            return candidate
    cwd = Path.cwd().resolve()
    if (cwd / "data" / "processed").exists():
        return cwd
    raise FileNotFoundError("Could not auto-detect repository root; provide --repo-root.")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def load_measured_daily_et(repo_root: Path, site: str) -> pd.DataFrame:
    """Build daily ET only from measured subdaily latent heat flux."""
    flux_dir = repo_root / "data" / "processed" / "flux"
    filename = "gaps_ML_EMS.csv" if site == "EMS" else "gaps_ML_HEM_2023.csv"
    flux = pd.read_csv(
        flux_dir / filename,
        usecols=["TIMESTAMP", "LE", "Tair"],
        low_memory=False,
    )
    flux["timestamp"] = pd.to_datetime(flux["TIMESTAMP"], errors="coerce")
    flux["LE_measured"] = numeric(flux["LE"])
    flux["Tair_C"] = numeric(flux["Tair"])
    interval_seconds = 3600.0 if site == "EMS" else 1800.0
    expected_intervals = int(round(86400.0 / interval_seconds))

    # Equivalent to 3,149,000 - 2,370 T(K), written directly in degrees C.
    latent_heat = 2_501_000.0 - 2_370.0 * flux["Tair_C"]
    valid = (
        flux["timestamp"].notna()
        & flux["LE_measured"].notna()
        & latent_heat.between(2.2e6, 2.7e6)
    )
    flux["date"] = flux["timestamp"].dt.normalize()
    flux["ET_interval"] = np.where(
        valid,
        flux["LE_measured"] * interval_seconds / latent_heat,
        np.nan,
    )
    daily = (
        flux.groupby("date", as_index=False)
        .agg(
            ET_measured=("ET_interval", lambda values: values.sum(min_count=1)),
            measured_intervals=("ET_interval", "count"),
        )
        .sort_values("date")
    )
    daily["measured_fraction"] = daily["measured_intervals"] / expected_intervals
    # Require at least 75% measured subdaily coverage so the held-out target is
    # not itself dominated by temporal gaps.
    return daily.loc[daily["measured_fraction"] >= 0.75].copy()


def load_site_daily(repo_root: Path, site: str) -> pd.DataFrame:
    path = repo_root / "data" / "processed" / "daily" / f"{site}_daily.csv"
    raw = pd.read_csv(path, low_memory=False)
    raw["date"] = pd.to_datetime(raw["TIMESTAMP"], errors="coerce").dt.normalize()

    if site == "EMS":
        mapping = {
            "ET": "ET",
            "Year": "Year",
            "DoY": "DoY",
            "NETRAD": "NETRAD_fill",
            "PAR": "PAR_fill",
            "Rg": "Rg_fill",
            "H": "H_Rd",
            "GPP": "GPP_g",
            "NEE": "nee.e.6mol.m2.s",
            "VPD": "VPD_fill",
            "Tair": "Tair_fill",
            "PA": "PA_Rd",
            "WS": "WS_Rd",
            "WD": "WD_gaps",
            "Ustar": "Ustar_gaps",
            "Month_cos": "Month_cos",
            "sos": "sos",
            "eos": "eos",
            "SoilTemp": "tsoil_upper_mean_gaps",
        }
        start, end = "1993-01-01", "2023-12-31"
    elif site == "HEM":
        mapping = {
            "ET": "ET",
            "Year": "Year",
            "DoY": "DoY",
            "NETRAD": "NETRAD",
            "PAR": "PPFD",
            "Rg": "Rg",
            "H": "H",
            "GPP": "GPP_g",
            "NEE": "NEE",
            "VPD": "VPD",
            "Tair": "Tair",
            "PA": "PA",
            "WS": "WS",
            "WD": "WD",
            "Ustar": "Ustar",
            "Month_cos": "Month_cos",
            "sos": "sos",
            "eos": "eos",
            "SoilTemp": "TS_PI_1",
        }
        # The processed HEM carbon-flux record has no valid GPP during 2023.
        start, end = "2005-01-01", "2022-12-31"
    else:
        raise ValueError(site)

    missing = [source for source in mapping.values() if source not in raw.columns]
    if missing:
        raise KeyError(f"{site}: missing required source columns: {missing}")

    out = pd.DataFrame({"date": raw["date"]})
    for canonical, source in mapping.items():
        out[canonical] = numeric(raw[source])
    out["site"] = site
    out = out.loc[out["date"].between(start, end)].sort_values("date").reset_index(drop=True)
    measured_et = load_measured_daily_et(repo_root, site)
    out = out.drop(columns="ET").merge(measured_et, on="date", how="inner")
    out = out.rename(columns={"ET_measured": "ET"})
    out = out.loc[out["ET"].notna()].sort_values("date").reset_index(drop=True)
    out["Year"] = out["date"].dt.year.astype(float)
    out["period"] = "all"
    if site == "HEM":
        year = out["date"].dt.year
        out.loc[year.between(2005, 2010), "period"] = "pre"
        out.loc[year.between(2011, 2014), "period"] = "transition"
        out.loc[year.between(2015, 2022), "period"] = "post"
    return out


def build_folds(data: pd.DataFrame, site: str, soil_overlap: bool = False) -> list[dict]:
    years = np.array(sorted(data["date"].dt.year.unique()))
    if site == "HEM":
        definitions = [
            ("pre", np.array([y for y in years if 2005 <= y <= 2010])),
            ("transition", np.array([y for y in years if 2011 <= y <= 2014])),
            ("post", np.array([y for y in years if 2015 <= y <= 2023])),
        ]
        folds = [{"label": label, "years": ys} for label, ys in definitions if len(ys)]
    else:
        n_folds = 4 if soil_overlap else 5
        folds = [
            {"label": f"block_{i + 1}", "years": block}
            for i, block in enumerate(np.array_split(years, min(n_folds, len(years))))
            if len(block)
        ]
    covered = sorted({int(y) for fold in folds for y in fold["years"]})
    if covered != sorted(int(y) for y in years):
        raise RuntimeError(f"{site}: fold definitions do not cover every analysis year")
    return folds


def matrix(data: pd.DataFrame, features: list[str]) -> np.ndarray:
    return data[features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)


def fit_blocked_predictions(
    data: pd.DataFrame,
    features: list[str],
    folds: list[dict],
    parameters: dict,
) -> pd.DataFrame:
    prediction = pd.Series(np.nan, index=data.index, dtype=float)
    fold_name = pd.Series("", index=data.index, dtype=object)
    year = data["date"].dt.year

    for fold in folds:
        test_mask = year.isin(fold["years"])
        train_mask = ~test_mask
        if train_mask.sum() < 100 or test_mask.sum() < 30:
            raise RuntimeError(
                f"Insufficient rows for fold {fold['label']}: "
                f"train={train_mask.sum()}, test={test_mask.sum()}"
            )
        model = XGBRegressor(**parameters)
        model.fit(matrix(data.loc[train_mask], features), data.loc[train_mask, "ET"].to_numpy())
        prediction.loc[test_mask] = model.predict(matrix(data.loc[test_mask], features))
        fold_name.loc[test_mask] = fold["label"]

    if prediction.isna().any():
        raise RuntimeError("Blocked validation did not generate predictions for every row")
    return pd.DataFrame(
        {
            "date": data["date"],
            "site": data["site"],
            "period": data["period"],
            "fold": fold_name,
            "observed_ET": data["ET"],
            "predicted_ET": prediction,
        }
    )


def metric_record(observed: Iterable[float], predicted: Iterable[float]) -> dict:
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    valid = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[valid], pred[valid]
    if len(obs) < 3:
        return {key: np.nan for key in ["n", "r2", "nse", "mae", "rmse", "pbias"]}
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    nse = 1.0 - np.sum((pred - obs) ** 2) / denominator if denominator > 0 else np.nan
    if np.std(obs) > 0 and np.std(pred) > 0:
        r2 = float(np.corrcoef(obs, pred)[0, 1] ** 2)
    else:
        r2 = np.nan
    total = np.sum(obs)
    pbias = 100.0 * np.sum(pred - obs) / total if total != 0 else np.nan
    return {
        "n": int(len(obs)),
        "r2": float(r2),
        "nse": float(nse),
        "mae": float(mean_absolute_error(obs, pred)),
        "rmse": float(math.sqrt(mean_squared_error(obs, pred))),
        "pbias": float(pbias),
    }


def summarize_predictions(predictions: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows = []
    site = str(predictions["site"].iloc[0])
    overall = metric_record(predictions["observed_ET"], predictions["predicted_ET"])
    rows.append({"site": site, "period": "all", "variant": variant, **overall})
    if site == "HEM":
        for period in ["pre", "transition", "post"]:
            subset = predictions.loc[predictions["period"] == period]
            rows.append(
                {
                    "site": site,
                    "period": period,
                    "variant": variant,
                    **metric_record(subset["observed_ET"], subset["predicted_ET"]),
                }
            )
    return pd.DataFrame(rows)


def run_full_record_models(data_by_site: dict[str, pd.DataFrame], output_dir: Path) -> tuple:
    prediction_tables = []
    performance_tables = []
    variant_performance = []

    for site, data in data_by_site.items():
        folds = build_folds(data, site)
        parameters = MODEL_PARAMETERS[site]
        baseline = fit_blocked_predictions(data, BASE_FEATURES, folds, parameters)
        prediction_tables.append(baseline)
        baseline_summary = summarize_predictions(baseline, "baseline")
        performance_tables.append(baseline_summary)
        variant_performance.append(baseline_summary)

        variants = {
            f"drop::{group}": [f for f in BASE_FEATURES if f not in features]
            for group, features in FEATURE_GROUPS.items()
        }
        variants["drop::GPP"] = [f for f in BASE_FEATURES if f != "GPP"]
        variants["drop::NEE"] = [f for f in BASE_FEATURES if f != "NEE"]

        for name, features in variants.items():
            predictions = fit_blocked_predictions(data, features, folds, parameters)
            variant_performance.append(summarize_predictions(predictions, name))

    predictions = pd.concat(prediction_tables, ignore_index=True)
    baseline_performance = pd.concat(performance_tables, ignore_index=True)
    variants = pd.concat(variant_performance, ignore_index=True)
    predictions.to_csv(output_dir / "blocked_predictions.csv", index=False)
    baseline_performance.to_csv(output_dir / "blocked_performance.csv", index=False)
    variants.to_csv(output_dir / "group_ablation_performance.csv", index=False)

    baseline = variants.loc[variants["variant"] == "baseline"].set_index(["site", "period"])
    rows = []
    for _, row in variants.loc[variants["variant"].str.startswith("drop::")].iterrows():
        base = baseline.loc[(row["site"], row["period"])]
        rows.append(
            {
                "site": row["site"],
                "period": row["period"],
                "removed": row["variant"].replace("drop::", ""),
                "delta_rmse": row["rmse"] - base["rmse"],
                "delta_mae": row["mae"] - base["mae"],
                "delta_nse": base["nse"] - row["nse"],
                "baseline_rmse": base["rmse"],
                "ablated_rmse": row["rmse"],
            }
        )
    importance = pd.DataFrame(rows)
    importance.to_csv(output_dir / "group_ablation_importance.csv", index=False)
    return predictions, baseline_performance, variants, importance


def build_within_period_folds(data: pd.DataFrame, period: str) -> list[dict]:
    """Hold out complete years while training only within one HEM vegetation period."""
    years = np.array(sorted(data["date"].dt.year.unique()))
    if period in {"pre", "transition"}:
        blocks = [np.array([year]) for year in years]
    else:
        blocks = [block for block in np.array_split(years, min(4, len(years))) if len(block)]
    return [
        {"label": f"{period}_within_{i + 1}", "years": block}
        for i, block in enumerate(blocks)
    ]


def run_hem_state_specific_models(
    hem_data: pd.DataFrame,
    common_predictions: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare cross-state transfer with blocked validation within each HEM state."""
    predictions = []
    comparison = []
    parameters = MODEL_PARAMETERS["HEM"]
    common_hem = common_predictions.loc[common_predictions["site"] == "HEM"].copy()

    for period in ["pre", "transition", "post"]:
        subset = hem_data.loc[hem_data["period"] == period].copy().reset_index(drop=True)
        folds = build_within_period_folds(subset, period)
        within = fit_blocked_predictions(subset, BASE_FEATURES, folds, parameters)
        within["strategy"] = "within_period_blocked"
        predictions.append(within)

        common_subset = common_hem.loc[common_hem["period"] == period]
        for strategy, table in [
            ("common_model_period_transfer", common_subset),
            ("within_period_blocked", within),
        ]:
            comparison.append(
                {
                    "site": "HEM",
                    "period": period,
                    "strategy": strategy,
                    **metric_record(table["observed_ET"], table["predicted_ET"]),
                }
            )

    prediction_df = pd.concat(predictions, ignore_index=True)
    comparison_df = pd.DataFrame(comparison)
    prediction_df.to_csv(output_dir / "hem_state_specific_predictions.csv", index=False)
    comparison_df.to_csv(output_dir / "hem_model_strategy_comparison.csv", index=False)
    return prediction_df, comparison_df


def run_hem_within_period_sensitivities(
    hem_data: pd.DataFrame,
    within_predictions: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run predictor removal with the HEM strategy selected by model comparison."""
    performance_rows = []
    parameters = MODEL_PARAMETERS["HEM"]
    variants = {
        f"drop::{group}": [f for f in BASE_FEATURES if f not in features]
        for group, features in FEATURE_GROUPS.items()
    }
    variants["drop::GPP"] = [f for f in BASE_FEATURES if f != "GPP"]
    variants["drop::NEE"] = [f for f in BASE_FEATURES if f != "NEE"]

    for period in ["pre", "transition", "post"]:
        subset = hem_data.loc[hem_data["period"] == period].copy().reset_index(drop=True)
        folds = build_within_period_folds(subset, period)
        baseline = within_predictions.loc[within_predictions["period"] == period]
        performance_rows.append(
            {
                "site": "HEM",
                "period": period,
                "variant": "baseline",
                **metric_record(baseline["observed_ET"], baseline["predicted_ET"]),
            }
        )
        for name, features in variants.items():
            predicted = fit_blocked_predictions(subset, features, folds, parameters)
            performance_rows.append(
                {
                    "site": "HEM",
                    "period": period,
                    "variant": name,
                    **metric_record(predicted["observed_ET"], predicted["predicted_ET"]),
                }
            )

    performance = pd.DataFrame(performance_rows)
    baseline = performance.loc[performance["variant"] == "baseline"].set_index("period")
    importance_rows = []
    for _, row in performance.loc[performance["variant"].str.startswith("drop::")].iterrows():
        base = baseline.loc[row["period"]]
        importance_rows.append(
            {
                "site": "HEM",
                "period": row["period"],
                "removed": row["variant"].replace("drop::", ""),
                "delta_rmse": row["rmse"] - base["rmse"],
                "delta_mae": row["mae"] - base["mae"],
                "delta_nse": base["nse"] - row["nse"],
                "baseline_rmse": base["rmse"],
                "ablated_rmse": row["rmse"],
            }
        )
    importance = pd.DataFrame(importance_rows)
    performance.to_csv(output_dir / "hem_within_period_ablation_performance.csv", index=False)
    importance.to_csv(output_dir / "hem_within_period_ablation_importance.csv", index=False)
    return performance, importance


def standardized_soil_index(repo_root: Path, site: str) -> pd.DataFrame:
    soil_dir = repo_root / "data" / "processed" / "soil_moisture"
    path = soil_dir / f"{site}_daily_soil_moisture.csv"
    soil = pd.read_csv(path, low_memory=False)
    soil["date"] = pd.to_datetime(soil["date"], errors="coerce").dt.normalize()
    if site == "EMS":
        sensor_cols = [
            "profile_a_15cm_m3_m3",
            "profile_b_15cm_m3_m3",
            "hf206_upper15_sensor1_m3_m3",
            "hf206_upper15_sensor2_m3_m3",
            "hf206_upper15_sensor3_m3_m3",
            "hf206_upper15_sensor4_m3_m3",
        ]
    else:
        sensor_cols = [
            "upper15_sensor1_m3_m3",
            "upper15_sensor2_m3_m3",
            "upper15_sensor3_m3_m3",
            "upper15_sensor4_m3_m3",
        ]

    standardized = pd.DataFrame(index=soil.index)
    valid_count = pd.Series(0, index=soil.index, dtype=int)
    for col in sensor_cols:
        values = numeric(soil[col])
        values = values.where(values.between(0, 1))
        valid_count += values.notna().astype(int)
        median = values.median()
        iqr = values.quantile(0.75) - values.quantile(0.25)
        if np.isfinite(iqr) and iqr > 0:
            standardized[col] = (values - median) / iqr
    soil_index = standardized.median(axis=1, skipna=True).where(valid_count >= 2)
    return pd.DataFrame(
        {"date": soil["date"], "SWC_index": soil_index, "SWC_sensor_count": valid_count}
    )


def grouped_permutation_for_folds(
    data: pd.DataFrame,
    features: list[str],
    folds: list[dict],
    parameters: dict,
    groups: dict[str, list[str]],
    repeats: int,
    model_name: str,
) -> pd.DataFrame:
    rows = []
    year = data["date"].dt.year
    rng = np.random.default_rng(20261101)
    for fold in folds:
        test_mask = year.isin(fold["years"])
        train_mask = ~test_mask
        model = XGBRegressor(**parameters)
        model.fit(matrix(data.loc[train_mask], features), data.loc[train_mask, "ET"].to_numpy())
        test = data.loc[test_mask].copy()
        observed = test["ET"].to_numpy()
        baseline_pred = model.predict(matrix(test, features))
        baseline_rmse = math.sqrt(mean_squared_error(observed, baseline_pred))

        for group_name, group_features in groups.items():
            present = [f for f in group_features if f in features]
            if not present:
                continue
            increases = []
            for _ in range(repeats):
                permuted = test.copy()
                order = rng.permutation(len(permuted))
                permuted.loc[:, present] = test[present].to_numpy()[order]
                pred = model.predict(matrix(permuted, features))
                increases.append(math.sqrt(mean_squared_error(observed, pred)) - baseline_rmse)
            rows.append(
                {
                    "site": str(data["site"].iloc[0]),
                    "fold": fold["label"],
                    "model": model_name,
                    "group": group_name,
                    "n": int(len(test)),
                    "baseline_rmse": baseline_rmse,
                    "rmse_increase_mean": float(np.mean(increases)),
                    "rmse_increase_sd": float(np.std(increases, ddof=1)) if repeats > 1 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def run_soil_moisture_sensitivity(
    repo_root: Path,
    data_by_site: dict[str, pd.DataFrame],
    output_dir: Path,
    repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    permutation = []
    for site, full in data_by_site.items():
        soil = standardized_soil_index(repo_root, site)
        matched = full.merge(soil, on="date", how="inner")
        matched = matched.loc[matched["SWC_index"].notna()].sort_values("date").reset_index(drop=True)
        parameters = MODEL_PARAMETERS[site]
        with_swc_features = [*BASE_FEATURES, "SWC_index"]
        groups_with_swc = {**FEATURE_GROUPS, "Upper-soil moisture": ["SWC_index"]}

        if site == "EMS":
            folds = build_folds(matched, site, soil_overlap=True)
            no_swc = fit_blocked_predictions(matched, BASE_FEATURES, folds, parameters)
            with_swc = fit_blocked_predictions(matched, with_swc_features, folds, parameters)
            subsets = [(matched, folds)]
        else:
            no_swc_parts = []
            with_swc_parts = []
            subsets = []
            for period in ["pre", "transition", "post"]:
                period_data = matched.loc[matched["period"] == period].copy().reset_index(drop=True)
                period_folds = build_within_period_folds(period_data, period)
                no_swc_parts.append(
                    fit_blocked_predictions(period_data, BASE_FEATURES, period_folds, parameters)
                )
                with_swc_parts.append(
                    fit_blocked_predictions(period_data, with_swc_features, period_folds, parameters)
                )
                subsets.append((period_data, period_folds))
            no_swc = pd.concat(no_swc_parts, ignore_index=True)
            with_swc = pd.concat(with_swc_parts, ignore_index=True)

        metrics.append(summarize_predictions(no_swc, "matched_without_SWC"))
        metrics.append(summarize_predictions(with_swc, "matched_with_SWC"))

        for period_data, period_folds in subsets:
            permutation.append(
                grouped_permutation_for_folds(
                    period_data,
                    BASE_FEATURES,
                    period_folds,
                    parameters,
                    FEATURE_GROUPS,
                    repeats,
                    "matched_without_SWC",
                )
            )
            permutation.append(
                grouped_permutation_for_folds(
                    period_data,
                    with_swc_features,
                    period_folds,
                    parameters,
                    groups_with_swc,
                    repeats,
                    "matched_with_SWC",
                )
            )

    metrics_df = pd.concat(metrics, ignore_index=True)
    permutation_df = pd.concat(permutation, ignore_index=True)
    metrics_df.to_csv(output_dir / "soil_moisture_sensitivity.csv", index=False)
    permutation_df.to_csv(output_dir / "soil_moisture_grouped_permutation.csv", index=False)
    return metrics_df, permutation_df


def input_coverage(data_by_site: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for site, data in data_by_site.items():
        for feature in ["ET", *BASE_FEATURES]:
            rows.append(
                {
                    "site": site,
                    "variable": feature,
                    "n_rows": len(data),
                    "n_nonmissing": int(data[feature].notna().sum()),
                    "percent_nonmissing": 100.0 * data[feature].notna().mean(),
                    "start_date": data["date"].min().date().isoformat(),
                    "end_date": data["date"].max().date().isoformat(),
                }
            )
    return pd.DataFrame(rows)


def save_fold_definitions(data_by_site: dict[str, pd.DataFrame], output_dir: Path) -> None:
    rows = []
    for site, data in data_by_site.items():
        for fold in build_folds(data, site):
            years = [int(y) for y in fold["years"]]
            rows.append(
                {
                    "site": site,
                    "fold": fold["label"],
                    "test_year_start": min(years),
                    "test_year_end": max(years),
                    "test_years": ",".join(map(str, years)),
                    "test_rows": int(data["date"].dt.year.isin(years).sum()),
                    "training_rows": int((~data["date"].dt.year.isin(years)).sum()),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "blocked_fold_definitions.csv", index=False)


def plot_blocked_validation(predictions: pd.DataFrame, performance: pd.DataFrame, path: Path) -> None:
    panels = [("EMS", "all"), ("HEM", "pre"), ("HEM", "transition"), ("HEM", "post")]
    titles = ["EMS: all years", "HEM: pre-disturbance", "HEM: transition", "HEM: post-disturbance"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    for label, ax, (site, period), title in zip("abcd", axes.ravel(), panels, titles):
        subset = predictions.loc[(predictions["site"] == site)]
        if period != "all":
            subset = subset.loc[subset["period"] == period]
        metric = performance.loc[
            (performance["site"] == site) & (performance["period"] == period)
        ].iloc[0]
        color = SITE_COLORS[site]
        ax.scatter(
            subset["observed_ET"],
            subset["predicted_ET"],
            s=24,
            alpha=0.42,
            color=color,
            edgecolors="none",
        )
        low = min(subset["observed_ET"].min(), subset["predicted_ET"].min())
        high = max(subset["observed_ET"].max(), subset["predicted_ET"].max())
        ax.plot([low, high], [low, high], color="black", linewidth=1.8)
        ax.set_title(title, fontsize=18)
        ax.set_xlabel("Observed ET (mm day⁻¹)", fontsize=16)
        ax.set_ylabel("Predicted ET (mm day⁻¹)", fontsize=16)
        ax.tick_params(labelsize=13)
        ax.text(
            0.04,
            0.96,
            f"R²={metric['r2']:.2f}\nNSE={metric['nse']:.2f}\nMAE={metric['mae']:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "0.8"},
        )
        ax.text(-0.14, 1.07, f"{label})", transform=ax.transAxes, fontsize=20, fontweight="bold")
        ax.grid(alpha=0.22)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_group_ablation(importance: pd.DataFrame, path: Path) -> None:
    panels = [("EMS", "all"), ("HEM", "pre"), ("HEM", "transition"), ("HEM", "post")]
    titles = [
        "EMS, 1993–2023",
        "HEM, pre-disturbance (2005–2010)",
        "HEM, transition (2011–2014)",
        "HEM, post-disturbance (2015–2022)",
    ]

    plotted = importance.loc[
        importance["removed"].isin(DISPLAY_GROUP_ORDER)
        & importance.set_index(["site", "period"]).index.isin(panels)
    ].copy()
    plotted["relative_delta_rmse"] = (
        100.0 * plotted["delta_rmse"] / plotted["baseline_rmse"]
    )
    global_min = min(0.0, float(plotted["relative_delta_rmse"].min()))
    global_max = max(0.0, float(plotted["relative_delta_rmse"].max()))
    extent = max(abs(global_min), abs(global_max), 0.01)
    x_limits = (-1.20 * extent, 1.20 * extent)
    label_offset = 0.012 * (x_limits[1] - x_limits[0])

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(22, 16),
        sharex=True,
    )
    fig.subplots_adjust(
        left=0.20,
        right=0.98,
        bottom=0.11,
        top=0.88,
        wspace=0.20,
        hspace=0.36,
    )
    for panel_index, (label, ax, (site, period), title) in enumerate(
        zip("abcd", axes.ravel(), panels, titles)
    ):
        subset = importance.loc[
            (importance["site"] == site)
            & (importance["period"] == period)
            & (importance["removed"].isin(DISPLAY_GROUP_ORDER))
        ].set_index("removed").reindex(DISPLAY_GROUP_ORDER)

        values = (
            100.0 * subset["delta_rmse"] / subset["baseline_rmse"]
        ).to_numpy(dtype=float)
        y_positions = np.arange(len(DISPLAY_GROUP_ORDER))
        colors = [SITE_COLORS[site] if value >= 0 else "#9E9E9E" for value in values]
        bars = ax.barh(
            y_positions,
            values,
            height=0.67,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        y_tick_labels = (
            [GROUP_DISPLAY_LABELS[group] for group in DISPLAY_GROUP_ORDER]
            if panel_index % 2 == 0
            else [""] * len(DISPLAY_GROUP_ORDER)
        )
        ax.set_yticks(y_positions, y_tick_labels)
        ax.invert_yaxis()
        ax.set_xlim(*x_limits)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.axvline(0, color="black", linewidth=1.8, zorder=4)
        ax.set_title(title, fontsize=27, pad=17)
        ax.tick_params(
            axis="x",
            labelsize=21,
            labelbottom=True,
            width=1.4,
            length=7,
        )
        ax.tick_params(axis="y", labelsize=22, width=1.4, length=0)
        ax.grid(axis="x", color="#D8D8D8", linewidth=0.9, alpha=0.75, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            -0.12 if panel_index % 2 == 0 else -0.08,
            1.08,
            f"{label})",
            transform=ax.transAxes,
            fontsize=30,
            fontweight="bold",
            ha="left",
            va="top",
        )

        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                continue
            x_text = value + label_offset if value >= 0 else value - label_offset
            displayed_value = 0.0 if abs(value) < 0.05 else value
            ax.text(
                x_text,
                bar.get_y() + bar.get_height() / 2,
                f"{displayed_value:+.1f}%" if displayed_value else "0.0%",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=19,
                fontweight="semibold",
                color="#222222",
                clip_on=False,
            )

    fig.supxlabel(
        "Relative change in RMSE when each predictor group was excluded (%)",
        fontsize=25,
        y=0.035,
    )
    legend_items = [
        (0.23, SITE_COLORS["EMS"], "EMS: RMSE increased"),
        (0.46, SITE_COLORS["HEM"], "HEM: RMSE increased"),
        (0.69, "#9E9E9E", "RMSE decreased"),
    ]
    for x_position, color, label in legend_items:
        fig.add_artist(
            Rectangle(
                (x_position, 0.957),
                0.017,
                0.017,
                transform=fig.transFigure,
                facecolor=color,
                edgecolor="none",
            )
        )
        fig.text(
            x_position + 0.023,
            0.9655,
            label,
            fontsize=21,
            ha="left",
            va="center",
        )
    fig.savefig(
        path,
        dpi=600,
        facecolor="white",
    )
    plt.close(fig)


def plot_sensitivity(
    variants: pd.DataFrame,
    soil_metrics: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(20, 8.5))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.22, top=0.86, wspace=0.18)

    carbon_rows = []
    for site, period in [("EMS", "all"), ("HEM", "pre"), ("HEM", "transition"), ("HEM", "post")]:
        subset = variants.loc[(variants["site"] == site) & (variants["period"] == period)]
        baseline = subset.loc[subset["variant"] == "baseline", "rmse"]
        if baseline.empty:
            continue
        base = float(baseline.iloc[0])
        for variant, label in [("drop::GPP", "Remove GPP"), ("drop::NEE", "Remove NEE"), ("drop::Carbon and canopy activity", "Remove both")]:
            value = subset.loc[subset["variant"] == variant, "rmse"]
            if not value.empty:
                carbon_rows.append(
                    {
                        "label": f"{site} {PERIOD_LABELS[period]}",
                        "variant": label,
                        "relative_delta": 100.0 * (float(value.iloc[0]) - base) / base,
                    }
                )
    carbon = pd.DataFrame(carbon_rows)
    labels = list(dict.fromkeys(carbon["label"]))
    x = np.arange(len(labels))
    width = 0.23
    variant_colors = {
        "Remove GPP": "#EF7D73",
        "Remove NEE": "#2E8D90",
        "Remove both": "#777777",
    }
    for i, variant in enumerate(["Remove GPP", "Remove NEE", "Remove both"]):
        values = [
            float(
                carbon.loc[
                    (carbon["label"] == label) & (carbon["variant"] == variant),
                    "relative_delta",
                ].iloc[0]
            )
            for label in labels
        ]
        bars = axes[0].bar(
            x + (i - 1) * width,
            values,
            width,
            color=variant_colors[variant],
            label=variant,
            zorder=3,
        )
        axes[0].bar_label(
            bars,
            labels=[f"{value:+.1f}%" for value in values],
            padding=3,
            fontsize=14,
            fontweight="semibold",
        )
    axes[0].axhline(0, color="black", linewidth=1.5, zorder=4)
    axes[0].set_xticks(
        x,
        ["EMS\nall years", "HEM\npre-disturbance", "HEM\ntransition", "HEM\npost-disturbance"],
    )
    axes[0].set_ylabel("Relative change in RMSE after exclusion (%)", fontsize=20)
    axes[0].set_title("Carbon-flux sensitivity", fontsize=25, pad=15)
    axes[0].tick_params(axis="x", labelsize=16, pad=8)
    axes[0].tick_params(axis="y", labelsize=16)
    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    axes[0].grid(axis="y", alpha=0.22, zorder=0)
    axes[0].text(0.00, 1.08, "a)", transform=axes[0].transAxes, fontsize=28, fontweight="bold")

    soil_rows = []
    for site, period in [("EMS", "all"), ("HEM", "pre"), ("HEM", "transition"), ("HEM", "post")]:
        subset = soil_metrics.loc[(soil_metrics["site"] == site) & (soil_metrics["period"] == period)]
        without = subset.loc[subset["variant"] == "matched_without_SWC", "rmse"]
        with_swc = subset.loc[subset["variant"] == "matched_with_SWC", "rmse"]
        if not without.empty and not with_swc.empty:
            soil_rows.append(
                {
                    "label": f"{site} {PERIOD_LABELS[period]}",
                    "improvement": 100.0
                    * (float(without.iloc[0]) - float(with_swc.iloc[0]))
                    / float(without.iloc[0]),
                }
            )
    soil = pd.DataFrame(soil_rows)
    colors = [SITE_COLORS["EMS"] if value.startswith("EMS") else SITE_COLORS["HEM"] for value in soil["label"]]
    bars = axes[1].bar(
        np.arange(len(soil)),
        soil["improvement"],
        color=colors,
        width=0.68,
        zorder=3,
    )
    axes[1].bar_label(
        bars,
        labels=[f"{value:+.1f}%" for value in soil["improvement"]],
        padding=3,
        fontsize=14,
        fontweight="semibold",
    )
    axes[1].axhline(0, color="black", linewidth=1.5, zorder=4)
    axes[1].set_xticks(
        np.arange(len(soil)),
        ["EMS\nall years", "HEM\npre-disturbance", "HEM\ntransition", "HEM\npost-disturbance"],
    )
    axes[1].set_ylabel(
        "Relative reduction in RMSE after adding\nsoil moisture (%)",
        fontsize=20,
    )
    axes[1].set_title("Soil-moisture sensitivity", fontsize=25, pad=15)
    axes[1].tick_params(axis="x", labelsize=16, pad=8)
    axes[1].tick_params(axis="y", labelsize=16)
    axes[1].grid(axis="y", alpha=0.22, zorder=0)
    axes[1].text(0.00, 1.08, "b)", transform=axes[1].transAxes, fontsize=28, fontweight="bold")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.29, 0.985),
        ncol=3,
        fontsize=15,
        frameon=False,
    )

    fig.savefig(path, dpi=600, facecolor="white")
    plt.close(fig)


def print_summary(
    performance: pd.DataFrame,
    hem_strategy: pd.DataFrame,
    importance: pd.DataFrame,
    soil_metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    print("\nBLOCKED VALIDATION")
    cols = ["site", "period", "n", "r2", "nse", "mae", "rmse", "pbias"]
    print(performance[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nHEM MODEL-STRATEGY COMPARISON")
    print(
        hem_strategy[["period", "strategy", "n", "r2", "nse", "mae", "rmse", "pbias"]]
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    print("\nPREDICTOR-GROUP REMOVAL (positive delta RMSE means predictive skill decreased)")
    display = importance.loc[importance["removed"].isin(DISPLAY_GROUP_ORDER)]
    print(
        display[["site", "period", "removed", "delta_rmse", "delta_nse"]]
        .sort_values(["site", "period", "delta_rmse"], ascending=[True, True, False])
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    print("\nSOIL-MOISTURE SENSITIVITY")
    print(
        soil_metrics[["site", "period", "variant", "n", "nse", "rmse", "pbias"]]
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )
    print(f"\nOutputs written to: {output_dir}")
    print("No original scripts, processed input files, or manuscript figures were modified.")


def main() -> None:
    args = parse_args()
    if args.plot_only:
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else Path.cwd() / "xgboost_reviewer_analysis_results"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        importance_path = output_dir / "selected_model_ablation_importance.csv"
        if not importance_path.exists():
            raise FileNotFoundError(
                f"Plot-only mode requires the saved sensitivity table: {importance_path}"
            )
        plot_group_ablation(
            pd.read_csv(importance_path),
            output_dir / "Figure_R2_group_ablation.png",
        )
        print(f"Updated Figure 6 candidate: {output_dir / 'Figure_R2_group_ablation.png'}")
        return

    repo_root = detect_repo_root(args.repo_root)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else repo_root / "outputs" / "xgboost_reviewer_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data_by_site = {site: load_site_daily(repo_root, site) for site in ["EMS", "HEM"]}
    coverage = input_coverage(data_by_site)
    coverage.to_csv(output_dir / "input_coverage.csv", index=False)
    measured_coverage = []
    for site, data in data_by_site.items():
        annual = (
            data.assign(year=data["date"].dt.year)
            .groupby("year", as_index=False)
            .agg(
                measured_days=("ET", "count"),
                mean_subdaily_coverage=("measured_fraction", "mean"),
                minimum_subdaily_coverage=("measured_fraction", "min"),
            )
        )
        annual.insert(0, "site", site)
        measured_coverage.append(annual)
    pd.concat(measured_coverage, ignore_index=True).to_csv(
        output_dir / "measured_target_coverage_by_year.csv", index=False
    )
    save_fold_definitions(data_by_site, output_dir)

    predictions, performance, variants, importance = run_full_record_models(data_by_site, output_dir)
    hem_state_predictions, hem_strategy = run_hem_state_specific_models(
        data_by_site["HEM"], predictions, output_dir
    )
    hem_within_variants, hem_within_importance = run_hem_within_period_sensitivities(
        data_by_site["HEM"], hem_state_predictions, output_dir
    )
    selected_variants = pd.concat(
        [
            variants.loc[(variants["site"] == "EMS") & (variants["period"] == "all")],
            hem_within_variants,
        ],
        ignore_index=True,
    )
    selected_importance = pd.concat(
        [
            importance.loc[(importance["site"] == "EMS") & (importance["period"] == "all")],
            hem_within_importance,
        ],
        ignore_index=True,
    )
    selected_variants.to_csv(output_dir / "selected_model_ablation_performance.csv", index=False)
    selected_importance.to_csv(output_dir / "selected_model_ablation_importance.csv", index=False)
    soil_metrics, soil_permutation = run_soil_moisture_sensitivity(
        repo_root,
        data_by_site,
        output_dir,
        repeats=args.permutation_repeats,
    )

    manifest = {
        "analysis": "XGBoost reviewer validation and sensitivity",
        "repo_root": str(repo_root),
        "data_periods": {
            site: {
                "start": data["date"].min().date().isoformat(),
                "end": data["date"].max().date().isoformat(),
                "rows": int(len(data)),
            }
            for site, data in data_by_site.items()
        },
        "model_parameters": MODEL_PARAMETERS,
        "base_features": BASE_FEATURES,
        "feature_groups": FEATURE_GROUPS,
        "validation": "Complete contiguous multi-year periods withheld; fixed optimized parameters",
        "notes": [
            "HEM 2023 excluded because the processed daily GPP record is unavailable.",
            "Validation targets use only days with at least 75% measured subdaily latent-heat coverage.",
            "The original latent-heat-flux field was excluded because ET is calculated from latent heat.",
            "HEM cross-state transfer was compared with blocked validation trained within each vegetation period.",
            "Soil-moisture comparisons use identical matched dates with and without SWC.",
            "Group-removal and permutation results are predictive sensitivities, not causal effects.",
        ],
    }
    with (output_dir / "analysis_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    plot_blocked_validation(
        predictions,
        performance,
        output_dir / "Figure_R1_blocked_validation.png",
    )
    manuscript_figure_dir = output_dir.parent / "figures"
    manuscript_figure_dir.mkdir(parents=True, exist_ok=True)
    plot_group_ablation(
        selected_importance,
        manuscript_figure_dir / "Figure8_XGBoost_predictor_group_sensitivity.png",
    )
    plot_sensitivity(
        selected_variants,
        soil_metrics,
        output_dir / "FigureS6_XGBoost_carbon_soil_sensitivity.png",
    )
    print_summary(performance, hem_strategy, selected_importance, soil_metrics, output_dir)


if __name__ == "__main__":
    main()
