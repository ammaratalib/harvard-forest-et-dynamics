"""Run one ET-partitioning method: Nelson et al. (2018) TEA.

Inputs are the standardized, manuscript-consistent subdaily files produced by
01_prepare_tea_inputs.py. The implementation follows the published TEA logic:
identify dry, physiologically active periods with the conservative surface
wetness index (CSWI), train a random forest on ecosystem WUE, use the 75th
percentile WUE prediction, and calculate T = GPP / predicted WUE and E = ET-T.

Method citation:
    Nelson, J. A., et al. (2018). Coupling Water and Carbon Fluxes to
    Constrain Estimates of Transpiration: The TEA Algorithm. Journal of
    Geophysical Research: Biogeosciences, 123, 3617-3632.
    https://doi.org/10.1029/2018JG004727

Original TEA code:
    https://github.com/jnelson18/TranspirationEstimationAlgorithm

The original package used pyquantrf. This implementation uses the maintained
quantile-forest RandomForestQuantileRegressor to perform the same required
conditional-quantile prediction (Meinshausen, 2006):
    https://github.com/zillow/quantile-forest

Run from anywhere:
    python -u code/09_et_partitioning/02_run_nelson_tea.py
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm, theilslopes
from quantile_forest import RandomForestQuantileRegressor


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
INPUT = REPO / "data" / "processed" / "et_partitioning"
OUTPUT = REPO / "outputs" / "et_partitioning"

# Harvard Forest tower coordinates. Longitude is only needed for local solar
# time; timestamps are treated as local standard time (UTC-5).
LATITUDE_DEG = 42.54
LONGITUDE_DEG = -72.17
TIME_ZONE_H = -5.0
RANDOM_SEED = 42
MOLAR_MASS_C_G_PER_UMOL = 12.011e-6
CSWI_LIMIT_MM = -0.5


def original_mann_kendall(values: pd.Series) -> tuple[float, float]:
    """Original Mann-Kendall test used throughout the manuscript workflow."""
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 2:
        return np.nan, np.nan
    s = sum(np.sign(y[j] - y[:j]).sum() for j in range(1, n))
    _, counts = np.unique(y, return_counts=True)
    ties = counts[counts > 1]
    var_s = (
        n * (n - 1) * (2 * n + 5)
        - np.sum(ties * (ties - 1) * (2 * ties + 5))
    ) / 18.0
    if var_s <= 0:
        return 0.0, 1.0
    z = (s - 1) / np.sqrt(var_s) if s > 0 else ((s + 1) / np.sqrt(var_s) if s < 0 else 0.0)
    return float(s / (0.5 * n * (n - 1))), float(2.0 * norm.sf(abs(z)))


def relative_humidity(tair_c: np.ndarray, vpd_kpa: np.ndarray) -> np.ndarray:
    """Fallback RH (%) reconstruction for legacy standardized inputs."""
    es = 0.6108 * np.exp(17.27 * tair_c / (tair_c + 237.3))
    return np.clip(100.0 * (1.0 - vpd_kpa / es), 0.0, 100.0)


def potential_radiation(timestamp: pd.Series) -> np.ndarray:
    """Potential radiation exactly following Nelson's released PotRad.py."""
    ts = pd.to_datetime(timestamp)
    doy = ts.dt.dayofyear.to_numpy(float)
    hour = (
        ts.dt.hour.to_numpy(float)
        + ts.dt.minute.to_numpy(float) / 60.0
        + ts.dt.second.to_numpy(float) / 3600.0
    )
    fractional_year = 2.0 * np.pi * (doy - 1.0) / 365.24
    equation_of_time_h = (
        0.0072 * np.cos(fractional_year)
        - 0.0528 * np.cos(2.0 * fractional_year)
        - 0.0012 * np.cos(3.0 * fractional_year)
        - 0.1229 * np.sin(fractional_year)
        - 0.1565 * np.sin(2.0 * fractional_year)
        - 0.0041 * np.sin(3.0 * fractional_year)
    )
    solar_time_h = hour + (LONGITUDE_DEG / 15.0 - TIME_ZONE_H) + equation_of_time_h
    solar_time_rad = (solar_time_h - 12.0) * np.pi / 12.0
    solar_time_rad = np.where(solar_time_rad < -np.pi, solar_time_rad + 2.0 * np.pi, solar_time_rad)
    solar_declination = np.deg2rad(
        0.33281
        - 22.984 * np.cos(fractional_year)
        - 0.34990 * np.cos(2.0 * fractional_year)
        - 0.13980 * np.cos(3.0 * fractional_year)
        + 3.7872 * np.sin(fractional_year)
        + 0.03205 * np.sin(2.0 * fractional_year)
        + 0.07187 * np.sin(3.0 * fractional_year)
    )
    lat = np.deg2rad(LATITUDE_DEG)
    solar_elevation = np.arcsin(
        np.sin(solar_declination) * np.sin(lat)
        + np.cos(solar_declination) * np.cos(lat) * np.cos(solar_time_rad)
    )
    extraterrestrial = 1366.1 * (
        1.00011
        + 0.034221 * np.cos(fractional_year)
        + 0.00128 * np.sin(fractional_year)
        + 0.000719 * np.cos(2.0 * fractional_year)
        + 0.000077 * np.sin(2.0 * fractional_year)
    )
    return extraterrestrial * np.maximum(np.sin(solar_elevation), 0.0)


def precipitation_at_timestep(df: pd.DataFrame) -> np.ndarray:
    """Place each measured daily total at the first timestep of that day.

    Exact storm timing is unavailable in the manuscript climate product. A
    single daily pulse is conservative for CSWI because it marks the following
    daytime period as wet and the storage subsequently declines through ET.
    """
    event = np.zeros(len(df), dtype=float)
    first = ~df["date"].duplicated()
    event[first.to_numpy()] = df.loc[first, "precip_mm_day"].fillna(5.0).to_numpy(float)
    return event


def cswi(precip: np.ndarray, et: np.ndarray, storage_max: float = 5.0) -> np.ndarray:
    """Conservative surface wetness index from Nelson et al. (2018)."""
    result = np.zeros(len(et), dtype=float)
    result[0] = storage_max
    p = np.where(np.isfinite(precip) & (precip >= 0), precip, storage_max)
    et_use = np.where(np.isfinite(et), et, 0.0)
    for i in range(1, len(result)):
        step = min(result[i - 1] + p[i] - et_use[i], storage_max)
        result[i] = max(step, min(p[i - 1], storage_max)) if p[i] > 0 else step
    return result


def grouped_diurnal_centroid(df: pd.DataFrame, value: str) -> pd.Series:
    """Daily flux centroid using the raw flux, as in Nelson's code."""
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    numerator = (df[value] * hour).groupby(df["date"]).transform("sum")
    denominator = df[value].groupby(df["date"]).transform("sum")
    return numerator / denominator.replace(0, np.nan)


def daily_correlation(df: pd.DataFrame) -> pd.Series:
    """Nelson daily water-carbon index: daylight Pearson r squared."""
    carbon = df["GPP_umol_m2_s"] * np.sqrt(df["VPD_kPa"].clip(lower=0))

    def corr(group: pd.DataFrame) -> float:
        idx = group.index
        x = df.loc[idx, "ET_mm"].copy()
        y = carbon.loc[idx].copy()
        daylight = df.loc[idx, "Rg_pot_W_m2"] > 0
        valid = daylight & x.notna() & y.notna()
        if valid.sum() < 6:
            return np.nan
        x = x[valid]
        y = y[valid]
        x_max = x.max()
        y_max = y.max()
        if not np.isfinite(x_max) or not np.isfinite(y_max) or x_max == 0 or y_max == 0:
            return np.nan
        r = (x / x_max).corr(y / y_max)
        return r * r if np.isfinite(r) else np.nan

    by_day = df.groupby("date", sort=False).apply(corr, include_groups=False)
    return df["date"].map(by_day)


def add_tea_variables(df: pd.DataFrame, timestep_seconds: int) -> pd.DataFrame:
    x = df.copy().sort_values("timestamp").reset_index(drop=True)
    x["timestamp"] = pd.to_datetime(x["timestamp"])
    x["date"] = x["timestamp"].dt.normalize()
    x["year"] = x["timestamp"].dt.year
    if "RH_percent" not in x:
        x["RH_percent"] = relative_humidity(
            x["Tair_C"].to_numpy(), x["VPD_kPa"].to_numpy()
        )
    else:
        x["RH_percent"] = pd.to_numeric(x["RH_percent"], errors="coerce")
    x["Rg_pot_W_m2"] = potential_radiation(x["timestamp"])
    x["precip_mm_timestep"] = precipitation_at_timestep(x)
    x["CSWI_mm"] = cswi(x["precip_mm_timestep"].to_numpy(), x["ET_mm"].to_numpy())

    daily_pot = (x["Rg_pot_W_m2"] * timestep_seconds / 1e6).groupby(x["date"]).transform("sum")
    x["Rg_pot_daily_MJ_m2"] = daily_pot
    x["C_Rg_ET_hours"] = grouped_diurnal_centroid(x, "ET_mm") - grouped_diurnal_centroid(x, "Rg_W_m2")
    x["DWCI"] = daily_correlation(x)

    daily = x.groupby("date", sort=False).agg(GPP_mean=("GPP_umol_m2_s", "mean"))
    daily["GPP_gradient"] = np.gradient(gaussian_filter1d(daily["GPP_mean"].fillna(0).to_numpy(), sigma=20))
    x["GPP_gradient"] = x["date"].map(daily["GPP_gradient"])
    # The TEA random forest uses the subdaily potential-radiation gradient.
    x["Rg_pot_gradient"] = np.gradient(x["Rg_pot_W_m2"].to_numpy())

    daily_gpp = (
        x["GPP_umol_m2_s"]
        * MOLAR_MASS_C_G_PER_UMOL
        * timestep_seconds
    ).groupby(x["date"]).transform("sum")
    x["season_flag"] = (x["Tair_C"] > 5.0) & (daily_gpp > 0.5) & (x["GPP_umol_m2_s"] > 0.05)
    x["daylight_flag"] = x["Rg_pot_W_m2"] > 0
    x["positive_flux_flag"] = (x["ET_mm"] > 0) & (x["GPP_umol_m2_s"] > 0)
    # 1 mm of water over 1 m2 equals 1 kg H2O.
    x["GPP_gC_timestep"] = (
        x["GPP_umol_m2_s"] * MOLAR_MASS_C_G_PER_UMOL * timestep_seconds
    )
    x["WUE_target_gC_kgH2O"] = (
        x["GPP_gC_timestep"] / x["ET_mm"].where(x["ET_mm"] > 0)
    )
    return x


def validate_input_units(x: pd.DataFrame, site: str, timestep_seconds: int) -> None:
    """Fail early when a source variable appears to have the wrong units."""
    checks = {
        "ET (mm timestep-1), 99th percentile": (x["ET_mm"].quantile(0.99), -0.1, 1.5),
        "GPP (umol m-2 s-1), 99th percentile": (x["GPP_umol_m2_s"].quantile(0.99), 0.0, 80.0),
        "Tair (degC), median": (x["Tair_C"].median(), -20.0, 35.0),
        "VPD (kPa), 99th percentile": (x["VPD_kPa"].quantile(0.99), 0.0, 8.0),
        "RH (%), median": (x["RH_percent"].median(), 0.0, 100.0),
        "wind (m s-1), 99th percentile": (x["WS_m_s"].quantile(0.99), 0.0, 30.0),
        "daily precipitation (mm), 99th percentile": (x["precip_mm_day"].quantile(0.99), 0.0, 150.0),
    }
    for name, (value, lower, upper) in checks.items():
        if not np.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{site}: unit/range check failed for {name}: {value}")
    expected_steps = 24 * 3600 // timestep_seconds
    observed_steps = x.groupby("date").size().median()
    if observed_steps != expected_steps:
        raise ValueError(
            f"{site}: expected {expected_steps} records/day but median is {observed_steps}"
        )


def quantile_prediction(
    model: RandomForestQuantileRegressor, features: pd.DataFrame, q: float = 0.75
) -> np.ndarray:
    answer = np.full(len(features), np.nan)
    chunk = 25_000
    for start in range(0, len(features), chunk):
        stop = min(start + chunk, len(features))
        answer[start:stop] = model.predict(features.iloc[start:stop], quantiles=q)
    return answer


def partition_site(
    path: Path, site: str, timestep_seconds: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path, low_memory=False)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw["date"] = raw["timestamp"].dt.normalize()
    x = add_tea_variables(raw, timestep_seconds)
    validate_input_units(x, site, timestep_seconds)

    predictors = [
        "Rg_W_m2", "Tair_C", "RH_percent", "WS_m_s",
        "Rg_pot_daily_MJ_m2", "Rg_pot_gradient", "year",
        "GPP_gradient", "DWCI", "C_Rg_ET_hours", "CSWI_mm",
    ]
    finite_predictors = np.isfinite(x[predictors]).all(axis=1)
    finite_core = finite_predictors & np.isfinite(x["ET_mm"]) & np.isfinite(x["GPP_umol_m2_s"])
    train = (
        x["daylight_flag"]
        & x["season_flag"]
        & x["positive_flux_flag"]
        & (x["CSWI_mm"] < CSWI_LIMIT_MM)
        & finite_core
        & np.isfinite(x["WUE_target_gC_kgH2O"])
        # Nelson et al. recommend flux quality flags for model training.
        # Prediction still covers the complete gap-filled time series.
        & x["LE_source"].eq("observed")
    )
    if train.sum() <= 240:
        raise RuntimeError(f"{site}: only {train.sum()} TEA training records")

    model = RandomForestQuantileRegressor(
        n_estimators=100,
        default_quantiles=0.75,
        max_features=int(np.ceil(len(predictors) / 3)),
        min_samples_leaf=1,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    model.fit(x.loc[train, predictors], x.loc[train, "WUE_target_gC_kgH2O"])

    # Nelson predicts every finite daylight record; night-time T is set to zero.
    predict = x["daylight_flag"] & finite_core
    wue = np.full(len(x), np.nan)
    wue[predict] = quantile_prediction(model, x.loc[predict, predictors], q=0.75)
    transpiration = np.zeros(len(x), dtype=float)
    # WUE is g C kg-1 H2O; GPP is g C m-2 per timestep. Because
    # 1 kg H2O m-2 = 1 mm, GPP/WUE returns T in mm per timestep.
    transpiration[predict] = x.loc[predict, "GPP_gC_timestep"].to_numpy() / wue[predict]
    transpiration[~np.isfinite(transpiration)] = np.nan
    x["TEA_WUE_q75_gC_kgH2O"] = wue
    x["T_mm"] = transpiration
    x["E_mm"] = x["ET_mm"] - x["T_mm"]
    x["T_over_ET"] = x["T_mm"] / x["ET_mm"].where(x["ET_mm"] > 0)
    x["E_over_ET"] = x["E_mm"] / x["ET_mm"].where(x["ET_mm"] > 0)
    x["TEA_training_record"] = train

    mass_balance_error = (x["ET_mm"] - x["T_mm"] - x["E_mm"]).abs().max()
    if mass_balance_error > 1e-12:
        raise ValueError(f"{site}: ET mass balance failed ({mass_balance_error:.3e} mm)")

    daily = x.groupby("date", as_index=False).agg(
        site=("site", "first"),
        year=("year", "first"),
        ET_mm=("ET_mm", "sum"),
        T_mm=("T_mm", "sum"),
        E_mm=("E_mm", "sum"),
        GPP_gC_m2=("GPP_umol_m2_s", lambda z: z.sum() * MOLAR_MASS_C_G_PER_UMOL * timestep_seconds),
        growing_season=("season_flag", "any"),
    )
    daily["T_over_ET"] = daily["T_mm"] / daily["ET_mm"].where(daily["ET_mm"] > 0)
    daily["E_over_ET"] = daily["E_mm"] / daily["ET_mm"].where(daily["ET_mm"] > 0)

    annual = daily.loc[daily["growing_season"]].groupby("year", as_index=False).agg(
        site=("site", "first"), ET_mm=("ET_mm", "sum"), T_mm=("T_mm", "sum"),
        E_mm=("E_mm", "sum"), GPP_gC_m2=("GPP_gC_m2", "sum"), days=("date", "size"),
    )
    annual["T_over_ET"] = annual["T_mm"] / annual["ET_mm"]
    annual["E_over_ET"] = annual["E_mm"] / annual["ET_mm"]

    print(f"\n{site} Nelson TEA")
    print(f"  training records: {train.sum():,} ({100 * train.mean():.2f}%)")
    dry = x["CSWI_mm"] < CSWI_LIMIT_MM
    print(f"  dry-surface records (CSWI < {CSWI_LIMIT_MM} mm): {dry.sum():,} ({100 * dry.mean():.2f}%)")
    print(f"  random-forest mean-prediction OOB R2 (diagnostic only): {model.oob_score_:.3f}")
    print(f"  growing-season mean T/ET: {annual['T_mm'].sum() / annual['ET_mm'].sum():.3f}")
    print(f"  subdaily negative E: {100 * (x['E_mm'] < 0).mean():.2f}%")
    print(f"  maximum ET-(T+E) error: {mass_balance_error:.3e} mm")
    for variable in ["T_over_ET", "E_mm", "T_mm"]:
        valid = annual[["year", variable]].dropna()
        tau, p = original_mann_kendall(valid[variable])
        slope = theilslopes(valid[variable], valid["year"])[0]
        print(f"  annual GS {variable:9s}: tau={tau:+.3f}, p={p:.4g}, Theil-Sen slope={slope:+.5g} yr-1")

    keep = [
        "timestamp", "date", "site", "ET_mm", "T_mm", "E_mm", "T_over_ET", "E_over_ET",
        "TEA_WUE_q75_gC_kgH2O", "GPP_umol_m2_s", "GPP_gC_timestep",
        "Rg_W_m2", "Tair_C", "VPD_kPa",
        "RH_percent", "WS_m_s", "precip_mm_timestep", "CSWI_mm",
        "season_flag", "TEA_training_record",
    ]
    return x[keep], daily, annual


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sites = [
        ("EMS", INPUT / "EMS_TEA_input_subdaily.csv", 3600),
        ("HEM", INPUT / "HEM_TEA_input_subdaily.csv", 1800),
    ]
    annual_all = []
    for site, path, seconds in sites:
        if not path.exists():
            raise FileNotFoundError(f"Run 01_prepare_tea_inputs.py first: {path}")
        subdaily, daily, annual = partition_site(path, site, seconds)
        subdaily.to_csv(OUTPUT / f"{site}_TEA_partition_subdaily.csv", index=False)
        daily.to_csv(OUTPUT / f"{site}_TEA_partition_daily.csv", index=False)
        annual.to_csv(OUTPUT / f"{site}_TEA_growing_season_annual.csv", index=False)
        annual_all.append(annual)
    pd.concat(annual_all, ignore_index=True).to_csv(
        OUTPUT / "TEA_growing_season_annual_both_sites.csv", index=False
    )
    print(f"\nOutputs written to: {OUTPUT}")
    print("Paste the complete terminal output back before changing canopy conductance or the manuscript.")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        main()
