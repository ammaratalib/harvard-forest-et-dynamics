"""Forest-specific, moisture-unlimited Penman--Monteith PET.

This is a separate reviewer-response analysis. It does not overwrite the
existing canopy-constrained PM or FAO-56 diagnostic workflows.

The workflow follows the Jarvis (1976) / Stewart (1988) approach:

    gc = gbase + gamp * fR(Rg) * fD(VPD) * fseason * fSWC(SWC)

One fixed parameter set is fitted for each site from dry-canopy conductance
observations in the upper quartile of the HF206 soil-moisture record. The
fitted parameters do not vary by year or disturbance period. This
moisture-sufficient calibration sets fSWC = 1 while retaining the fitted
radiation, VPD, and fixed seasonal responses. Following a published
Jarvis--Stewart application, the temperature response is set to unity because
it was not separately identifiable from VPD in these records. PET is calculated as

                         delta (Rn-G) + rho cp VPD / ra
    lambda E_p = ---------------------------------------------------
                   delta + gamma (1 + rs,pot / ra)

where rs,pot = 1 / gc,pot after converting molar conductance to m s-1.

Primary methodological references
---------------------------------
Jarvis, P. G. (1976), Philos. Trans. R. Soc. B, 273, 593-610,
https://doi.org/10.1098/rstb.1976.0035

Stewart, J. B. (1988), Agric. For. Meteorol., 43, 19-35,
https://doi.org/10.1016/0168-1923(88)90003-2

Ogink-Hendriks, M. J. (1995), Agric. For. Meteorol., 74, 99-118,
https://doi.org/10.1016/0168-1923(94)02180-R

Run from Git Bash at the repository root:

    python -u "code/05_pet/15_forest_jarvis_penman_monteith_pet.py"

Outputs are written to outputs/forest_jarvis_pet.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm, theilslopes


def locate_repository() -> Path:
    candidates: list[Path] = []
    if "__file__" in globals():
        here = Path(__file__).resolve()
        candidates.extend([here.parent, *here.parents])
    current = Path.cwd().resolve()
    candidates.extend([current, *current.parents])
    for candidate in dict.fromkeys(candidates):
        if (
            (candidate / "data" / "processed" / "pet_et").is_dir()
            and (candidate / "data" / "processed" / "soil_moisture").is_dir()
        ):
            return candidate
    raise FileNotFoundError("Could not locate harvard-forest-et-dynamics")


REPO = locate_repository()
PET_DATA = REPO / "data" / "processed" / "pet_et"
SOIL_DATA = REPO / "data" / "processed" / "soil_moisture"
CONDUCTANCE_DATA = REPO / "data" / "processed" / "conductance"
FIGURE_DATA = REPO / "outputs" / "figures"
OUT = REPO / "outputs" / "forest_jarvis_pet"

SECONDS_PER_DAY = 86_400.0
LATENT_HEAT_J_KG = 2.45e6
CP_AIR_J_KG_K = 1004.0
R_DRY_AIR_J_KG_K = 287.05
R_UNIVERSAL_J_MOL_K = 8.314462618
EPSILON = 0.622
VON_KARMAN = 0.41
CANOPY_HEIGHT_M = 25.0


SITE_CONFIG = {
    "EMS": {
        "daily": PET_DATA / "daily_ems_pet_ET.csv",
        "subdaily_pet": REPO / "data" / "processed" / "flux" / "gaps_ML_EMS.csv",
        "flux": REPO / "data" / "processed" / "flux" / "gaps_ML_EMS.csv",
        "soil": SOIL_DATA / "EMS_daily_soil_moisture.csv",
        "soil_columns": [f"hf206_upper15_sensor{i}_m3_m3" for i in range(1, 5)],
        "timestamp": "TIMESTAMP",
        "year": "Year",
        "temperature": "Tair_fill",
        "vpd": "VPD_fill",
        "wind": "WS_Rd",
        "pressure": "PA_Rd",
        "radiation": "Rg_fill",
        "net_radiation": "NETRAD_fill",
        "first_year": 1993,
        "last_year": 2023,
        "measurement_height_m": 29.0,
        "title": "US-Ha1, EMS Tower",
        "color": "#ef7f7d",
    },
    "HEM": {
        "daily": PET_DATA / "daily_hem_pet_ET.csv",
        "subdaily_pet": REPO / "data" / "processed" / "flux" / "gaps_ML_HEM_2023.csv",
        "flux": REPO / "data" / "processed" / "flux" / "gaps_ML_HEM_2023.csv",
        "soil": SOIL_DATA / "HEM_daily_soil_moisture.csv",
        "soil_columns": [f"upper15_sensor{i}_m3_m3" for i in range(1, 5)],
        "timestamp": "TIMESTAMP",
        "year": "Year",
        "temperature": "Tair",
        "vpd": "VPD",
        "wind": "WS",
        "pressure": "PA",
        "radiation": "Rg",
        "net_radiation": "NETRAD",
        "first_year": 2005,
        "last_year": 2023,
        "measurement_height_m": 33.0,
        "title": "US-Ha2, HEM Tower",
        "color": "#ef7f7d",
    },
}


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def original_mann_kendall(values) -> tuple[float, float]:
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 2:
        return np.nan, np.nan
    score = sum(np.sign(y[j] - y[:j]).sum() for j in range(1, n))
    _, counts = np.unique(y, return_counts=True)
    ties = counts[counts > 1]
    variance = (
        n * (n - 1) * (2 * n + 5)
        - np.sum(ties * (ties - 1) * (2 * ties + 5))
    ) / 18.0
    if variance <= 0:
        return 0.0, 1.0
    z = ((score - 1) / np.sqrt(variance) if score > 0 else
         (score + 1) / np.sqrt(variance) if score < 0 else 0.0)
    tau = score / (0.5 * n * (n - 1))
    return float(tau), float(2.0 * norm.sf(abs(z)))


def robust_soil_index(site: str) -> pd.DataFrame:
    """Daily site index from four HF206 upper-15-cm sensors.

    Each sensor is centered by its median and scaled by its interquartile
    range before the daily median is taken. At least two sensors are required.
    """
    cfg = SITE_CONFIG[site]
    soil = pd.read_csv(cfg["soil"], low_memory=False)
    soil["date"] = pd.to_datetime(soil["date"], errors="coerce").dt.normalize()
    standardized = []
    for column in cfg["soil_columns"]:
        values = numeric(soil, column)
        valid = values.between(0.0, 1.0)
        values = values.where(valid)
        median = values.median()
        iqr = values.quantile(0.75) - values.quantile(0.25)
        if np.isfinite(iqr) and iqr > 0:
            standardized.append((values - median) / iqr)
    if len(standardized) < 2:
        raise ValueError(f"{site}: fewer than two usable HF206 sensors")
    matrix = pd.concat(standardized, axis=1)
    soil["n_swc_sensors"] = matrix.notna().sum(axis=1)
    soil["swc_index"] = matrix.median(axis=1, skipna=True)
    soil.loc[soil["n_swc_sensors"] < 2, "swc_index"] = np.nan
    return soil[["date", "swc_index", "n_swc_sensors"]]


def fixed_season_bounds(daily: pd.DataFrame) -> tuple[int, int]:
    """Climatological season bounds; they do not vary among years."""
    sos = int(round(numeric(daily, "sos").dropna().median()))
    eos = int(round(numeric(daily, "eos").dropna().median()))
    return sos, eos


def conductance_training_data(site: str, soil: pd.DataFrame) -> pd.DataFrame:
    """Invert PM for dry-canopy surface conductance at subdaily scale."""
    cfg = SITE_CONFIG[site]
    raw = pd.read_csv(cfg["flux"], low_memory=False)
    raw["timestamp"] = pd.to_datetime(raw["TIMESTAMP"], errors="coerce")
    raw["date"] = raw["timestamp"].dt.normalize()
    raw["doy"] = raw["timestamp"].dt.dayofyear
    raw["Rg"] = numeric(raw, "Rg")
    raw["Tair"] = numeric(raw, "Tair")
    raw["RH"] = numeric(raw, "RH")
    raw["Rn"] = numeric(raw, "NETRAD")
    raw["LE_measured"] = numeric(raw, "LE")
    raw["H_measured"] = numeric(raw, "H")
    raw["wind"] = numeric(raw, "WS")
    raw["pressure_kPa"] = numeric(raw, "PA")

    # Recalculate VPD from T and RH to avoid the documented HEM 2023 unit
    # discontinuity and to use one equation at both sites.
    es_kpa = 0.6108 * np.exp(17.27 * raw["Tair"] / (raw["Tair"] + 237.3))
    raw["VPD"] = es_kpa * (1.0 - raw["RH"].clip(0.0, 100.0) / 100.0)

    daily = pd.read_csv(cfg["daily"], low_memory=False)
    daily["date"] = pd.to_datetime(daily["TIMESTAMP"], errors="coerce").dt.normalize()
    sos, eos = fixed_season_bounds(daily)
    selected = pd.DataFrame({
        "date": daily["date"],
        "year": daily["date"].dt.year,
        "doy": daily["date"].dt.dayofyear,
        "ET": numeric(daily, "ET"),
        "prec": numeric(daily, "prec") if "prec" in daily.columns else np.nan,
    })
    if selected["prec"].isna().all():
        climate_path = REPO / "data" / "source" / "climate" / "hf300-05-daily-m.csv"
        climate = pd.read_csv(climate_path, low_memory=False)
        climate["date"] = pd.to_datetime(climate["date"], errors="coerce").dt.normalize()
        climate = climate[["date", "prec"]]
        climate["prec"] = numeric(climate, "prec")
        selected = selected.drop(columns="prec").merge(
            climate, on="date", how="left", validate="one_to_one"
        )
    selected["prec_previous"] = selected["prec"].shift(1)
    selected["fixed_growing_season"] = selected["doy"].between(sos, eos)

    joined = (
        raw.merge(selected, on="date", how="inner", validate="many_to_one")
        .merge(soil, on="date", how="inner", validate="many_to_one")
    )
    joined = joined[
        joined["swc_index"].notna()
        & joined["Rg"].ge(50.0)
        & joined["Rn"].gt(20.0)
        & joined["LE_measured"].gt(20.0)
        & joined["VPD"].gt(0.05)
        & joined["wind"].gt(0.2)
        & joined["pressure_kPa"].between(80.0, 110.0)
        & joined["prec"].fillna(0.0).eq(0.0)
        & joined["prec_previous"].fillna(0.0).eq(0.0)
    ].dropna(subset=["Rg", "Tair", "VPD", "Rn", "LE_measured", "wind",
                     "pressure_kPa", "H_measured"])

    # Apply the same multiplicative energy-balance closure concept used by the
    # conductance analysis, but estimate the site coefficient reproducibly.
    energy = joined["LE_measured"] + joined["H_measured"]
    closure_slope = float(np.sum(joined["Rn"] * energy) / np.sum(joined["Rn"] ** 2))
    if not 0.5 <= closure_slope <= 1.1:
        raise ValueError(f"{site}: implausible energy-balance slope {closure_slope}")
    joined["closure_slope"] = closure_slope
    joined["LE_corrected"] = joined["LE_measured"] / closure_slope

    temperature = joined["Tair"].to_numpy(dtype=float)
    pressure_pa = joined["pressure_kPa"].to_numpy(dtype=float) * 1000.0
    vpd_pa = joined["VPD"].to_numpy(dtype=float) * 1000.0
    delta = (
        4098.0 * (0.6108 * np.exp(17.27 * temperature / (temperature + 237.3)))
        / np.square(temperature + 237.3)
    ) * 1000.0
    gamma = CP_AIR_J_KG_K * pressure_pa / (EPSILON * LATENT_HEAT_J_KG)
    rho = pressure_pa / (R_DRY_AIR_J_KG_K * (temperature + 273.15))
    ra = aerodynamic_resistance(joined["wind"], cfg["measurement_height_m"])
    numerator = (
        delta * joined["Rn"].to_numpy(dtype=float)
        + rho * CP_AIR_J_KG_K * vpd_pa / ra
    )
    le = joined["LE_corrected"].to_numpy(dtype=float)
    rs = ra * ((numerator / le - delta) / gamma - 1.0)
    molar_volume = R_UNIVERSAL_J_MOL_K * (temperature + 273.15) / pressure_pa
    gc_mol = (1.0 / rs) / molar_volume
    joined["ra_s_m"] = ra
    joined["rs_inverted_s_m"] = rs
    joined["gc_obs"] = gc_mol
    joined = joined[
        np.isfinite(joined["gc_obs"])
        & joined["gc_obs"].between(0.02, 3.0)
        & joined["rs_inverted_s_m"].gt(0.0)
    ].copy()
    # PM inversion is unstable as inferred surface resistance approaches zero.
    # Match the established conductance workflow's symmetric 5th--95th
    # percentile quality screen before model calibration.
    lower_gc, upper_gc = joined["gc_obs"].quantile([0.05, 0.95])
    joined = joined[joined["gc_obs"].between(lower_gc, upper_gc)].copy()
    joined["gc_screen_lower"] = lower_gc
    joined["gc_screen_upper"] = upper_gc

    if joined["date"].nunique() < 150:
        raise ValueError(
            f"{site}: only {joined['date'].nunique()} conductance calibration days"
        )
    return joined.sort_values("timestamp").reset_index(drop=True)


def response_functions(parameters, data, swc_dry, swc_wet, potential=False):
    """Return model conductance and its dimensionless response functions."""
    g_base, g_amplitude, r50, k_vpd, leaf_off = parameters
    radiation = np.asarray(data["Rg"], dtype=float)
    temperature = np.asarray(data["Tair"], dtype=float)
    vpd = np.asarray(data["VPD"], dtype=float)
    in_season = np.asarray(data["fixed_growing_season"], dtype=bool)

    # Stewart radiation response, normalized to one at 1000 W m-2.
    f_r = np.clip((1.0 + r50 / 1000.0) * radiation / (radiation + r50), 0.0, 1.2)
    f_t = np.ones(len(data), dtype=float)
    f_d = np.exp(-k_vpd * np.clip(vpd, 0.0, None))
    f_season = np.where(in_season, 1.0, leaf_off)
    if potential:
        f_swc = np.ones(len(data), dtype=float)
    else:
        swc = np.asarray(data["swc_index"], dtype=float)
        f_swc = np.clip((swc - swc_dry) / (swc_wet - swc_dry), 0.05, 1.0)
    gc = g_base + g_amplitude * f_r * f_t * f_d * f_season * f_swc
    return gc, {"f_R": f_r, "f_T": f_t, "f_D": f_d,
                "f_season": f_season, "f_SWC": f_swc}


def fit_jarvis_model(site: str, data: pd.DataFrame):
    """Fit wet-soil fixed parameters and validate on alternate dates."""
    swc_dry = float(data["swc_index"].quantile(0.10))
    swc_wet = float(data["swc_index"].quantile(0.75))
    if swc_wet <= swc_dry:
        raise ValueError(f"{site}: invalid soil-moisture response thresholds")

    # Moisture-sufficient calibration: the soil-water response is fixed at 1.
    # This avoids estimating potential conductance from moisture-stressed days.
    data = data[data["swc_index"].ge(swc_wet)].copy().reset_index(drop=True)
    if data["date"].nunique() < 100:
        raise ValueError(f"{site}: insufficient upper-quartile soil-moisture days")

    ordinal = data["date"].map(pd.Timestamp.toordinal)
    train = ordinal.mod(2).eq(0).to_numpy()
    test = ~train
    lower = np.array([0.0, 0.0, 0.1, 0.0, 0.03])
    upper = np.array([1.5, 4.0, 1200.0, 3.0, 1.00])
    initial = np.array([0.10, 1.10, 120.0, 0.25, 0.50])

    def residual(parameters):
        predicted, _ = response_functions(
            parameters, data.loc[train], swc_dry, swc_wet, potential=True
        )
        observed = data.loc[train, "gc_obs"].to_numpy(dtype=float)
        return predicted - observed

    result = least_squares(
        residual, initial, bounds=(lower, upper), loss="linear",
        max_nfev=30_000,
    )
    if not result.success:
        raise RuntimeError(f"{site}: Jarvis fit failed: {result.message}")

    parameters = result.x
    all_predicted, factors = response_functions(
        parameters, data, swc_dry, swc_wet, potential=True
    )
    data = data.copy()
    data["gc_fitted"] = all_predicted
    for name, values in factors.items():
        data[name] = values

    temperature = data["Tair"].to_numpy(dtype=float)
    pressure_pa = data["pressure_kPa"].to_numpy(dtype=float) * 1000.0
    vpd_pa = data["VPD"].to_numpy(dtype=float) * 1000.0
    delta = (
        4098.0 * (0.6108 * np.exp(17.27 * temperature / (temperature + 237.3)))
        / np.square(temperature + 237.3)
    ) * 1000.0
    gamma = CP_AIR_J_KG_K * pressure_pa / (EPSILON * LATENT_HEAT_J_KG)
    rho = pressure_pa / (R_DRY_AIR_J_KG_K * (temperature + 273.15))
    molar_volume = R_UNIVERSAL_J_MOL_K * (temperature + 273.15) / pressure_pa
    fitted_rs = 1.0 / np.clip(data["gc_fitted"].to_numpy() * molar_volume, 1e-8, None)
    ra = data["ra_s_m"].to_numpy(dtype=float)
    data["LE_fitted"] = (
        delta * data["Rn"].to_numpy(dtype=float) + rho * CP_AIR_J_KG_K * vpd_pa / ra
    ) / (delta + gamma * (1.0 + fitted_rs / ra))

    def metrics(mask):
        observed = data.loc[mask, "gc_obs"].to_numpy(dtype=float)
        predicted = data.loc[mask, "gc_fitted"].to_numpy(dtype=float)
        residuals = observed - predicted
        sst = np.sum(np.square(observed - observed.mean()))
        r2 = 1.0 - np.sum(np.square(residuals)) / sst if sst > 0 else np.nan
        observed_le = data.loc[mask, "LE_corrected"].to_numpy(dtype=float)
        predicted_le = data.loc[mask, "LE_fitted"].to_numpy(dtype=float)
        le_residual = observed_le - predicted_le
        le_sst = np.sum(np.square(observed_le - observed_le.mean()))
        le_r2 = (1.0 - np.sum(np.square(le_residual)) / le_sst
                 if le_sst > 0 else np.nan)
        return {
            "n": int(mask.sum()),
            "r2": float(r2),
            "rmse": float(np.sqrt(np.mean(np.square(residuals)))),
            "mae": float(np.mean(np.abs(residuals))),
            "bias": float(np.mean(predicted - observed)),
            "le_r2": float(le_r2),
            "le_rmse_W_m2": float(np.sqrt(np.mean(np.square(le_residual)))),
            "le_bias_W_m2": float(np.mean(predicted_le - observed_le)),
            "le_total_ratio": float(np.sum(predicted_le) / np.sum(observed_le)),
        }

    names = ["g_base_mol_m2_s", "g_amplitude_mol_m2_s", "radiation_c_W_m2",
             "kVPD_kPa_inv", "leaf_off_factor"]
    parameter_table = pd.DataFrame({"parameter": names, "value": parameters})
    parameter_table = pd.concat([
        parameter_table,
        pd.DataFrame({
            "parameter": ["SWC_dry_q10_index", "SWC_wet_q75_index",
                          "fixed_SOS_DOY", "fixed_EOS_DOY"],
            "value": [swc_dry, swc_wet, 0, 0],
        })
    ], ignore_index=True)
    return data, parameters, swc_dry, swc_wet, metrics(train), metrics(test), parameter_table


def aerodynamic_resistance(wind_speed, measurement_height_m):
    """Neutral aerodynamic resistance for the fixed forest geometry."""
    displacement = 0.67 * CANOPY_HEIGHT_M
    z0_momentum = 0.123 * CANOPY_HEIGHT_M
    z0_heat = 0.10 * z0_momentum
    factor = (
        math.log((measurement_height_m - displacement) / z0_momentum)
        * math.log((measurement_height_m - displacement) / z0_heat)
        / VON_KARMAN**2
    )
    wind = np.clip(np.asarray(wind_speed, dtype=float), 0.2, None)
    return factor / wind


def fill_subdaily_driver(values, timestamp, name):
    """Fill small driver gaps without using ET or conductance information."""
    x = pd.to_numeric(values, errors="coerce").copy()
    missing_before = int(x.isna().sum())
    x = x.interpolate(limit=6, limit_direction="both", limit_area="inside")
    keys = [timestamp.dt.month, timestamp.dt.hour, timestamp.dt.minute]
    x = x.fillna(x.groupby(keys).transform("median")).fillna(x.median())
    if x.isna().any():
        raise ValueError(f"Could not fill {name}")
    return x, missing_before


def load_subdaily_pet_drivers(site: str):
    """Load a continuous native-resolution meteorological record."""
    cfg = SITE_CONFIG[site]
    raw = pd.read_csv(cfg["subdaily_pet"], low_memory=False)
    if site == "EMS":
        timestamp = pd.to_datetime(raw["TIMESTAMP"], errors="coerce")
        columns = {
            "Rg": "Rg", "Tair": "Tair", "VPD": "VPD",
            "wind": "WS", "pressure_kPa": "PA", "Rn": "NETRAD",
        }
        interval_seconds = 3600.0
        frame = pd.DataFrame({"timestamp": timestamp})
        for target, source in columns.items():
            frame[target] = numeric(raw, source)
    else:
        timestamp = pd.to_datetime(raw["TIMESTAMP"], errors="coerce")
        frame = pd.DataFrame({
            "timestamp": timestamp,
            "Rg": numeric(raw, "Rg"),
            "Tair": numeric(raw, "Tair"),
            "wind": numeric(raw, "WS"),
            "pressure_kPa": numeric(raw, "PA"),
            "Rn": numeric(raw, "NETRAD"),
        })
        # The stored HEM VPD has a factor-of-ten discontinuity in 2023.
        # Recalculate it uniformly from complete Tair and RH in the source file.
        rh = numeric(raw, "RH").clip(0.0, 100.0)
        es = 0.6108 * np.exp(17.27 * frame["Tair"] / (frame["Tair"] + 237.3))
        frame["VPD"] = es * (1.0 - rh / 100.0)
        interval_seconds = 1800.0

    frame = frame.dropna(subset=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp")
    frequency = "1h" if site == "EMS" else "30min"
    complete_index = pd.date_range(frame["timestamp"].min(), frame["timestamp"].max(), freq=frequency)
    frame = frame.set_index("timestamp").reindex(complete_index).rename_axis("timestamp").reset_index()

    fill_counts = {}
    for column in ("Rg", "Tair", "VPD", "wind", "pressure_kPa", "Rn"):
        frame[column], fill_counts[column] = fill_subdaily_driver(
            frame[column], frame["timestamp"], f"{site} {column}"
        )
    frame["Rg"] = frame["Rg"].clip(lower=0.0)
    frame["VPD"] = frame["VPD"].clip(lower=0.0)
    frame["wind"] = frame["wind"].clip(lower=0.0)
    print(
        f"{site} subdaily PET drivers: {len(frame):,} intervals at "
        f"{interval_seconds / 60:.0f} min; filled values "
        + ", ".join(f"{key}={value}" for key, value in fill_counts.items())
    )
    return frame, interval_seconds


def calculate_forest_pet(site: str, parameters, swc_dry, swc_wet, soil):
    """Calculate PET subdaily, then aggregate it to the daily ET record."""
    cfg = SITE_CONFIG[site]
    daily = pd.read_csv(cfg["daily"], low_memory=False)
    daily["date"] = pd.to_datetime(daily["TIMESTAMP"], errors="coerce").dt.normalize()
    daily["water_year"] = numeric(daily, "water_year")
    daily = daily[daily["water_year"].between(
        cfg["first_year"], cfg["last_year"]
    )].copy()
    sos_fixed, eos_fixed = fixed_season_bounds(daily)

    subdaily, interval_seconds = load_subdaily_pet_drivers(site)
    subdaily["date"] = subdaily["timestamp"].dt.normalize()
    subdaily = subdaily[subdaily["date"].isin(daily["date"])].copy()
    subdaily["fixed_growing_season"] = subdaily["timestamp"].dt.dayofyear.between(
        sos_fixed, eos_fixed
    )
    model_data = subdaily[["Rg", "Tair", "VPD", "fixed_growing_season"]].copy()
    model_data["swc_index"] = np.nan
    gc_potential_mol, factors = response_functions(
        parameters, model_data, swc_dry, swc_wet, potential=True
    )

    temperature = subdaily["Tair"].to_numpy(dtype=float)
    pressure_pa = subdaily["pressure_kPa"].to_numpy(dtype=float) * 1000.0
    vpd_pa = subdaily["VPD"].to_numpy(dtype=float) * 1000.0
    rn = subdaily["Rn"].to_numpy(dtype=float)
    delta = (
        4098.0 * (0.6108 * np.exp(17.27 * temperature / (temperature + 237.3)))
        / np.square(temperature + 237.3)
    ) * 1000.0
    gamma = CP_AIR_J_KG_K * pressure_pa / (EPSILON * LATENT_HEAT_J_KG)
    rho = pressure_pa / (R_DRY_AIR_J_KG_K * (temperature + 273.15))
    ra = aerodynamic_resistance(subdaily["wind"], cfg["measurement_height_m"])
    molar_volume = R_UNIVERSAL_J_MOL_K * (temperature + 273.15) / pressure_pa
    gc_potential_ms = np.clip(gc_potential_mol * molar_volume, 1e-8, None)
    rs_potential = 1.0 / gc_potential_ms
    numerator = delta * rn + rho * CP_AIR_J_KG_K * vpd_pa / ra
    denominator = delta + gamma * (1.0 + rs_potential / ra)
    lambda_e = np.clip(numerator / denominator, 0.0, None)
    pet_interval = lambda_e * interval_seconds / LATENT_HEAT_J_KG

    subdaily["forest_PET_interval_mm"] = pet_interval
    subdaily["gc_potential_mol_m2_s"] = gc_potential_mol
    subdaily["gc_potential_m_s"] = gc_potential_ms
    subdaily["ra_s_m"] = ra
    subdaily["rs_potential_s_m"] = rs_potential
    for name, values in factors.items():
        subdaily[name] = values

    pet_daily = subdaily.groupby("date", as_index=False).agg(
        forest_PET_mm=("forest_PET_interval_mm", "sum"),
        n_intervals=("timestamp", "size"),
        gc_potential_mol_m2_s=("gc_potential_mol_m2_s", "median"),
        gc_potential_m_s=("gc_potential_m_s", "median"),
        ra_s_m=("ra_s_m", "median"),
        rs_potential_s_m=("rs_potential_s_m", "median"),
        f_R=("f_R", "mean"), f_T=("f_T", "mean"),
        f_D=("f_D", "mean"), f_season=("f_season", "mean"),
        f_SWC_potential=("f_SWC", "mean"),
    )
    expected = int(SECONDS_PER_DAY / interval_seconds)
    incomplete = int(pet_daily["n_intervals"].ne(expected).sum())
    if incomplete:
        raise ValueError(f"{site}: {incomplete} incomplete PET days after reindexing")

    metadata = daily[["date", "water_year", "DoY", "sos", "eos", "ET"]].copy()
    out = metadata.merge(pet_daily, on="date", how="inner", validate="one_to_one")
    out.insert(0, "site", site)
    out["season"] = np.where(
        numeric(out, "DoY").between(numeric(out, "sos"), numeric(out, "eos")),
        "GS", "NGS"
    )
    out["ET_mm"] = numeric(out, "ET")
    out = out.drop(columns="ET")
    required = ["forest_PET_mm", "ET_mm", "gc_potential_mol_m2_s"]
    if out[required].isna().any().any() or len(out) != len(daily):
        raise ValueError(f"{site}: incomplete daily aggregation from subdaily PET")
    return out


def summarize_site(data: pd.DataFrame):
    rows = []
    products = {}
    for season in ("Annual", "GS", "NGS"):
        sample = data if season == "Annual" else data[data["season"].eq(season)]
        annual = sample.groupby("water_year", as_index=False).agg(
            PET_mm=("forest_PET_mm", "sum"),
            ET_mm=("ET_mm", "sum"),
            n_days=("date", "size"),
        )
        products[season] = annual
        for variable, column in (("PET", "PET_mm"), ("ET", "ET_mm")):
            tau, p_value = original_mann_kendall(annual[column])
            slope = float(theilslopes(annual[column], annual["water_year"])[0])
            rows.append({
                "site": data["site"].iat[0], "season": season,
                "variable": variable, "mean_mm": annual[column].mean(),
                "minimum_mm": annual[column].min(),
                "maximum_mm": annual[column].max(), "kendall_tau": tau,
                "kendall_p": p_value, "theil_sen_slope_mm_yr": slope,
                "years": len(annual),
            })
            print(
                f"{data['site'].iat[0]} {season:6s} {variable}: "
                f"mean={annual[column].mean():.1f}, "
                f"range={annual[column].min():.1f}-{annual[column].max():.1f}, "
                f"tau={tau:+.3f}, p={p_value:.4g}, slope={slope:+.2f} mm yr-1"
            )
    return products, rows


def format_p(value: float) -> str:
    if value < 0.001:
        return "p<0.001"
    if value < 0.01:
        return f"p={value:.3f}"
    return f"p={value:.2f}"


def add_time_series_panel(ax, annual, season, title, panel):
    markers = {
        "Annual": {"PET": "*", "ET": "<"},
        "GS": {"PET": "X", "ET": "H"},
        "NGS": {"PET": "d", "ET": "v"},
    }
    colors = {"PET": "#ef7f7d", "ET": "#0b28e8"}
    for variable, column in (("PET", "PET_mm"), ("ET", "ET_mm")):
        tau, p_value = original_mann_kendall(annual[column])
        slope = float(theilslopes(annual[column], annual["water_year"])[0])
        ax.scatter(
            annual["water_year"], annual[column], s=125,
            marker=markers[season][variable], color=colors[variable],
            label=f"{variable}, {format_p(p_value)}, slope={slope:+.2f}", zorder=3,
        )
        fit = np.polyfit(annual["water_year"], annual[column], 1)
        ax.plot(
            annual["water_year"], np.polyval(fit, annual["water_year"]),
            color=colors[variable], linewidth=3.0,
            linestyle="-" if variable == "PET" else "--",
        )
    ax.set_title(title, fontsize=18, pad=11, linespacing=1.05)
    ax.set_xlabel("Water Year", fontsize=19, labelpad=8)
    ax.tick_params(axis="both", labelsize=16, width=1.7, length=7)
    ax.legend(loc="upper left", fontsize=14, frameon=True, labelspacing=0.25)
    ax.text(-0.16, 1.11, panel, transform=ax.transAxes, fontsize=27,
            fontweight="bold", va="top")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def make_manuscript_figure(products):
    fig, axes = plt.subplots(2, 3, figsize=(19, 14))
    seasons = ["Annual", "GS", "NGS"]
    season_titles = ["Annual", "Growing Season", "Non-Growing Season"]
    panels = list("abcdef")
    index = 0
    for row, site in enumerate(("EMS", "HEM")):
        for column, (season, season_title) in enumerate(zip(seasons, season_titles)):
            title = f"{SITE_CONFIG[site]['title']}\n{season_title}"
            add_time_series_panel(
                axes[row, column], products[site][season], season, title,
                f"{panels[index]})"
            )
            index += 1
    axes[0, 0].set_ylabel("Total annual PET or ET (mm)", fontsize=19, labelpad=9)
    axes[1, 0].set_ylabel("Total annual PET or ET (mm)", fontsize=19, labelpad=9)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.96,
                        wspace=0.25, hspace=0.40)
    path = OUT / "Figure5_forest_PET_measured_ET.png"
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return path


def make_validation_figure(fits):
    """Plot alternate-day latent-heat validation for the two sites."""
    # A 2:1 canvas keeps the square panels large without leaving a wide,
    # unused band between them.
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    for column, site in enumerate(("EMS", "HEM")):
        data, test_metrics = fits[site]
        ordinal = data["date"].map(pd.Timestamp.toordinal)
        test = ordinal.mod(2).ne(0)
        ax = axes[column]
        ax.scatter(data.loc[test, "LE_corrected"], data.loc[test, "LE_fitted"],
                   s=56, alpha=0.52, color="#27858a", edgecolors="none")
        limits = [0, max(data.loc[test, ["LE_corrected", "LE_fitted"]].max()) * 1.05]
        ax.plot(limits, limits, color="black", linewidth=2.7)
        ax.set(xlim=limits, ylim=limits,
               xlabel="Observed LE (W m$^{-2}$)",
               ylabel="Modeled LE (W m$^{-2}$)")
        ax.set_title(
            f"{site} alternate-day evaluation\n"
            f"R$^2$={test_metrics['le_r2']:.2f}, "
            f"modeled/observed total={test_metrics['le_total_ratio']:.2f}",
            fontsize=27,
            pad=18,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="0.90", linewidth=0.9)
        ax.tick_params(axis="both", which="major", labelsize=22,
                       width=1.8, length=8)
        ax.xaxis.label.set_size(26)
        ax.yaxis.label.set_size(26)
        ax.text(-0.13, 1.08, f"{'ab'[column]})", transform=ax.transAxes,
                fontsize=34, fontweight="bold", va="top")

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.84,
                        wspace=0.06)
    path = OUT / "Jarvis_model_validation_diagnostic.png"
    fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    products = {}
    summary_rows = []
    fit_diagnostics = {}

    for site in ("EMS", "HEM"):
        soil = robust_soil_index(site)
        calibration = conductance_training_data(site, soil)
        fitted, parameters, swc_dry, swc_wet, train_metrics, test_metrics, parameter_table = (
            fit_jarvis_model(site, calibration)
        )
        # Store actual fixed season bounds in the parameter table.
        daily_for_bounds = pd.read_csv(SITE_CONFIG[site]["daily"], low_memory=False)
        sos, eos = fixed_season_bounds(daily_for_bounds)
        parameter_table.loc[parameter_table["parameter"].eq("fixed_SOS_DOY"), "value"] = sos
        parameter_table.loc[parameter_table["parameter"].eq("fixed_EOS_DOY"), "value"] = eos
        parameter_table.insert(0, "site", site)
        parameter_table.to_csv(OUT / f"{site}_Jarvis_parameters.csv", index=False)
        fitted.to_csv(OUT / f"{site}_Jarvis_calibration_daily.csv", index=False)

        print(f"\n{site} Jarvis--Stewart calibration")
        print(f"  moisture-sufficient dry-canopy records: {len(fitted):,} across "
              f"{fitted['date'].nunique():,} days "
              f"({fitted['date'].min().date()} to {fitted['date'].max().date()})")
        for row in parameter_table.itertuples(index=False):
            print(f"  {row.parameter}: {row.value:.5g}")
        print(f"  training: n={train_metrics['n']}, R2={train_metrics['r2']:.3f}, "
              f"RMSE={train_metrics['rmse']:.3f}, bias={train_metrics['bias']:+.3f}")
        print(f"  validation: n={test_metrics['n']}, R2={test_metrics['r2']:.3f}, "
              f"RMSE={test_metrics['rmse']:.3f}, bias={test_metrics['bias']:+.3f}")
        print(f"  validation latent heat: R2={test_metrics['le_r2']:.3f}, "
              f"RMSE={test_metrics['le_rmse_W_m2']:.1f} W m-2, "
              f"bias={test_metrics['le_bias_W_m2']:+.1f} W m-2, "
              f"total ratio={test_metrics['le_total_ratio']:.3f}")

        final = calculate_forest_pet(site, parameters, swc_dry, swc_wet, soil)
        final.to_csv(OUT / f"{site}_daily_forest_PET.csv", index=False)
        site_products, rows = summarize_site(final)
        products[site] = site_products
        summary_rows.extend(rows)
        fit_diagnostics[site] = (fitted, test_metrics)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "forest_PET_trend_summary.csv", index=False)
    manuscript_figure = make_manuscript_figure(products)
    validation_figure = make_validation_figure(fit_diagnostics)
    print(f"\nCandidate manuscript figure: {manuscript_figure}")
    print(f"Model validation diagnostic: {validation_figure}")
    print(f"All tables: {OUT}")
    print("Existing PET and conductance scripts and outputs were not changed.")


if __name__ == "__main__":
    main()
