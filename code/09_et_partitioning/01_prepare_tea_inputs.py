"""Step 1: prepare and cross-check subdaily inputs for Nelson TEA.

This script never modifies the manuscript source files. It reads the final
gap-filled flux products used to create EMS_daily.csv and HEM_daily.csv,
creates consistently named TEA input tables, and verifies that aggregated
subdaily ET and GPP reproduce the existing daily products.

Run from anywhere:
    python code/09_et_partitioning/01_prepare_tea_inputs.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
PROJECT = REPO.parent

EMS_SOURCE = PROJECT / "EMS_xg_Rd_fill.csv"
HEM_SOURCE = PROJECT / "results" / "HEM_ML_fill_2023.csv"
HEM_FLUX = REPO / "data" / "processed" / "flux" / "gaps_ML_HEM_2023.csv"
EMS_DAILY = REPO / "data" / "processed" / "daily" / "EMS_daily.csv"
HEM_DAILY = REPO / "data" / "processed" / "daily" / "HEM_daily.csv"
PRECIP = REPO / "data" / "source" / "climate" / "hf300-05-daily-m.csv"
OUTPUT = REPO / "data" / "processed" / "et_partitioning"


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def latent_heat_j_kg(tair_c: pd.Series) -> pd.Series:
    """Temperature-dependent latent heat of vaporization, J kg-1."""
    return 3_149_000.0 - 2_370.0 * (tair_c + 273.15)


def relative_humidity(tair_c: pd.Series, vpd_kpa: pd.Series) -> pd.Series:
    """Derive RH (%) only where the final source file provides no RH series."""
    saturation_kpa = 0.6108 * np.exp(17.27 * tair_c / (tair_c + 237.3))
    return (100.0 * (1.0 - vpd_kpa / saturation_kpa)).clip(0.0, 100.0)


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        raise KeyError(f"Missing required column: {column}")
    return pd.to_numeric(df[column], errors="coerce")


def precipitation() -> pd.DataFrame:
    p = pd.read_csv(PRECIP, low_memory=False)
    p["date"] = pd.to_datetime(p["date"], errors="coerce").dt.normalize()
    p["precip_mm_day"] = pd.to_numeric(p["prec"], errors="coerce")
    return p[["date", "precip_mm_day"]]


def prepare_ems(p: pd.DataFrame) -> pd.DataFrame:
    src = pd.read_csv(EMS_SOURCE, low_memory=False)
    timestamp = pd.to_datetime(src["TIMESTAMP"], errors="coerce")
    tair = numeric(src, "Tair_fill")
    le = numeric(src, "LE_xg_Rd")
    # Track the provenance of the final LE value. LE_Rd retains observed LE and
    # fills its gaps with REddyProc; LE_xg_Rd fills only the remaining gaps.
    le_source = np.select(
        [numeric(src, "LE_gaps").notna(), numeric(src, "LE_Rd").notna()],
        ["observed", "REddyProc"],
        default="XGBoost",
    )
    gpp = -numeric(src, "gee.e.6mol.m2.s")  # source convention: uptake is negative
    out = pd.DataFrame(
        {
            "timestamp": timestamp,
            "site": "EMS",
            "ET_mm": le * 3600.0 / latent_heat_j_kg(tair),
            "GPP_umol_m2_s": gpp,
            "Rg_W_m2": numeric(src, "Rg_fill"),
            "Tair_C": tair,
            "VPD_kPa": numeric(src, "VPD_fill"),
            # This is the completed RH series used in the manuscript workflow.
            "RH_percent": numeric(src, "RH_fill"),
            # Wind is a Nelson TEA predictor. WS_gaps is the exact wind series
            # used by the final manuscript XGBoost workflow.
            "WS_m_s": numeric(src, "WS_gaps"),
            "LE_W_m2": le,
            "LE_source": le_source,
        }
    )
    out = out.loc[out["timestamp"].dt.year.between(1992, 2023)].copy()
    out["date"] = out["timestamp"].dt.normalize()
    return out.merge(p, on="date", how="left")


def prepare_hem(p: pd.DataFrame) -> pd.DataFrame:
    src = pd.read_csv(HEM_SOURCE, low_memory=False)
    src["timestamp"] = pd.to_datetime(src["TIMESTAMP"], errors="coerce")
    met = pd.read_csv(HEM_FLUX, usecols=["TIMESTAMP", "RH"], low_memory=False)
    met["timestamp"] = pd.to_datetime(met["TIMESTAMP"], errors="coerce")
    met = met[["timestamp", "RH"]].drop_duplicates("timestamp")
    src = src.merge(met, on="timestamp", how="left", validate="one_to_one")
    timestamp = src["timestamp"]
    tair = numeric(src, "Tair")
    vpd = numeric(src, "VPD")
    le = numeric(src, "LE_fill")
    le_source = np.where(numeric(src, "LE").notna(), "observed", "XGBoost")
    out = pd.DataFrame(
        {
            "timestamp": timestamp,
            "site": "HEM",
            "ET_mm": le * 1800.0 / latent_heat_j_kg(tair),
            "GPP_umol_m2_s": numeric(src, "GPP"),
            "Rg_W_m2": numeric(src, "Rg"),
            "Tair_C": tair,
            "VPD_kPa": vpd,
            # RH comes from the archived HEM flux file used by the original
            # conductance workflow; use reconstruction only for any rare gap.
            "RH_percent": numeric(src, "RH").fillna(relative_humidity(tair, vpd)),
            "WS_m_s": numeric(src, "WS"),
            "LE_W_m2": le,
            "LE_source": le_source,
        }
    )
    out = out.loc[out["timestamp"].dt.year.between(2006, 2022)].copy()
    out["date"] = out["timestamp"].dt.normalize()
    return out.merge(p, on="date", how="left")


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    required = [
        "timestamp", "ET_mm", "GPP_umol_m2_s", "Rg_W_m2",
        "Tair_C", "VPD_kPa", "RH_percent", "WS_m_s", "precip_mm_day",
    ]
    result["all_required_present"] = result[required].notna().all(axis=1)
    result["daytime_positive_flux"] = (
        result["ET_mm"].gt(0)
        & result["GPP_umol_m2_s"].gt(0)
        & result["Rg_W_m2"].gt(10)
    )
    # Diagnostic only. The original TEA CSWI calculation will be implemented
    # in the partitioning step; this is not the final dry-surface filter.
    result["rain_day"] = result["precip_mm_day"].gt(0)
    result["candidate_training_record"] = (
        result["all_required_present"]
        & result["daytime_positive_flux"]
        & ~result["rain_day"]
    )
    result["not_xgboost_filled_LE"] = result["LE_source"].ne("XGBoost")
    result["candidate_training_no_xg_LE"] = (
        result["candidate_training_record"] & result["not_xgboost_filled_LE"]
    )
    return result


def compare_daily(subdaily: pd.DataFrame, daily_path: Path, seconds: int, site: str) -> None:
    daily = pd.read_csv(daily_path, low_memory=False)
    daily["date"] = pd.to_datetime(daily["TIMESTAMP"], errors="coerce").dt.normalize()
    daily_et = pd.to_numeric(daily["ET"], errors="coerce")
    daily_gpp = pd.to_numeric(daily["GPP_g"], errors="coerce")
    daily = pd.DataFrame({"date": daily["date"], "ET_reference": daily_et, "GPP_reference": daily_gpp})

    calc = subdaily.groupby("date", as_index=False).agg(
        ET_calculated=("ET_mm", lambda x: x.sum(min_count=1)),
        GPP_umol_sum=("GPP_umol_m2_s", lambda x: x.sum(min_count=1)),
    )
    calc["GPP_calculated"] = calc["GPP_umol_sum"] * 12e-6 * seconds
    merged = calc.merge(daily, on="date", how="inner")

    for variable in ("ET", "GPP"):
        calculated = merged[f"{variable}_calculated"]
        reference = merged[f"{variable}_reference"]
        valid = calculated.notna() & reference.notna()
        difference = calculated[valid] - reference[valid]
        correlation = calculated[valid].corr(reference[valid])
        print(
            f"{site} {variable}: n={valid.sum():,}, "
            f"MAE={difference.abs().mean():.6g}, "
            f"bias={difference.mean():.6g}, r={correlation:.6f}"
        )


def report(df: pd.DataFrame, site: str) -> None:
    print(f"\n{site} standardized input")
    print(f"  rows: {len(df):,}")
    print(f"  period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    for column in [
        "ET_mm", "GPP_umol_m2_s", "Rg_W_m2", "Tair_C",
        "VPD_kPa", "RH_percent", "WS_m_s", "precip_mm_day",
    ]:
        print(f"  {column:18s}: {100 * df[column].notna().mean():6.2f}% non-missing")
    print(f"  all required present : {100 * df['all_required_present'].mean():6.2f}%")
    print(f"  candidate training   : {100 * df['candidate_training_record'].mean():6.2f}%")
    print(f"  candidate, no XGB LE : {100 * df['candidate_training_no_xg_LE'].mean():6.2f}%")
    print("  LE provenance:")
    source_counts = df["LE_source"].value_counts(dropna=False)
    for source, count in source_counts.items():
        print(f"    {str(source):10s}: {count:8,d} ({100 * count / len(df):6.2f}%)")


def main() -> None:
    for path in (EMS_SOURCE, HEM_SOURCE, HEM_FLUX, EMS_DAILY, HEM_DAILY, PRECIP):
        require(path)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    p = precipitation()
    ems = add_flags(prepare_ems(p))
    hem = add_flags(prepare_hem(p))

    report(ems, "EMS")
    report(hem, "HEM")

    print("\nCross-check against existing manuscript daily files")
    compare_daily(ems, EMS_DAILY, seconds=3600, site="EMS")
    compare_daily(hem, HEM_DAILY, seconds=1800, site="HEM")

    ems_path = OUTPUT / "EMS_TEA_input_subdaily.csv"
    hem_path = OUTPUT / "HEM_TEA_input_subdaily.csv"
    ems.to_csv(ems_path, index=False)
    hem.to_csv(hem_path, index=False)
    print("\nWrote standardized inputs (source manuscript files unchanged):")
    print(f"  {ems_path}")
    print(f"  {hem_path}")
    print("\nPaste this complete terminal output back before running partitioning.")


if __name__ == "__main__":
    main()
