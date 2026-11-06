"""Audit soil-moisture data for the EMS and HEM conductance sensitivity test.

Run this before choosing or combining any soil-water record. The script reads
the downloaded Harvard Forest source files unchanged, identifies mean SWC/VWC
variables, reports date ranges, units inferred from names/value scales, annual
coverage, and overlap with the final paired ET- versus T-derived conductance
records. It does not alter source data or fit a soil-moisture model.
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd


def locate_repository():
    candidates = []
    if "__file__" in globals():
        path = Path(__file__).resolve()
        candidates.extend([path.parent, *path.parents])
    current = Path.cwd().resolve()
    candidates.extend([current, *current.parents])
    candidates.append(Path(
        r"G:\My Drive\ammara_MD\a_harvard\Project\HF_paper"
        r"\harvard-forest-et-dynamics"
    ))
    checked = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.add(candidate)
        if (candidate / "data" / "source").is_dir() and (candidate / "outputs").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate harvard-forest-et-dynamics")


REPO = locate_repository()
ARCHIVE = REPO / "data" / "source" / "harvard_forest_archive_download_2026-10-22"
COMPARISON = REPO / "outputs" / "et_partitioning" / "conductance_comparison"
OUT = REPO / "outputs" / "soil_moisture"

SOURCES = [
    ("HEM", ARCHIVE / "HF206" / "hf206-03-HK-understory.csv", "continuous understory"),
    ("EMS", ARCHIVE / "HF206" / "hf206-05-EMS-understory.csv", "continuous understory"),
    ("EMS", ARCHIVE / "HF069" / "hf069-07-soil-moist.csv", "intermittent field/profile"),
    ("EMS", ARCHIVE / "HF069" / "hf069-18-swc-2007-2017.csv", "continuous profile"),
    ("EMS", ARCHIVE / "HF069" / "hf069-19-swc-since-2017.csv", "continuous profile"),
    ("HEM", ARCHIVE / "HF153" / "hf153-01-hem-swc-1998-2000.csv", "intermittent field"),
    ("HEM", ARCHIVE / "HF153" / "hf153-02-hem-swc-2001-2007.csv", "intermittent field"),
]

TIME_NAMES = ("Time_EST", "datetime", "date", "TIMESTAMP", "DateTime")


def is_soil_water_column(name):
    lower = name.lower()
    if any(token in lower for token in ("_max", "_min", "temp", "tsoil")):
        return False
    return bool(
        # Accept both numbered sensor names (for example, swc1_ave) and
        # depth-labelled names (for example, vwc_15_a).
        re.search(r"(^|_)(swc|vwc|gwc)\d*(_|$)", lower)
        or ("soil" in lower and any(token in lower for token in ("moist", "water")))
    )


def find_time_column(columns):
    for candidate in TIME_NAMES:
        if candidate in columns:
            return candidate
    lookup = {column.lower(): column for column in columns}
    for candidate in TIME_NAMES:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def infer_unit(column, values):
    lower = column.lower()
    if "m_3_m_3" in lower or "m3_m3" in lower:
        return "m3 m-3 (from header)"
    if lower == "gwc" or "grav" in lower:
        return "gravimetric fraction (from header)"
    median = values.median()
    p95 = values.quantile(0.95)
    if np.isfinite(p95) and p95 <= 1.5:
        return "fraction, likely m3 m-3"
    if np.isfinite(median) and median > 1.5:
        return "percent-scale SWC (inferred)"
    return "unknown"


def depth_hint(column):
    match = re.search(r"(15|35|40|50|90)\s*cm", column.lower())
    if match:
        return f"{match.group(1)} cm"
    match = re.search(r"(?:swc|vwc)[_]?([0-9]+)", column.lower())
    if match:
        return f"sensor/profile {match.group(1)}"
    return "not encoded"


def paired_dates(site):
    path = COMPARISON / f"{site}_paired_subdaily_conductance.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run 04_compare_et_vs_t_conductance.py first."
        )
    frame = pd.read_csv(path, usecols=["timestamp"], low_memory=False)
    timestamp = pd.to_datetime(frame["timestamp"], errors="coerce")
    return set(timestamp.dropna().dt.normalize().unique())


def read_source(site, path, kind, paired):
    if not path.exists():
        print(f"MISSING: {path}")
        return [], []
    header = pd.read_csv(path, nrows=0).columns.tolist()
    time_column = find_time_column(header)
    soil_columns = [column for column in header if is_soil_water_column(column)]
    print(f"\n[{site}] {path.name} ({kind})")
    if time_column is None:
        print("  No recognizable date/time column; skipped")
        return [], []
    if not soil_columns:
        print("  No mean SWC/VWC/GWC variables found; skipped")
        return [], []

    usecols = [time_column, *soil_columns]
    frame = pd.read_csv(
        path, usecols=usecols, low_memory=False, on_bad_lines="skip",
        na_values=["NA", "NaN", "nan", -9999, -9999.0, -6999, -6999.0],
    )
    timestamp = pd.to_datetime(frame[time_column], errors="coerce")
    frame["date"] = timestamp.dt.normalize()
    frame = frame[frame["date"].notna()].copy()
    summaries, annual_records = [], []

    for column in soil_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = pd.DataFrame({"date": frame["date"], "value": values}).dropna()
        if valid.empty:
            print(f"  {column}: no numeric observations")
            continue
        daily = valid.groupby("date", as_index=False)["value"].median()
        dates = set(daily["date"].unique())
        overlap_days = len(dates.intersection(paired))
        paired_coverage = 100.0 * overlap_days / len(paired) if paired else np.nan
        annual = daily.assign(year=daily["date"].dt.year).groupby("year").agg(
            days=("date", "nunique"), median=("value", "median"),
            p05=("value", lambda x: x.quantile(0.05)),
            p95=("value", lambda x: x.quantile(0.95)),
        ).reset_index()
        years_60 = int((annual["days"] >= 60).sum())
        unit = infer_unit(column, daily["value"])
        summary = {
            "site": site, "file": path.name, "kind": kind, "variable": column,
            "depth_hint": depth_hint(column), "unit_inference": unit,
            "first_date": daily["date"].min().date(),
            "last_date": daily["date"].max().date(),
            "daily_observations": len(daily), "years_with_60_days": years_60,
            "paired_overlap_days": overlap_days,
            "paired_date_coverage_percent": paired_coverage,
            "median": daily["value"].median(),
            "p05": daily["value"].quantile(0.05),
            "p95": daily["value"].quantile(0.95),
        }
        summaries.append(summary)
        for row in annual.itertuples(index=False):
            annual_records.append({
                "site": site, "file": path.name, "variable": column,
                "year": row.year, "days": row.days, "median": row.median,
                "p05": row.p05, "p95": row.p95,
            })
        print(
            f"  {column}: {daily['date'].min().date()} to {daily['date'].max().date()}, "
            f"{len(daily):,} days, {years_60} years with >=60 days; "
            f"paired overlap={overlap_days:,} days ({paired_coverage:.1f}%); "
            f"median={daily['value'].median():.4g}, "
            f"5--95%={daily['value'].quantile(.05):.4g}--"
            f"{daily['value'].quantile(.95):.4g}; {unit}; {depth_hint(column)}"
        )
    return summaries, annual_records


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paired = {site: paired_dates(site) for site in ("EMS", "HEM")}
    for site in paired:
        ordered = sorted(paired[site])
        print(
            f"{site} paired-conductance target: {len(ordered):,} unique dates, "
            f"{pd.Timestamp(ordered[0]).date()} to {pd.Timestamp(ordered[-1]).date()}"
        )

    summaries, annual_records = [], []
    for site, path, kind in SOURCES:
        source_summaries, source_annual = read_source(
            site, path, kind, paired[site]
        )
        summaries.extend(source_summaries)
        annual_records.extend(source_annual)

    summary = pd.DataFrame(summaries)
    annual = pd.DataFrame(annual_records)
    summary.to_csv(OUT / "soil_moisture_audit_summary.csv", index=False)
    annual.to_csv(OUT / "soil_moisture_annual_coverage.csv", index=False)

    print("\nTOP CONTINUOUS CANDIDATES BY SITE")
    for site in ("EMS", "HEM"):
        ranked = summary[(summary["site"] == site) & summary["kind"].str.contains("continuous")]
        ranked = ranked.sort_values(
            ["paired_overlap_days", "years_with_60_days", "daily_observations"],
            ascending=False,
        ).head(8)
        print(f"\n{site}")
        if ranked.empty:
            print("  No continuous candidate found")
        else:
            print(ranked[[
                "file", "variable", "depth_hint", "first_date", "last_date",
                "paired_overlap_days", "paired_date_coverage_percent",
                "years_with_60_days", "unit_inference",
            ]].to_string(index=False))

    print(f"\nAudit outputs written to: {OUT}")
    print("No source or processed data files were changed.")


if __name__ == "__main__":
    main()
