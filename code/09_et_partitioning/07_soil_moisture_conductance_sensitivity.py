"""Evaluate soil-moisture sensitivity of ET- and T-derived conductance.

Early field campaigns and later continuous sensors are kept as separate
records. They are never spliced, converted across source scales, interpolated,
or gap-filled. Each component is centered and divided by its own interquartile
range before components within a record are combined.

Continuous records are matched by date to the dry, high-radiation,
growing-season conductance observations created by
04_compare_et_vs_t_conductance.py. Historical field campaigns are compared
only at the annual growing-season scale and are explicitly exploratory.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t, theilslopes


def locate_repository():
    candidates = []
    if "__file__" in globals():
        here = Path(__file__).resolve()
        candidates.extend([here.parent, *here.parents])
    current = Path.cwd().resolve()
    candidates.extend([current, *current.parents])
    for candidate in dict.fromkeys(candidates):
        if (candidate / "data" / "processed" / "soil_moisture").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate harvard-forest-et-dynamics")


REPO = locate_repository()
ARCHIVE = REPO / "data" / "source" / "harvard_forest_archive_download_2026-10-22"
SOIL = REPO / "data" / "processed" / "soil_moisture"
COMPARISON = REPO / "outputs" / "et_partitioning" / "conductance_comparison"
OUT = REPO / "outputs" / "soil_moisture" / "conductance_sensitivity"

MIN_COMPONENTS = 1
MIN_DAYS_CONTINUOUS = 30
MIN_DATES_HISTORICAL = 3
MIN_YEARS_MODEL = 6
GROWING_MONTHS = (5, 6, 7, 8, 9)


def original_mann_kendall(values):
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 2:
        return np.nan, np.nan
    s = sum(np.sign(y[j] - y[:j]).sum() for j in range(1, n))
    _, counts = np.unique(y, return_counts=True)
    ties = counts[counts > 1]
    var_s = (n * (n - 1) * (2 * n + 5)
             - np.sum(ties * (ties - 1) * (2 * ties + 5))) / 18.0
    if var_s <= 0:
        return 0.0, 1.0
    z = ((s - 1) / np.sqrt(var_s) if s > 0 else
         (s + 1) / np.sqrt(var_s) if s < 0 else 0.0)
    tau = s / (0.5 * n * (n - 1))
    return float(tau), float(2.0 * norm.sf(abs(z)))


def mk_sen(year, values):
    frame = pd.DataFrame({"year": year, "value": values}).dropna()
    if len(frame) < 3:
        return np.nan, np.nan, np.nan
    tau, p_value = original_mann_kendall(frame["value"])
    slope = theilslopes(frame["value"], frame["year"])[0]
    return tau, p_value, float(slope)


def ols_hc3(y, predictors):
    """Return OLS coefficients and HC3 p values without statsmodels."""
    y = np.asarray(y, dtype=float)
    x_values = [np.asarray(value, dtype=float) for value in predictors.values()]
    valid = np.isfinite(y)
    for value in x_values:
        valid &= np.isfinite(value)
    y = y[valid]
    x_values = [value[valid] for value in x_values]
    names = ["intercept", *predictors.keys()]
    x = np.column_stack([np.ones(len(y)), *x_values])
    if len(y) <= x.shape[1]:
        raise ValueError("Not enough annual observations for model")
    xtx_inverse = np.linalg.pinv(x.T @ x)
    beta = xtx_inverse @ x.T @ y
    residual = y - x @ beta
    leverage = np.sum((x @ xtx_inverse) * x, axis=1)
    scaled = residual / np.clip(1.0 - leverage, 1e-8, None)
    covariance = xtx_inverse @ (x.T @ ((scaled ** 2)[:, None] * x)) @ xtx_inverse
    standard_error = np.sqrt(np.clip(np.diag(covariance), 0, None))
    dof = len(y) - x.shape[1]
    t_value = np.divide(beta, standard_error, out=np.full_like(beta, np.nan),
                        where=standard_error > 0)
    p_value = 2.0 * student_t.sf(np.abs(t_value), dof)
    sse = np.sum(residual ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    r_squared = 1.0 - sse / sst if sst > 0 else np.nan
    adjusted_r_squared = (1.0 - (1.0 - r_squared) * (len(y) - 1) / dof
                            if dof > 0 else np.nan)
    return {"n": len(y), "beta": dict(zip(names, beta)),
            "p": dict(zip(names, p_value)), "adjusted_r2": adjusted_r_squared}


def robust_record(site, record, source, kind, frame, date_column, components,
                  raw_unit, note):
    """Create one daily source-specific record without filling missing dates."""
    work = frame[[date_column, *components]].copy()
    work["date"] = pd.to_datetime(work[date_column], errors="coerce").dt.normalize()
    for column in components:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["date"])
    daily = work.groupby("date", as_index=False)[components].median()
    robust_columns, anomaly_columns = [], []
    for column in components:
        values = daily[column]
        median = values.median()
        mean = values.mean()
        iqr = values.quantile(0.75) - values.quantile(0.25)
        if not np.isfinite(iqr) or iqr <= 0:
            continue
        robust = f"{column}__robust"
        anomaly = f"{column}__anomaly"
        daily[robust] = (values - median) / iqr
        daily[anomaly] = values - mean
        robust_columns.append(robust)
        anomaly_columns.append(anomaly)
    if not robust_columns:
        raise ValueError(f"No variable component in {site} {record}")
    daily["n_components"] = daily[components].notna().sum(axis=1)
    daily["swc_raw_median"] = daily[components].median(axis=1, skipna=True)
    daily["swc_anomaly"] = daily[anomaly_columns].median(axis=1, skipna=True)
    daily["swc_index"] = daily[robust_columns].median(axis=1, skipna=True)
    daily.loc[daily["n_components"] < MIN_COMPONENTS,
              ["swc_raw_median", "swc_anomaly", "swc_index"]] = np.nan
    daily.insert(0, "site", site)
    daily.insert(1, "record", record)
    daily.insert(2, "source", source)
    daily.insert(3, "record_type", kind)
    daily["raw_unit"] = raw_unit
    daily["note"] = note
    return daily[["site", "record", "source", "record_type", "date",
                  "swc_raw_median", "swc_anomaly", "swc_index",
                  "n_components", "raw_unit", "note"]]


def read_csv(path, **kwargs):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, na_values=["NA", "NaN", "nan", -9999, -6999],
                       low_memory=False, **kwargs)


def build_records():
    records = []
    hf005 = read_csv(ARCHIVE / "HF005" / "hf005-01-trace-gas.csv")
    hf005 = hf005[pd.to_numeric(hf005["treatment"], errors="coerce").eq(1)].copy()
    hf005["component"] = "block_" + hf005["block"].astype("Int64").astype(str)
    wide = hf005.pivot_table(index="date", columns="component",
                             values="vol_h2o", aggfunc="median").reset_index()
    records.append(robust_record(
        "EMS", "EMS_HF005_unheated_controls", "HF005", "historical", wide,
        "date", [c for c in wide if c != "date"], "m3 m-3",
        "Unheated control plots; TDR volumetric soil water content."))

    hf006 = read_csv(ARCHIVE / "HF006" / "hf006-01-soil-respiration.csv")
    hf006 = hf006[hf006["site"].isin(["NWN", "SWF"])].copy()
    wide = hf006.pivot_table(index="date", columns="site", values="vsm",
                             aggfunc="median").reset_index()
    records.append(robust_record(
        "EMS", "EMS_HF006_well_drained_transects", "HF006", "historical", wide,
        "date", [c for c in wide if c != "date"], "archive-reported g g-1",
        "NWN and SWF only; ambiguous units; within-record index only."))

    hf069 = read_csv(ARCHIVE / "HF069" / "hf069-07-soil-moist.csv")
    hf069["plot"] = hf069["plot_probe"].astype(str).str.split(".").str[0]
    hf069 = hf069[hf069["plot"].isin(["B3", "B4", "C3", "D5", "E3", "G5"])]
    wide = hf069.pivot_table(index="date", columns="plot", values="wc",
                             aggfunc="median").reset_index()
    records.append(robust_record(
        "EMS", "EMS_HF069_unharvested_plots", "HF069", "historical", wide,
        "date", [c for c in wide if c != "date"], "source scale",
        "Unharvested plots only; independent field-campaign record."))

    ems = read_csv(SOIL / "EMS_daily_soil_moisture.csv")
    records.append(robust_record(
        "EMS", "EMS_tower_profiles_15cm", "AmeriFlux/HF069", "continuous", ems,
        "date", ["profile_a_15cm_m3_m3", "profile_b_15cm_m3_m3"], "m3 m-3",
        "Two tower profiles; duplicate sources de-duplicated upstream."))
    records.append(robust_record(
        "EMS", "EMS_HF206_upper15cm", "HF206", "continuous", ems, "date",
        [f"hf206_upper15_sensor{i}_m3_m3" for i in range(1, 5)], "m3 m-3",
        "Four independent upper-15-cm sensors."))

    hem = read_csv(SOIL / "HEM_daily_soil_moisture.csv")
    records.append(robust_record(
        "HEM", "HEM_HF153_field_VWC", "HF153", "historical", hem, "date",
        ["field_vwc_m3_m3"], "m3 m-3",
        "Field-sampled VWC; separate from continuous sensors."))
    records.append(robust_record(
        "HEM", "HEM_continuous_upper15cm", "AmeriFlux/HF206", "continuous", hem,
        "date", [f"upper15_sensor{i}_m3_m3" for i in range(1, 5)], "m3 m-3",
        "Four upper-15-cm sensors; duplicate sources de-duplicated upstream."))
    return records


def annual_soil_record(daily):
    growing = daily[daily["date"].dt.month.isin(GROWING_MONTHS)].copy()
    growing = growing.dropna(subset=["swc_index"])
    growing["year"] = growing["date"].dt.year
    identity = ["site", "record", "source", "record_type", "raw_unit", "note"]
    annual = growing.groupby([*identity, "year"], as_index=False, dropna=False).agg(
        swc_raw_median=("swc_raw_median", "median"),
        swc_anomaly=("swc_anomaly", "median"),
        swc_index=("swc_index", "median"),
        observation_dates=("date", "nunique"),
        median_components=("n_components", "median"))
    threshold = (MIN_DAYS_CONTINUOUS if daily["record_type"].iat[0] == "continuous"
                 else MIN_DATES_HISTORICAL)
    return annual[annual["observation_dates"] >= threshold].copy()


def daily_conductance(site):
    frame = read_csv(COMPARISON / f"{site}_paired_subdaily_conductance.csv",
                     usecols=["timestamp", "gc_ET", "gc_T"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["date"] = frame["timestamp"].dt.normalize()
    return frame.groupby("date", as_index=False).agg(
        gc_ET=("gc_ET", "median"), gc_T=("gc_T", "median"),
        n_subdaily=("timestamp", "size"))


def annual_conductance(site):
    return read_csv(COMPARISON / f"{site}_annual_median_conductance.csv",
                    usecols=["site", "year", "gc_ET", "gc_T", "n"])


def fit_sensitivity(frame, site, record, match_type, outcome):
    frame = frame.dropna(subset=["year", outcome, "swc_index"]).copy()
    row = {"site": site, "record": record, "match_type": match_type,
           "outcome": outcome, "n_years": len(frame),
           "first_year": int(frame["year"].min()),
           "last_year": int(frame["year"].max())}
    raw_tau, raw_p, raw_slope = mk_sen(frame["year"], frame[outcome])
    row.update(raw_mk_tau=raw_tau, raw_mk_p=raw_p, raw_sen_slope=raw_slope)
    if len(frame) < MIN_YEARS_MODEL:
        row["model_status"] = "descriptive_only_fewer_than_6_years"
        return row, frame
    year_centered = frame["year"] - frame["year"].mean()
    unadjusted = ols_hc3(frame[outcome], {"year": year_centered})
    adjusted = ols_hc3(frame[outcome],
                       {"year": year_centered, "swc_index": frame["swc_index"]})
    adjusted_column = f"{outcome}_adjusted_for_swc"
    frame = frame.copy()
    frame[adjusted_column] = (frame[outcome]
                              - adjusted["beta"]["swc_index"] * frame["swc_index"])
    adj_tau, adj_p, adj_slope = mk_sen(frame["year"], frame[adjusted_column])
    raw_ols, adjusted_ols = unadjusted["beta"]["year"], adjusted["beta"]["year"]
    change = 100.0 * (adjusted_ols / raw_ols - 1.0) if raw_ols != 0 else np.nan
    row.update(model_status="modeled", year_only_ols_slope=raw_ols,
               year_only_ols_hc3_p=unadjusted["p"]["year"],
               adjusted_ols_year_slope=adjusted_ols,
               adjusted_ols_year_hc3_p=adjusted["p"]["year"],
               adjusted_ols_swc_beta=adjusted["beta"]["swc_index"],
               adjusted_ols_swc_hc3_p=adjusted["p"]["swc_index"],
               adjusted_ols_r2=adjusted["adjusted_r2"],
               year_slope_change_percent=change,
               swc_adjusted_mk_tau=adj_tau, swc_adjusted_mk_p=adj_p,
               swc_adjusted_sen_slope=adj_slope)
    return row, frame


def source_trend_rows(annual):
    rows = []
    for keys, frame in annual.groupby(["site", "record", "source", "record_type"]):
        site, record, source, kind = keys
        tau, p_value, slope = mk_sen(frame["year"], frame["swc_index"])
        a_tau, a_p, a_slope = mk_sen(frame["year"], frame["swc_anomaly"])
        rows.append({"site": site, "record": record, "source": source,
                     "record_type": kind, "n_years": len(frame),
                     "first_year": int(frame["year"].min()),
                     "last_year": int(frame["year"].max()),
                     "index_mk_tau": tau, "index_mk_p": p_value,
                     "index_sen_slope_per_year": slope,
                     "anomaly_mk_tau": a_tau, "anomaly_mk_p": a_p,
                     "anomaly_sen_slope_per_year": a_slope})
    return rows


def make_source_figure(annual):
    fig, axes = plt.subplots(2, 1, figsize=(15, 11))
    for ax, site in zip(axes, ("EMS", "HEM")):
        for record, frame in annual[annual["site"] == site].groupby("record"):
            style = "-" if frame["record_type"].iat[0] == "continuous" else "--"
            ax.plot(frame["year"], frame["swc_index"], marker="o", linestyle=style,
                    linewidth=2.2, markersize=7, label=record.replace("_", " "))
        ax.axhline(0, color="0.5", linewidth=1)
        ax.set_title(f"{site}: independent growing-season soil-moisture records",
                     fontsize=18, fontweight="bold")
        ax.set_ylabel("Within-record soil-moisture index\n(median/IQR scale)", fontsize=15)
        ax.tick_params(labelsize=13)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=11, ncol=2)
    axes[-1].set_xlabel("Year", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "soil_moisture_source_periods.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def make_conductance_figure(model_frames):
    wanted = {"EMS_tower_profiles_15cm", "EMS_HF206_upper15cm",
              "HEM_continuous_upper15cm"}
    frames = [f for f in model_frames if f["record"].iat[0] in wanted and
              "gc_T_adjusted_for_swc" in f]
    if not frames:
        return
    fig, axes = plt.subplots(1, len(frames), figsize=(7 * len(frames), 6),
                             squeeze=False)
    colors = {"EMS": "#d62728", "HEM": "#218c8d"}
    for ax, frame in zip(axes[0], frames):
        site = frame["site"].iat[0]
        observed = frame["gc_T"]
        adjusted = frame["gc_T_adjusted_for_swc"]
        adjusted = adjusted - adjusted.mean() + observed.mean()
        ax.plot(frame["year"], observed, "o-", color=colors[site], linewidth=2.4,
                markersize=8, label="T-derived conductance")
        ax.plot(frame["year"], adjusted, "s--", color="#333333", linewidth=2.1,
                markersize=7, label="Adjusted to mean soil moisture")
        ax.set_title(frame["record"].iat[0].replace("_", " "), fontsize=15,
                     fontweight="bold")
        ax.set_xlabel("Year", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=11)
    axes[0, 0].set_ylabel(
        "Growing-season median $g_c$ (mol m$^{-2}$ s$^{-1}$)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "soil_moisture_adjusted_T_conductance.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    daily_records = build_records()
    annual_frames = [annual_soil_record(record) for record in daily_records]
    all_annual_soil = pd.concat(annual_frames, ignore_index=True).sort_values(
        ["site", "record", "year"])
    all_annual_soil.to_csv(OUT / "soil_moisture_source_annual.csv", index=False)
    trends = source_trend_rows(all_annual_soil)
    pd.DataFrame(trends).to_csv(OUT / "soil_moisture_source_trends.csv", index=False)

    statistics, model_frames = [], []
    matched_by_site = {"EMS": [], "HEM": []}
    for daily in daily_records:
        site, record = daily["site"].iat[0], daily["record"].iat[0]
        if daily["record_type"].iat[0] == "continuous":
            matched = daily_conductance(site).merge(
                daily[["date", "swc_raw_median", "swc_anomaly", "swc_index",
                       "n_components"]], on="date", how="inner", validate="one_to_one"
            ).dropna(subset=["gc_ET", "gc_T", "swc_index"])
            matched["year"] = matched["date"].dt.year
            annual = matched.groupby("year", as_index=False).agg(
                gc_ET=("gc_ET", "median"), gc_T=("gc_T", "median"),
                swc_index=("swc_index", "median"),
                swc_anomaly=("swc_anomaly", "median"),
                matched_days=("date", "nunique"),
                subdaily_records=("n_subdaily", "sum"),
                median_components=("n_components", "median"))
            annual = annual[annual["matched_days"] >= MIN_DAYS_CONTINUOUS].copy()
            matched.insert(0, "site", site)
            matched.insert(1, "record", record)
            matched_by_site[site].append(matched)
            match_type = "exact_date_continuous"
        else:
            soil_annual = annual_soil_record(daily)
            annual = annual_conductance(site).merge(
                soil_annual[["year", "swc_index", "swc_anomaly", "observation_dates"]],
                on="year", how="inner", validate="one_to_one").rename(
                    columns={"observation_dates": "matched_days"})
            match_type = "annual_historical_exploratory"
        if annual.empty:
            continue
        if "record" not in annual.columns:
            annual.insert(0, "record", record)
        if "site" not in annual.columns:
            annual.insert(0, "site", site)
        for outcome in ("gc_ET", "gc_T"):
            result, fitted = fit_sensitivity(annual.copy(), site, record,
                                             match_type, outcome)
            statistics.append(result)
            if outcome == "gc_T":
                model_frames.append(fitted)

    for site, frames in matched_by_site.items():
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(
                OUT / f"{site}_daily_matched_conductance_swc.csv", index=False)
    pd.concat(model_frames, ignore_index=True, sort=False).to_csv(
        OUT / "annual_conductance_soil_moisture.csv", index=False)
    pd.DataFrame(statistics).to_csv(
        OUT / "soil_moisture_sensitivity_statistics.csv", index=False)
    make_source_figure(all_annual_soil)
    make_conductance_figure(model_frames)

    print("\nSOURCE-SPECIFIC GROWING-SEASON SOIL-MOISTURE RECORDS")
    for row in trends:
        print(f"  {row['record']}: {row['n_years']} years "
              f"({row['first_year']}-{row['last_year']}), "
              f"tau={row['index_mk_tau']:+.3f}, p={row['index_mk_p']:.4g}, "
              f"Sen={row['index_sen_slope_per_year']:+.4f} index yr-1")
    print("\nCONDUCTANCE SENSITIVITY TO SOIL MOISTURE")
    for row in statistics:
        print(f"  {row['site']} | {row['record']} | {row['outcome']}: "
              f"{row['n_years']} years; {row['model_status']}")
        if row["model_status"] == "modeled":
            print(f"    raw Sen={row['raw_sen_slope']:+.5f}, p={row['raw_mk_p']:.4g}; "
                  f"adjusted OLS year={row['adjusted_ols_year_slope']:+.5f}, "
                  f"p={row['adjusted_ols_year_hc3_p']:.4g}; "
                  f"SWC p={row['adjusted_ols_swc_hc3_p']:.4g}; "
                  f"year-slope change={row['year_slope_change_percent']:+.1f}%")
    print(f"\nOutputs written to: {OUT}")
    print("Records were analyzed separately; no cross-source conversion, splicing, interpolation, or gap filling was used.")
    print("Historical annual comparisons are exploratory; exact-date continuous analyses are the primary tests.")


if __name__ == "__main__":
    main()
