"""Build de-duplicated daily soil-moisture files for EMS and HEM.

The AmeriFlux BASE exports and Harvard Forest archive contain overlapping
copies of the same soil-water measurements. This script converts all
volumetric soil water content to m3 m-3, aggregates subdaily values to daily
medians, and uses AmeriFlux as the primary value with Harvard Forest as a
fallback only when the AmeriFlux copy is missing. Duplicate observations are
never averaged or counted twice.

No temporal interpolation or statistical gap filling is performed. Missing
codes (-9999, -6999, NA, NaN) and values outside the physical source ranges
are converted to missing values. Dates with no soil-moisture observation from
any source are omitted; gaps in individual sensors remain blank.

EMS output includes:
  * Two four-depth profiles (15, 35/40, 50, and 90 cm), de-duplicated between
    AmeriFlux US-Ha1 and Harvard Forest HF069.
  * Four independent upper-15-cm HF206 sensors.

HEM output includes:
  * Four upper-15-cm sensors, de-duplicated between AmeriFlux US-Ha2
    SWC_PI_1--4 and Harvard Forest HF206.
  * Historical HF153 field-sampled volumetric and gravimetric water content,
    kept separate from the continuous sensor record.

The files stop at 2023-12-31, matching the manuscript data period. Short
AmeriFlux profiles beginning in late 2022/2023 are intentionally excluded
because their depth metadata are not encoded in the variable labels and they
cannot support the long-term analysis.
"""
from pathlib import Path

import numpy as np
import pandas as pd


END_DATE = pd.Timestamp("2023-12-31")
AMF_MISSING = [-9999, -9999.0, -6999, -6999.0]
HF_MISSING = ["NA", "NaN", "nan", -9999, -9999.0, -6999, -6999.0]


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
        if (candidate / "data" / "processed" / "flux").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate harvard-forest-et-dynamics")


REPO = locate_repository()
FLUX = REPO / "data" / "processed" / "flux"
ARCHIVE = REPO / "data" / "source" / "harvard_forest_archive_download_2026-10-22"
OUT = REPO / "data" / "processed" / "soil_moisture"


def numeric_clean(frame, columns, lower, upper, divisor=1.0):
    cleaned = pd.DataFrame(index=frame.index)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        values = values.where(values.between(lower, upper)) / divisor
        cleaned[column] = values
    return cleaned


def read_amf_daily(path, columns):
    frame = pd.read_csv(
        path, comment="#", usecols=["TIMESTAMP_START", *columns],
        na_values=AMF_MISSING, low_memory=False,
    )
    stamp = pd.to_datetime(
        frame["TIMESTAMP_START"].astype("Int64").astype(str),
        format="%Y%m%d%H%M", errors="coerce",
    )
    values = numeric_clean(frame, columns, 0.0, 100.0, divisor=100.0)
    values["date"] = stamp.dt.normalize()
    values = values[values["date"].notna() & values["date"].le(END_DATE)]
    return values.groupby("date")[columns].median()


def find_time_column(path):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    for candidate in ("Time_EST", "datetime", "date", "TIMESTAMP", "DateTime"):
        if candidate in header:
            return candidate
    raise ValueError(f"No recognized time column in {path}")


def read_hf_daily(path, columns):
    time_column = find_time_column(path)
    frame = pd.read_csv(
        path, usecols=[time_column, *columns], na_values=HF_MISSING,
        low_memory=False, on_bad_lines="skip",
    )
    stamp = pd.to_datetime(frame[time_column], errors="coerce")
    values = numeric_clean(frame, columns, 0.0, 1.0)
    values["date"] = stamp.dt.normalize()
    values = values[values["date"].notna() & values["date"].le(END_DATE)]
    return values.groupby("date")[columns].median()


def merge_periods(preferred, fallback):
    """Use preferred values and fill only their missing cells from fallback."""
    index = preferred.index.union(fallback.index).sort_values()
    return preferred.reindex(index).combine_first(fallback.reindex(index))


def fuse_duplicate(primary, fallback, output_names, label):
    """Fuse duplicate sensor tables and retain value-level source provenance."""
    index = primary.index.union(fallback.index).sort_values()
    primary = primary.reindex(index)
    fallback = fallback.reindex(index)
    output = pd.DataFrame(index=index)
    qa = []
    for source_column, output_column in output_names.items():
        amf = primary[source_column]
        hf = fallback[source_column]
        overlap = pd.concat([amf.rename("amf"), hf.rename("hf")], axis=1).dropna()
        difference = (overlap["amf"] - overlap["hf"]).abs()
        qa.append({
            "sensor": output_column,
            "overlap_days": len(overlap),
            "median_absolute_difference": difference.median() if len(difference) else np.nan,
            "maximum_absolute_difference": difference.max() if len(difference) else np.nan,
            "days_difference_over_0.005": int(difference.gt(0.005).sum()),
        })
        output[output_column] = amf.combine_first(hf)
        output[f"{output_column}_source"] = np.select(
            [amf.notna(), hf.notna()],
            ["AmeriFlux", "HarvardForest_fallback"],
            default="",
        )
    print(f"\n{label} duplicate-source checks (all values in m3 m-3)")
    print(pd.DataFrame(qa).to_string(index=False))
    return output


def retain_observation_days(frame):
    # Retain only dates containing at least one real measurement. A row with no
    # measurement is not an observation and adds no information to the CSV.
    measurement_columns = [
        column for column in frame
        if column.endswith("_m3_m3") or column.endswith("_fraction")
    ]
    observed = frame[measurement_columns]
    valid_dates = observed.dropna(how="all").index
    if valid_dates.empty:
        raise ValueError("No valid soil-moisture observations")
    return frame.loc[valid_dates].sort_index()


def build_ems():
    amf_columns = [f"SWC_{horizontal}_{vertical}_1"
                   for horizontal in (7, 8) for vertical in range(1, 5)]
    amf = read_amf_daily(FLUX / "AMF_US-Ha1_BASE_HR_27-5.csv", amf_columns)

    profile_names = {
        "SWC_7_1_1": "profile_a_15cm_m3_m3",
        "SWC_7_2_1": "profile_a_depth2_35_40cm_m3_m3",
        "SWC_7_3_1": "profile_a_50cm_m3_m3",
        "SWC_7_4_1": "profile_a_90cm_m3_m3",
        "SWC_8_1_1": "profile_b_15cm_m3_m3",
        "SWC_8_2_1": "profile_b_depth2_35_40cm_m3_m3",
        "SWC_8_3_1": "profile_b_50cm_m3_m3",
        "SWC_8_4_1": "profile_b_90cm_m3_m3",
    }

    old_names = {
        "vwc_15_a": "SWC_7_1_1", "vwc_40_a": "SWC_7_2_1",
        "vwc_50_a": "SWC_7_3_1", "vwc_90_a": "SWC_7_4_1",
        "vwc_15_b": "SWC_8_1_1", "vwc_40_b": "SWC_8_2_1",
        "vwc_50_b": "SWC_8_3_1", "vwc_90_b": "SWC_8_4_1",
    }
    new_names = {
        "vwc_15cm_avg_m_3_m_3_a": "SWC_7_1_1",
        "vwc_35cm_avg_m_3_m_3_a": "SWC_7_2_1",
        "vwc_50cm_avg_m_3_m_3_a": "SWC_7_3_1",
        "vwc_90cm_avg_m_3_m_3_a": "SWC_7_4_1",
        "vwc_15cm_avg_m_3_m_3_b": "SWC_8_1_1",
        "vwc_35cm_avg_m_3_m_3_b": "SWC_8_2_1",
        "vwc_50cm_avg_m_3_m_3_b": "SWC_8_3_1",
        "vwc_90cm_avg_m_3_m_3_b": "SWC_8_4_1",
    }
    old = read_hf_daily(
        ARCHIVE / "HF069" / "hf069-18-swc-2007-2017.csv", list(old_names)
    ).rename(columns=old_names)
    new = read_hf_daily(
        ARCHIVE / "HF069" / "hf069-19-swc-since-2017.csv", list(new_names)
    ).rename(columns=new_names)
    # The newer HF069 system takes priority on its overlapping transition days.
    hf_profiles = merge_periods(new, old)
    profiles = fuse_duplicate(amf, hf_profiles, profile_names, "EMS profiles")

    hf206_columns = [f"swc{i}_ave" for i in range(1, 5)]
    hf206 = read_hf_daily(
        ARCHIVE / "HF206" / "hf206-05-EMS-understory.csv", hf206_columns
    ).rename(columns={
        f"swc{i}_ave": f"hf206_upper15_sensor{i}_m3_m3" for i in range(1, 5)
    })

    output = profiles.join(hf206, how="outer")
    profile_value_columns = list(profile_names.values())
    hf206_value_columns = list(hf206.columns)
    output["profile_valid_sensor_count"] = output[profile_value_columns].notna().sum(axis=1)
    output["hf206_upper15_valid_sensor_count"] = output[hf206_value_columns].notna().sum(axis=1)
    output["all_continuous_valid_sensor_count"] = output[
        profile_value_columns + hf206_value_columns
    ].notna().sum(axis=1)
    output = retain_observation_days(output)
    count_columns = [
        "profile_valid_sensor_count", "hf206_upper15_valid_sensor_count",
        "all_continuous_valid_sensor_count",
    ]
    output[count_columns] = output[count_columns].fillna(0).astype("Int64")
    output.insert(0, "site", "EMS")
    return output.reset_index()


def build_hem():
    amf_columns = [f"SWC_PI_{i}" for i in range(1, 5)]
    amf = read_amf_daily(FLUX / "AMF_US-Ha2_BASE_HH_16-5.csv", amf_columns)
    hf_columns = [f"swc{i}_ave" for i in range(1, 5)]
    hf = read_hf_daily(
        ARCHIVE / "HF206" / "hf206-03-HK-understory.csv", hf_columns
    ).rename(columns={f"swc{i}_ave": f"SWC_PI_{i}" for i in range(1, 5)})
    names = {f"SWC_PI_{i}": f"upper15_sensor{i}_m3_m3" for i in range(1, 5)}
    continuous = fuse_duplicate(amf, hf, names, "HEM upper-15-cm sensors")

    field_frames = []
    for filename in (
        "hf153-01-hem-swc-1998-2000.csv",
        "hf153-02-hem-swc-2001-2007.csv",
    ):
        field_frames.append(read_hf_daily(
            ARCHIVE / "HF153" / filename, ["vwc", "gwc"]
        ))
    field = pd.concat(field_frames).groupby(level=0).median().rename(columns={
        "vwc": "field_vwc_m3_m3",
        "gwc": "field_gwc_fraction",
    })

    output = continuous.join(field, how="outer")
    continuous_columns = list(names.values())
    output["continuous_valid_sensor_count"] = output[continuous_columns].notna().sum(axis=1)
    output["field_vwc_available"] = output["field_vwc_m3_m3"].notna().astype("Int64")
    output = retain_observation_days(output)
    output[["continuous_valid_sensor_count", "field_vwc_available"]] = output[
        ["continuous_valid_sensor_count", "field_vwc_available"]
    ].fillna(0).astype("Int64")
    output.insert(0, "site", "HEM")
    return output.reset_index()


def verify_and_write(frame, site, filename):
    if frame["date"].duplicated().any():
        raise AssertionError(f"Duplicate dates remain in {site}")
    if not frame["date"].is_monotonic_increasing:
        raise AssertionError(f"Dates are not ordered in {site}")
    vwc_columns = [column for column in frame if column.endswith("_m3_m3")]
    fraction_columns = vwc_columns + [
        column for column in frame if column.endswith("_fraction")
    ]
    bad = frame[fraction_columns].apply(
        lambda x: (~x.between(0.0, 1.0)) & x.notna()
    ).sum().sum()
    if bad:
        raise AssertionError(f"{site} contains {bad} out-of-range VWC values")

    # Avoid binary floating-point artifacts in the plain-text CSV while
    # retaining more precision than the source measurements provide.
    frame[fraction_columns] = frame[fraction_columns].round(6)

    path = OUT / filename
    frame.to_csv(path, index=False, date_format="%Y-%m-%d", na_rep="")
    print(f"\n{site} output: {path}")
    print(f"  rows: {len(frame):,}; dates: {frame['date'].min().date()} to {frame['date'].max().date()}")
    print(f"  duplicate dates: {frame['date'].duplicated().sum()}")
    print(f"  out-of-range VWC values: {bad}")
    for column in vwc_columns:
        nonmissing = int(frame[column].notna().sum())
        if nonmissing:
            first = frame.loc[frame[column].notna(), "date"].min().date()
            last = frame.loc[frame[column].notna(), "date"].max().date()
            print(f"  {column}: {nonmissing:,} days, {first} to {last}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ems = build_ems()
    hem = build_hem()
    verify_and_write(ems, "EMS", "EMS_daily_soil_moisture.csv")
    verify_and_write(hem, "HEM", "HEM_daily_soil_moisture.csv")
    print("\nNo interpolation or temporal gap filling was used.")
    print("AmeriFlux duplicate values were selected first; Harvard Forest supplied fallback values only.")
    print("Source files and earlier processed files were not modified.")


if __name__ == "__main__":
    main()
