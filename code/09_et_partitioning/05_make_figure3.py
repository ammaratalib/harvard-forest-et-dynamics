"""Regenerate manuscript Figure 3 with the corrected HEM conductance series.

EMS is read unchanged from the manuscript's final processed table. HEM is
recomputed from the repository's final half-hourly flux table using the
original flux-gradient equations, eligibility criteria, rain-day filter,
season definitions, aggregation, and plotting style. Corrections are limited
to the latent heat of vaporization and two documented 2023 scale changes:
VPD is converted from hPa to kPa, and the PPFD screening variable is rebuilt
from shortwave radiation using the robust 2005--2022 PPFD/Rg relationship.
The latter affects only the original PAR >= 500 eligibility screen; PPFD does
not otherwise enter the conductance calculation.
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import norm, theilslopes

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
OUT = REPO / "outputs" / "figures"
R = 8.314472
MM_DRY = 0.02897
RBH = 10.0
RBV = 2.0 * RBH * ((1.05 / 1.57278) / 0.71) ** (2.0 / 3.0)


def numeric(d, name):
    return pd.to_numeric(d[name], errors="coerce")


def latent_heat_j_kg(tair_c):
    """Original intended formula, returned in J kg-1."""
    return 3_149_000.0 - 2_370.0 * (tair_c + 273.16)


def original_mann_kendall(values):
    """Original Mann-Kendall test with tie correction and continuity correction."""
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
    p = 2.0 * norm.sf(abs(z))
    tau = s / (0.5 * n * (n - 1))
    return float(tau), float(p)


def corrected_hem_rows():
    source = REPO / "data/processed/flux/gaps_ML_HEM_2023.csv"
    d = pd.read_csv(source, low_memory=False)
    required = {
        "TIMESTAMP", "DoY", "sos", "eos", "Tair", "PA", "RH", "H",
        "LE", "VPD", "PPFD", "Rg", "NETRAD",
    }
    missing = required.difference(d.columns)
    if missing:
        raise ValueError(f"HEM flux input is missing columns: {sorted(missing)}")
    d["TIMESTAMP"] = pd.to_datetime(d["TIMESTAMP"], errors="coerce")

    # The 2023 VPD values are stored in hPa, while 2004--2022 are in kPa.
    # Verify the scale change against VPD independently reconstructed from
    # measured temperature and relative humidity before applying /10.
    calendar_year = d["TIMESTAMP"].dt.year
    tair = numeric(d, "Tair")
    rh = numeric(d, "RH")
    sat_air_pa = 100.0 * 6.112 * np.exp(17.62 * tair / (243.12 + tair))
    vpd_from_rh_kpa = sat_air_pa * (1.0 - rh / 100.0) / 1000.0
    ratio_2023 = (
        numeric(d.loc[calendar_year.eq(2023)], "VPD")
        / vpd_from_rh_kpa.loc[calendar_year.eq(2023)]
    ).replace([np.inf, -np.inf], np.nan).median()
    if not 9.5 < ratio_2023 < 10.5:
        raise ValueError(f"Unexpected 2023 HEM VPD scale ratio: {ratio_2023:.3f}")
    d["VPD_original"] = numeric(d, "VPD")
    d["VPD_use"] = d["VPD_original"]
    d.loc[calendar_year.eq(2023), "VPD_use"] /= 10.0

    # PPFD changes scale in 2023. Rebuild only the radiation-screening value
    # from Rg using the robust historical daytime relationship. This avoids
    # treating the unit change as missing daylight data and leaves all earlier
    # years exactly on their observed PPFD screen.
    historical = calendar_year.between(2005, 2022) & numeric(d, "Rg").gt(50)
    ppfd_rg = (numeric(d.loc[historical], "PPFD") / numeric(d.loc[historical], "Rg"))
    ppfd_rg = ppfd_rg.replace([np.inf, -np.inf], np.nan)
    ppfd_rg = ppfd_rg[ppfd_rg.between(0.5, 5.0)].dropna()
    ppfd_rg_ratio = ppfd_rg.median()
    if not 1.5 < ppfd_rg_ratio < 2.5:
        raise ValueError(f"Implausible historical HEM PPFD/Rg ratio: {ppfd_rg_ratio:.3f}")
    d["PPFD_original"] = numeric(d, "PPFD")
    d["PPFD_screen"] = d["PPFD_original"]
    d.loc[calendar_year.eq(2023), "PPFD_screen"] = (
        numeric(d.loc[calendar_year.eq(2023)], "Rg") * ppfd_rg_ratio
    )

    original_base = numeric(d, "LE").ge(50) & d["PPFD_original"].ge(500)
    corrected_base = numeric(d, "LE").ge(50) & d["PPFD_screen"].ge(500)
    original_2023 = original_base & calendar_year.eq(2023)
    corrected_2023 = corrected_base & calendar_year.eq(2023)
    print(
        "HEM calendar-2023 radiation/LE eligibility: "
        f"{original_2023.sum():,} records across "
        f"{d.loc[original_2023, 'TIMESTAMP'].dt.date.nunique()} days before correction; "
        f"{corrected_2023.sum():,} records across "
        f"{d.loc[corrected_2023, 'TIMESTAMP'].dt.date.nunique()} days after correction"
    )
    d = d.loc[corrected_base].copy()

    lam = latent_heat_j_kg(numeric(d, "Tair"))
    if not lam.dropna().between(2.3e6, 2.7e6).all():
        raise ValueError("Corrected HEM latent heat is outside the physical 2.3--2.7 MJ kg-1 range")

    # Original annual-average energy-budget closure calculation.
    available = numeric(d, "NETRAD") * (1.0 - 15.0 / 700.0)
    turbulent = numeric(d, "H") + numeric(d, "LE")
    ok = np.isfinite(available) & np.isfinite(turbulent)
    closure = np.polyfit(available[ok], turbulent[ok], 1)[0]
    if not 0.3 < closure < 1.5:
        raise ValueError(f"Implausible HEM energy-closure slope: {closure}")

    tair, pa, rh = numeric(d, "Tair"), numeric(d, "PA"), numeric(d, "RH")
    h_corrected = numeric(d, "H") / closure
    sat = 100.0 * 6.112 * np.exp(17.62 * tair / (243.12 + tair))
    vp = sat - numeric(d, "VPD_use") * 1000.0
    dry_conc = (pa * 1000.0 - vp) / (R * (tair + 273.15))
    wet_conc = pa * 1000.0 / (R * (tair + 273.15))
    cp_dry = 1003.0 + 5.0 * ((tair + 23.16) / 100.0)
    cp_vapor = cp_dry * (1.0 + 0.84 * rh / 100.0)
    rho_dry = dry_conc * MM_DRY
    rho_vapor = (wet_conc - dry_conc) * 0.018
    rho_wet = rho_dry + rho_vapor
    cp_wet = (rho_dry * cp_dry + rho_vapor * cp_vapor) / rho_wet
    leaf_t = tair + h_corrected * RBH / (rho_wet * cp_wet)
    sat_leaf = 100.0 * 6.112 * np.exp(17.62 * leaf_t / (243.12 + leaf_t))
    concentration_difference = (sat_leaf - vp) / (R * (tair + 273.15))
    def conductance(lambda_j_kg):
        water_mol = (numeric(d, "LE") / closure) / (lambda_j_kg * 0.018)
        surface_resistance = concentration_difference / water_mol - RBV
        return pa * 1000.0 / (R * (leaf_t + 273.15) * surface_resistance)

    d["gs_FG"] = conductance(lam)
    d["lambda_corrected_J_kg"] = lam

    # Original processing removes all rain days using the Harvard Forest daily
    # precipitation record, then retains the 5th--95th conductance percentiles.
    rain = pd.read_csv(REPO / "data/source/climate/hf300-05-daily-m.csv", low_memory=False)
    rain["date"] = pd.to_datetime(rain["date"], errors="coerce").dt.normalize()
    rain = rain[["date", "prec"]].drop_duplicates("date")
    d["date"] = d["TIMESTAMP"].dt.normalize()
    d = d.merge(rain, on="date", how="left")
    d = d[d["prec"].eq(0)].copy()
    finite_gc = d.loc[np.isfinite(d["gs_FG"]), "gs_FG"]
    lower, upper = np.percentile(finite_gc, [5, 95])
    d = d[d["gs_FG"].between(lower, upper)].copy()
    if not d["prec"].eq(0).all():
        raise ValueError("Rain-day filter failed")
    d["water_year"] = np.where(d["TIMESTAMP"].dt.month < 10,
                                d["TIMESTAMP"].dt.year,
                                d["TIMESTAMP"].dt.year + 1)
    wy2023 = d[d["water_year"].eq(2023)]
    wy2023_days = wy2023["TIMESTAMP"].dt.date.nunique()
    if len(wy2023) < 500 or wy2023_days < 75:
        raise ValueError(
            "Corrected HEM water-year 2023 remains too sparse: "
            f"{len(wy2023)} records across {wy2023_days} days"
        )
    print(f"HEM energy-closure slope: {closure:.6f}")
    print(f"HEM historical PPFD/Rg screening ratio: {ppfd_rg_ratio:.6f}")
    print(f"HEM calendar-2023 VPD stored/reconstructed ratio: {ratio_2023:.6f}")
    print(f"Corrected lambda median: {d['lambda_corrected_J_kg'].median():.0f} J kg-1")
    print(f"Corrected HEM gs_FG 5th--95th percentile limits: {lower:.6f}, {upper:.6f}")
    print(
        f"Corrected HEM water-year 2023 retained: {len(wy2023):,} records "
        f"across {wy2023_days} days"
    )
    return d


def classify_and_summarize(d, site, first_year, last_year):
    d = d[d["water_year"].between(first_year, last_year)].copy()
    d["Grow_sea"] = np.where(
        numeric(d, "DoY").between(numeric(d, "sos"), numeric(d, "eos")), "GS", "NGS"
    )
    products = {}
    for name in ("Annual", "GS", "NGS"):
        z = d if name == "Annual" else d[d["Grow_sea"].eq(name)]
        z = z.groupby("water_year", as_index=False)["gs_FG"].median().dropna()
        tau, p = original_mann_kendall(z["gs_FG"])
        slope = theilslopes(z["gs_FG"], z["water_year"])[0]
        ols = np.polyfit(z["water_year"], z["gs_FG"], 1)[0]
        print(f"{site} {name}: years={len(z)}, median={z.gs_FG.median():.4f}, "
              f"range={z.gs_FG.min():.4f}--{z.gs_FG.max():.4f}, "
              f"tau={tau:+.3f}, p={p:.4g}, Theil-Sen={slope:+.5f}, OLS={ols:+.5f}")
        z.insert(0, "site", site)
        z["season"] = name
        z["kendall_tau"] = tau
        z["kendall_p"] = p
        z["theil_sen_slope"] = slope
        z["ols_slope"] = ols
        products[name] = z
    return products


def add_panel(ax, series, site, title, colors):
    styles = {
        "Annual": dict(linestyle="--", marker="o", markersize=13, linewidth=3.0),
        "GS": dict(linestyle="None", marker="2", markersize=19, markeredgewidth=2.0),
        "NGS": dict(linestyle="None", marker="*", markersize=19),
    }
    names = {"Annual": "Annual", "GS": "Growing", "NGS": "Non Growing"}

    def format_p(value):
        # Two decimal places normally; three when leading zeros would hide a
        # small p-value.
        return f"{value:.3f}" if value < 0.01 else f"{value:.2f}"

    for key in ("Annual", "GS", "NGS"):
        z = series[key]
        p = z["kendall_p"].iloc[0]
        slope = z["theil_sen_slope"].iloc[0]
        ax.plot(z["water_year"], z["gs_FG"], color=colors[key],
                label=f"{names[key]}, p={format_p(p)}, slope={slope:+.3f}", **styles[key])
        fit = np.polyfit(z["water_year"], z["gs_FG"], 1)
        ax.plot(z["water_year"], np.polyval(fit, z["water_year"]),
                color=colors[key], linestyle="--" if key != "Annual" else "-",
                linewidth=3.0)
    ax.set_title(title, fontsize=22, pad=14)
    ax.set_xlabel("Water Year", fontsize=20, labelpad=10)
    ax.set_ylabel(r"$g_c$ ($mol$ $m^{-2}$ $s^{-1}$)", fontsize=20, labelpad=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="both", which="major", labelsize=18, width=1.6, length=7)
    ax.legend(loc="upper left", fontsize=16, labelspacing=0.30,
              borderpad=0.55, handlelength=2.0, frameon=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # EMS final Figure 3 table is used unchanged.
    ems = pd.read_csv(REPO / "data/processed/conductance/gs_FG_all.csv", low_memory=False)
    hem = corrected_hem_rows()
    ems_series = classify_and_summarize(ems, "EMS", 1993, 2023)
    hem_series = classify_and_summarize(hem, "HEM", 2005, 2023)

    hem.to_csv(OUT / "Figure3_HEM_conductance.csv", index=False)
    summary = pd.concat([*ems_series.values(), *hem_series.values()], ignore_index=True)
    summary.to_csv(OUT / "Figure3_statistics.csv", index=False)

    # Side-by-side candidate supplied for convenient manuscript replacement.
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    add_panel(axes[0], ems_series, "EMS", "Harvard Forest, US-Ha1\nMedian Canopy Conductance ($g_c$)",
              {"Annual": "red", "GS": "lightcoral", "NGS": "lightsalmon"})
    add_panel(axes[1], hem_series, "HEM", "Harvard Forest, US-Ha2\nMedian Canopy Conductance ($g_c$)",
              {"Annual": "teal", "GS": "aqua", "NGS": "deepskyblue"})
    axes[1].set_xticks([2005, 2008, 2011, 2014, 2017, 2020, 2023])
    axes[1].set_xlim(2004.5, 2023.5)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.80, wspace=0.22)
    fig.text(0.012, 0.975, "a)", fontsize=26, fontweight="bold", va="top", ha="left")
    fig.text(0.505, 0.975, "b)", fontsize=26, fontweight="bold", va="top", ha="left")
    fig.savefig(OUT / "Figure3_canopy_conductance.png", dpi=600,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Outputs written to: {OUT}")


if __name__ == "__main__":
    main()
