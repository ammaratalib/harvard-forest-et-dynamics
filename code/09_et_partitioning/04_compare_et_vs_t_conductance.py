"""Compare flux-gradient conductance calculated from total ET versus TEA T.

This reviewer-response diagnostic reproduces the flux-gradient equations used
in the manuscript workflow (after Wehr et al.) on one common set of subdaily
records. The only difference between the paired estimates is the water flux:
total ET or Nelson-TEA transpiration. Original manuscript products are never
overwritten.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, theilslopes

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
PART = REPO / "outputs" / "et_partitioning"
OUT = PART / "conductance_comparison"

R_GAS = 8.314472
MOLAR_MASS_DRY_AIR = 0.02897
RBH_S_M = 10.0
SC_CO2 = 1.05
SC_H2O = SC_CO2 / 1.57278
PR = 0.71
RBV_S_M = 2.0 * RBH_S_M * (SC_H2O / PR) ** (2.0 / 3.0)


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
    z = (s - 1) / np.sqrt(var_s) if s > 0 else ((s + 1) / np.sqrt(var_s) if s < 0 else 0.0)
    return float(s / (0.5 * n * (n - 1))), float(2.0 * norm.sf(abs(z)))


def latent_heat(tair_c):
    """Latent heat of vaporization, J kg-1."""
    return 3_149_000.0 - 2_370.0 * (tair_c + 273.15)


def numeric(frame, name):
    return pd.to_numeric(frame[name], errors="coerce")


def source_data(site):
    if site == "EMS":
        d = pd.read_csv(
            REPO / "data" / "processed" / "flux" / "gaps_ML_EMS.csv",
            low_memory=False,
        )
        return pd.DataFrame({
            "timestamp": pd.to_datetime(d["TIMESTAMP"], errors="coerce"),
            "H": numeric(d, "H"), "PA": numeric(d, "PA"),
            "RH": numeric(d, "RH"), "PAR": numeric(d, "PAR"),
            "NETRAD": numeric(d, "NETRAD"),
            "sos": numeric(d, "sos"), "eos": numeric(d, "eos"),
        })
    d = pd.read_csv(REPO / "data" / "processed" / "flux" / "gaps_ML_HEM_2023.csv",
                    low_memory=False)
    return pd.DataFrame({
        "timestamp": pd.to_datetime(d["TIMESTAMP"], errors="coerce"),
        "H": numeric(d, "H"), "PA": numeric(d, "PA"),
        "RH": numeric(d, "RH"),
        "PAR": numeric(d, "PPFD"), "NETRAD": numeric(d, "NETRAD"),
        "sos": numeric(d, "sos"), "eos": numeric(d, "eos"),
    })


def conductance(frame, water_mm, seconds, closure):
    """Flux-gradient conductance to water vapor, mol m-2 s-1."""
    tair = frame["Tair_C"].to_numpy(float)
    pa = frame["PA"].to_numpy(float)
    rh = np.clip(frame["RH"].to_numpy(float), 0, 100)
    h_corrected = frame["H"].to_numpy(float) / closure
    lam = latent_heat(tair)

    sat_vp = 100.0 * 6.112 * np.exp(17.62 * tair / (243.12 + tair))
    vp = sat_vp - frame["VPD_kPa"].to_numpy(float) * 1000.0
    air_conc_dry = (pa * 1000.0 - vp) / (R_GAS * (tair + 273.15))
    air_conc_wet = pa * 1000.0 / (R_GAS * (tair + 273.15))
    cp_dry = 1003.0 + 5.0 * ((tair + 23.16) / 100.0)
    cp_vapor = cp_dry * (1.0 + 0.84 * rh / 100.0)
    rho_dry = air_conc_dry * MOLAR_MASS_DRY_AIR
    rho_vapor = (air_conc_wet - air_conc_dry) * 0.018
    rho_wet = rho_dry + rho_vapor
    cp_wet = (rho_dry * cp_dry + rho_vapor * cp_vapor) / rho_wet

    leaf_temp = tair + h_corrected * RBH_S_M / (rho_wet * cp_wet)
    sat_leaf = 100.0 * 6.112 * np.exp(17.62 * leaf_temp / (243.12 + leaf_temp))
    concentration_difference = (sat_leaf - vp) / (R_GAS * (tair + 273.15))
    water_mol = np.asarray(water_mm, float) / seconds / 0.018 / closure
    with np.errstate(divide="ignore", invalid="ignore"):
        surface_resistance = concentration_difference / water_mol - RBV_S_M
        gc = pa * 1000.0 / (R_GAS * (leaf_temp + 273.15) * surface_resistance)
    return gc


def trend(yearly, column):
    z = yearly[["year", column]].dropna()
    tau, p = original_mann_kendall(z[column])
    slope = theilslopes(z[column], z["year"])[0]
    return tau, p, slope


def run_site(site, seconds, first_year, last_year):
    p = pd.read_csv(PART / f"{site}_TEA_partition_subdaily.csv", low_memory=False)
    p["timestamp"] = pd.to_datetime(p["timestamp"], errors="coerce")
    p = p.merge(source_data(site), on="timestamp", how="left", validate="one_to_one")
    p["date"] = p["timestamp"].dt.normalize()
    p["year"] = p["timestamp"].dt.year
    p["doy"] = p["timestamp"].dt.dayofyear
    p["rain_day"] = p.groupby("date")["precip_mm_timestep"].transform("sum").gt(0)
    # Recover the gap-filled total latent heat flux used to create ET.
    p["LE_W_m2"] = p["ET_mm"] * latent_heat(p["Tair_C"]) / seconds

    # The manuscript energy-closure correction is estimated from total turbulent
    # heat flux, then held identical for the paired ET- and T-based calculations.
    base = p["LE_W_m2"].ge(50) & p["PAR"].ge(500)
    finite_energy = base & p[["H", "LE_W_m2", "NETRAD"]].notna().all(axis=1)
    available = p.loc[finite_energy, "NETRAD"] * (1.0 - 15.0 / 700.0)
    turbulent = p.loc[finite_energy, "H"] + p.loc[finite_energy, "LE_W_m2"]
    closure = np.polyfit(available, turbulent, 1)[0]

    p["gc_ET"] = conductance(p, p["ET_mm"], seconds, closure)
    p["gc_T"] = conductance(p, p["T_mm"], seconds, closure)
    p["growing_season"] = p["doy"].between(p["sos"], p["eos"])
    # Match manuscript rain-day, radiation, LE and growing-season screening.
    use = (base & p["growing_season"] & ~p["rain_day"]
           & p["year"].between(first_year, last_year)
           & p[["gc_ET", "gc_T"]].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
           & p["gc_ET"].gt(0) & p["gc_T"].gt(0))
    sample = p.loc[use].copy()

    yearly = sample.groupby("year", as_index=False).agg(
        gc_ET=("gc_ET", "median"), gc_T=("gc_T", "median"),
        n=("timestamp", "size"), T_over_ET=("T_over_ET", "median"),
    )
    yearly["T_change_percent"] = 100.0 * (yearly["gc_T"] / yearly["gc_ET"] - 1.0)
    yearly.insert(0, "site", site)

    print(f"\n{site}: closure slope={closure:.3f}; paired GS records={len(sample):,}")
    for col in ("gc_ET", "gc_T"):
        tau, pval, slope = trend(yearly, col)
        print(f"  {col}: median={yearly[col].median():.3f}, tau={tau:+.3f}, "
              f"p={pval:.4g}, slope={slope:+.5f} mol m-2 s-1 yr-1")
    print(f"  median change from ET-based to T-based gc: {yearly['T_change_percent'].median():+.1f}%")

    sample[["timestamp", "site", "ET_mm", "T_mm", "gc_ET", "gc_T",
            "T_over_ET"]].to_csv(OUT / f"{site}_paired_subdaily_conductance.csv", index=False)
    yearly.to_csv(OUT / f"{site}_annual_median_conductance.csv", index=False)
    return yearly


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_yearly = pd.concat([
        run_site("EMS", 3600, 1993, 2023),
        run_site("HEM", 1800, 2006, 2022),
    ], ignore_index=True)
    all_yearly.to_csv(OUT / "annual_median_conductance_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (site, d) in zip(axes, all_yearly.groupby("site", sort=False)):
        ax.plot(d["year"], d["gc_ET"], "o-", label="Total ET-derived")
        ax.plot(d["year"], d["gc_T"], "o-", label="TEA T-derived")
        ax.set(title=site, xlabel="Year")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Annual median conductance (mol m$^{-2}$ s$^{-1}$)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "ET_vs_T_annual_median_conductance.png", dpi=300)
    plt.close(fig)
    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
