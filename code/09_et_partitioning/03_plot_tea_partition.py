"""Create the concise main-text reviewer-response figure for ET partitioning.

Panels a-b show annual T/ET during the manuscript-defined growing and
non-growing seasons. Panels c-d compare the original total-ET conductance
(g_c), TEA-transpiration conductance (g_s), and g_s adjusted for upper-15-cm
soil moisture during the continuous HF206 overlap.

Color meaning is kept consistent with the manuscript: EMS is red and HEM is
teal. Line style and marker shape distinguish quantities, avoiding a large
set of unrelated colors.

Required upstream outputs:
  outputs/et_partitioning/TEA_annual_GS_NGS_manuscript_definition.csv
  outputs/et_partitioning/conductance_comparison/
      annual_median_conductance_comparison.csv
  outputs/soil_moisture/conductance_sensitivity/
      annual_conductance_soil_moisture.csv
      soil_moisture_sensitivity_statistics.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import norm, theilslopes


def locate_repository():
    """Locate the repository when run as a script, from Bash, or interactively."""
    candidates = []
    if "__file__" in globals():
        script = Path(__file__).resolve()
        candidates.extend([script.parent, *script.parents])
    current = Path.cwd().resolve()
    candidates.extend([current, *current.parents])
    checked = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.add(candidate)
        if (candidate / "outputs" / "et_partitioning").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate harvard-forest-et-dynamics")


REPO = locate_repository()
PARTITION = REPO / "outputs" / "et_partitioning"
CONDUCTANCE = PARTITION / "conductance_comparison"
SOIL = REPO / "outputs" / "soil_moisture" / "conductance_sensitivity"
FIGURES = PARTITION / "figures"

SITE_PALETTE = {
    # Shades follow the existing Figure 3 palette.
    "EMS": {"dark": "#ef1414", "mid": "#f17f78", "light": "#f69a73"},
    "HEM": {"dark": "#17878a", "mid": "#26d4df", "light": "#27aee8"},
}
PANEL_LABELS = ["a)", "b)", "c)", "d)"]
SWC_COLOR = "#202020"
PRIMARY_SWC_RECORD = {
    "EMS": "EMS_HF206_upper15cm",
    "HEM": "HEM_continuous_upper15cm",
}
ANALYSIS_WINDOW = {"EMS": (1993, 2023), "HEM": (2006, 2022)}
RATIO_Y_LIMITS = {"EMS": (0.00, 1.02), "HEM": (0.40, 0.90)}
CONDUCTANCE_Y_LIMITS = {"EMS": (0.30, 1.00), "HEM": (0.20, 0.72)}


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 20,
    "axes.titlesize": 25,
    "axes.labelsize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
    "axes.linewidth": 1.7,
    "xtick.major.width": 1.7,
    "ytick.major.width": 1.7,
    "xtick.major.size": 7,
    "ytick.major.size": 7,
})


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required output not found: {path}\n"
            "Run scripts 02, 04, and 07 before this plotting script."
        )
    return path


def original_mann_kendall(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return np.nan, np.nan
    s_value = sum(np.sign(values[j] - values[:j]).sum() for j in range(1, n))
    _, counts = np.unique(values, return_counts=True)
    ties = counts[counts > 1]
    variance = (
        n * (n - 1) * (2 * n + 5)
        - np.sum(ties * (ties - 1) * (2 * ties + 5))
    ) / 18.0
    if variance <= 0:
        return 0.0, 1.0
    z_value = (
        (s_value - 1) / np.sqrt(variance) if s_value > 0
        else (s_value + 1) / np.sqrt(variance) if s_value < 0
        else 0.0
    )
    tau = s_value / (0.5 * n * (n - 1))
    return float(tau), float(2.0 * norm.sf(abs(z_value)))


def trend(year, values):
    frame = pd.DataFrame({"year": year, "value": values}).dropna()
    if len(frame) < 3:
        return np.nan, np.nan, np.nan, np.nan
    tau, p_value = original_mann_kendall(frame["value"])
    slope, intercept, _, _ = theilslopes(frame["value"], frame["year"])
    return float(tau), float(p_value), float(slope), float(intercept)


def format_p(value):
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    if value < 0.01:
        return f"{value:.3f}"
    return f"{value:.2f}"


def p_text(value):
    formatted = format_p(value)
    return f"p{formatted}" if formatted.startswith("<") else f"p={formatted}"


def load_inputs():
    ratios = pd.read_csv(require_file(
        PARTITION / "TEA_annual_GS_NGS_manuscript_definition.csv"
    ))
    conductance = pd.read_csv(require_file(
        CONDUCTANCE / "annual_median_conductance_comparison.csv"
    ))
    soil_annual = pd.read_csv(require_file(
        SOIL / "annual_conductance_soil_moisture.csv"
    ))
    soil_stats = pd.read_csv(require_file(
        SOIL / "soil_moisture_sensitivity_statistics.csv"
    ))
    for frame in (ratios, conductance, soil_annual):
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    ratios = pd.concat([
        ratios[(ratios["site"] == site)
               & ratios["year"].between(first, last)]
        for site, (first, last) in ANALYSIS_WINDOW.items()
    ], ignore_index=True)
    conductance = pd.concat([
        conductance[(conductance["site"] == site)
                    & conductance["year"].between(first, last)]
        for site, (first, last) in ANALYSIS_WINDOW.items()
    ], ignore_index=True)
    return ratios, conductance, soil_annual, soil_stats


def add_panel_label(ax, label):
    ax.text(-0.105, 1.055, label, transform=ax.transAxes,
            fontsize=30, fontweight="bold", va="bottom", ha="left")


def style_axis(ax):
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))


def plot_ratio_panel(ax, data, site, statistics_rows):
    palette = SITE_PALETTE[site]
    styles = {
        "GS": dict(marker="o", linestyle="-", color=palette["mid"],
                   markerfacecolor=palette["mid"], markeredgecolor=palette["mid"],
                   label="Growing season"),
        "NGS": dict(marker="D", linestyle="--", color=palette["light"],
                    markerfacecolor="white", markeredgecolor=palette["light"],
                    label="Non-growing season"),
    }
    for season in ("GS", "NGS"):
        frame = data[(data["site"] == site) & (data["season"] == season)].copy()
        frame = frame.sort_values("year").dropna(subset=["T_over_ET"])
        tau, p_value, slope, intercept = trend(frame["year"], frame["T_over_ET"])
        style = styles[season]
        label = f"{style['label']}: {p_text(p_value)}"
        ax.plot(
            frame["year"], frame["T_over_ET"], color=style["color"],
            marker=style["marker"], linestyle=style["linestyle"],
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["markeredgecolor"], markeredgewidth=1.8,
            markersize=9.0, linewidth=2.7, label=label,
        )
        ax.plot(
            frame["year"], intercept + slope * frame["year"],
            color=style["color"], linestyle=style["linestyle"],
            linewidth=3.5, alpha=0.58,
        )
        statistics_rows.append({
            "site": site, "analysis": "T_over_ET", "series": season,
            "first_year": int(frame["year"].min()),
            "last_year": int(frame["year"].max()), "n_years": len(frame),
            "tau": tau, "p_value": p_value, "slope_per_year": slope,
            "swc_term_p": np.nan,
        })
    ax.axhline(1.0, color="#888888", linewidth=1.1, linestyle=":")
    ax.set_ylim(*RATIO_Y_LIMITS[site])
    ax.set_title(
        f"US-Ha{'1' if site == 'EMS' else '2'} ({site})",
        pad=14,
    )
    ax.set_xlabel("Year")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc",
              framealpha=0.92, loc="best")
    style_axis(ax)


def plot_conductance_panel(ax, conductance, soil_annual, soil_stats, site,
                           statistics_rows):
    palette = SITE_PALETTE[site]
    frame = conductance[conductance["site"] == site].sort_values("year").copy()
    tau_gc, p_gc, slope_gc, _ = trend(frame["year"], frame["gc_ET"])
    tau_gs, p_gs, slope_gs, _ = trend(frame["year"], frame["gc_T"])

    ax.plot(
        frame["year"], frame["gc_ET"], color=palette["dark"],
        marker="o", linestyle="-",
        linewidth=2.8, markersize=9.0,
        label=rf"ET-based $g_c$: {p_text(p_gc)}",
    )
    ax.plot(
        frame["year"], frame["gc_T"], color=palette["mid"],
        marker="s", linestyle="--",
        markerfacecolor="white", markeredgecolor=palette["mid"],
        markeredgewidth=1.8,
        linewidth=2.8, markersize=8.5,
        label=rf"T-based $g_s$: {p_text(p_gs)}",
    )

    record = PRIMARY_SWC_RECORD[site]
    adjusted = soil_annual[soil_annual["record"] == record].copy()
    adjusted = adjusted.dropna(subset=["gc_T", "gc_T_adjusted_for_swc"])
    adjusted = adjusted.sort_values("year")
    # Center the adjusted series on the observed g_s mean so the adjustment
    # changes interannual variation but not the visual baseline.
    adjusted["gs_adjusted_plot"] = (
        adjusted["gc_T_adjusted_for_swc"]
        - adjusted["gc_T_adjusted_for_swc"].mean()
        + adjusted["gc_T"].mean()
    )
    selected_stats = soil_stats[
        (soil_stats["site"] == site)
        & (soil_stats["record"] == record)
        & (soil_stats["outcome"] == "gc_T")
    ]
    if len(selected_stats) != 1:
        raise ValueError(f"Expected one soil-moisture result for {site} {record}")
    selected = selected_stats.iloc[0]
    p_adjusted = float(selected["adjusted_ols_year_hc3_p"])
    p_swc = float(selected["adjusted_ols_swc_hc3_p"])
    adjusted_slope = float(selected["adjusted_ols_year_slope"])
    ax.plot(
        adjusted["year"], adjusted["gs_adjusted_plot"], color=SWC_COLOR,
        marker="^", linestyle=":", markerfacecolor="white",
        markeredgecolor=SWC_COLOR, markeredgewidth=2.0,
        linewidth=2.9, markersize=9.0,
        label=rf"$g_s$ adjusted for SWC: {p_text(p_adjusted)}",
    )

    statistics_rows.extend([
        {
            "site": site, "analysis": "conductance", "series": "g_c_ET",
            "first_year": int(frame["year"].min()),
            "last_year": int(frame["year"].max()), "n_years": len(frame),
            "tau": tau_gc, "p_value": p_gc, "slope_per_year": slope_gc,
            "swc_term_p": np.nan,
        },
        {
            "site": site, "analysis": "conductance", "series": "g_s_TEA_T",
            "first_year": int(frame["year"].min()),
            "last_year": int(frame["year"].max()), "n_years": len(frame),
            "tau": tau_gs, "p_value": p_gs, "slope_per_year": slope_gs,
            "swc_term_p": np.nan,
        },
        {
            "site": site, "analysis": "conductance",
            "series": "g_s_TEA_T_adjusted_for_SWC",
            "first_year": int(adjusted["year"].min()),
            "last_year": int(adjusted["year"].max()),
            "n_years": len(adjusted), "tau": np.nan,
            "p_value": p_adjusted, "slope_per_year": adjusted_slope,
            "swc_term_p": p_swc,
        },
    ])
    ax.set_ylim(*CONDUCTANCE_Y_LIMITS[site])
    ax.set_xlabel("Year")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc",
              framealpha=0.92, loc="best")
    style_axis(ax)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    ratios, conductance, soil_annual, soil_stats = load_inputs()
    statistics_rows = []

    fig, axes = plt.subplots(2, 2, figsize=(22.0, 16.0), sharey=False)
    for column, site in enumerate(("EMS", "HEM")):
        plot_ratio_panel(axes[0, column], ratios, site, statistics_rows)
        plot_conductance_panel(
            axes[1, column], conductance, soil_annual, soil_stats,
            site, statistics_rows,
        )

    axes[0, 0].set_ylabel("T/ET")
    axes[1, 0].set_ylabel(r"Median conductance (mol m$^{-2}$ s$^{-1}$)")
    for ax, label in zip(axes.flat, PANEL_LABELS):
        add_panel_label(ax, label)

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.085, top=0.955,
                        wspace=0.16, hspace=0.30)
    png_path = FIGURES / "Figure4_ET_partitioning_conductance.png"
    pdf_path = FIGURES / "Figure4_ET_partitioning_conductance.pdf"
    csv_path = FIGURES / "Figure4_ET_partitioning_conductance_statistics.csv"
    fig.savefig(png_path, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    pd.DataFrame(statistics_rows).to_csv(csv_path, index=False)

    print("Created the reviewer-focused main-text figure.")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print(f"Statistics: {csv_path}")
    print("\nKey soil-moisture sensitivity results used in panels c-d:")
    for site in ("EMS", "HEM"):
        record = PRIMARY_SWC_RECORD[site]
        row = soil_stats[
            (soil_stats["site"] == site)
            & (soil_stats["record"] == record)
            & (soil_stats["outcome"] == "gc_T")
        ].iloc[0]
        print(
            f"  {site}: adjusted g_s year slope="
            f"{row['adjusted_ols_year_slope']:+.5f} mol m-2 s-1 yr-1, "
            f"year {p_text(row['adjusted_ols_year_hc3_p'])}, "
            f"SWC {p_text(row['adjusted_ols_swc_hc3_p'])}"
        )


if __name__ == "__main__":
    main()
