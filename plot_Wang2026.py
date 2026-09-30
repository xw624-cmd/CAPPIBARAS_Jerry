#!/usr/bin/env python3
"""Make combined plots from all six independent CAPPIBARAS chains.

The six GetDist corner plots are combined into one 2x3 figure (major on top,
minor on bottom). Diagonal titles report the weighted posterior median with
the 16th/84th percentiles (+/- 1 sigma).
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml
import cobaya
import getdist.plots as gdplt
from scipy.stats import chi2 as chi2_dist
from matplotlib.lines import Line2D

from config import OUTPUT_PATH, set_plot_style
from plot_chains import get_param_results


PARAMS = ["P0", "beta_P", "A_2h_P"]
PARAM_LABELS = [r"$P_0$", r"$\beta_P$", r"$A_{2h,P}$"]
SECTORS = ["major", "minor"]

CAP_MAJOR_COLOR = "#771704"
CAP_MINOR_COLOR = "#5b7fa6"
SECTOR_COLORS = {
    "major": CAP_MAJOR_COLOR,
    "minor": CAP_MINOR_COLOR,
}

MASS_BIN_LABELS = [
    r"$11.0 < \log_{10}\!\left(\frac{M_\ast}{M_\odot}\right) \leq 11.4$",
    r"$11.4 < \log_{10}\!\left(\frac{M_\ast}{M_\odot}\right) \leq 11.7$",
    r"$11.7 < \log_{10}\!\left(\frac{M_\ast}{M_\odot}\right) \leq 12.0$",
]

POINT_X_OFFSETS = {
    "major": -0.04,
    "minor": +0.04,
}


# =============================================================================
# Plot layout and font settings -- edit these.
# Font sizes set to None use the default from set_plot_style().
# Spacing: wspace/hspace are fractions of the average panel width/height;
# w_pad/h_pad are in units of the font size.
# =============================================================================

# ---- Saving ----------------------------------------------------------------
# All figures are saved with bbox_inches="tight", which crops to the drawn
# content. SAVE_PAD_INCHES is the white border left around that crop
# (matplotlib's default is 0.1 in). Set to 0 for no border at all.
SAVE_PAD_INCHES = 0.02

# ---- Posterior summaries -----------------------------------------------------
# A parameter is reported as an upper limit when its posterior is still high at
# the prior edge (GetDist's rule, same as the original title_limit=1 plots):
# normalized density at the edge > exp(-1/2) ~ 0.61. Otherwise median +/- 1 sigma.
# 0 -> quote 68% upper limits (what the original plots showed)
# 1 -> quote 95% upper limits
UPPER_LIMIT_LEVEL = 0

# ---- Corner plots (2x3 figure) ----------------------------------------------
CORNER_FIGSIZE = (18.0, 12.0)
CORNER_BLOCK_WSPACE = 0.0     # gap between the three mass-bin columns
CORNER_BLOCK_HSPACE = 0.0     # gap between the major row and the minor row
CORNER_PANEL_WSPACE = 0.0    # gap between panels inside one corner plot
CORNER_PANEL_HSPACE = 0.0
# White space inside each corner block (fractions of the block), which holds the
# tick labels, axis labels and titles. This is what sets the visible gap between
# blocks. The top row needs more room on top for its two-line title.
CORNER_MARGINS_TOP_ROW = dict(left=0.085, right=0.995, bottom=0.07, top=0.90)
CORNER_MARGINS_BOTTOM_ROW = dict(left=0.085, right=0.995, bottom=0.07, top=0.925)
CORNER_GETDIST_SIZE = 1.7     # ~width of one panel (inches); scales GetDist line widths
CORNER_BLOCK_TITLE_FONTSIZE = 13   # mass-bin / "Major axis" title above each corner
CORNER_VALUE_FONTSIZE = 11          # median +/- 1 sigma titles on the diagonal
CORNER_LABEL_FONTSIZE = 11         # axis labels (P0, beta_P, A2h)
CORNER_TICK_FONTSIZE = 9
CORNER_BLOCK_TITLE_Y = 1.0

# ---- Posterior plot (3x3 figure) --------------------------------------------
POSTERIOR_FIGSIZE = (12, 9)
POSTERIOR_PAD = 0.2           # padding between figure edge and subplots
POSTERIOR_W_PAD = 0.4         # horizontal gap between columns
POSTERIOR_H_PAD = 0.4         # vertical gap between rows
POSTERIOR_TITLE_FONTSIZE = None
POSTERIOR_LABEL_FONTSIZE = None
POSTERIOR_TICK_FONTSIZE = None
POSTERIOR_LEGEND_FONTSIZE = None

# ---- Profile plot (1x3 figure) ----------------------------------------------
PROFILE_FIGSIZE = (16.5, 5.2)
PROFILE_PAD = 0.2             # padding between figure edge and subplots
PROFILE_W_PAD = 0          # horizontal gap between the three panels
PROFILE_LEGEND_SPACE = 0.08   # fraction of figure height reserved for the legend
PROFILE_LEGEND_Y = -0.015     # vertical position of the legend (figure coords)
PROFILE_LEGEND_FONTSIZE = 10
PROFILE_STATS_FONTSIZE = 10  # chi^2 box
PROFILE_TITLE_FONTSIZE = 11
PROFILE_LABEL_FONTSIZE = 10
PROFILE_TICK_FONTSIZE = None
PROFILE_MARKER_SIZE = 4.5
PROFILE_CAPSIZE = 2.5


def fontsize(size):
    """Pass fontsize only when it is set, so None keeps the style default."""
    return {} if size is None else {"fontsize": size}


def ticksize(ax, size):
    if size is not None:
        ax.tick_params(labelsize=size)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job-prefix", default="Wang2026_gnfw")
    p.add_argument("--output-dir", default=None)
    p.add_argument(
        "--ignore-rows",
        type=float,
        default=0.3,
        help="Burn-in fraction removed from each GetDist chain before posterior summaries.",
    )
    return p.parse_args()


def chain_root(job_prefix, mass_bin, sector):
    job = f"{job_prefix}_b{mass_bin}_{sector}"
    return Path(OUTPUT_PATH) / job / job


def load_getdist(root, ignore_rows):
    gd = cobaya.load_samples(str(root), to_getdist=True)
    if ignore_rows > 0:
        gd.removeBurnFraction(ignore_rows)
    return gd


def best_chain_params(root, gd):
    """Return the sampled parameter vector at the highest-likelihood chain point.

    GetDist stores the per-sample negative log-likelihood in ``gd.loglikes``.
    The fitted priors here are uniform inside hard bounds, so minimizing this
    quantity also selects the maximum-posterior point within the allowed region.
    """
    input_yaml = Path(str(root) + ".input.yaml")
    if not input_yaml.exists():
        raise FileNotFoundError(input_yaml)

    with open(input_yaml) as f:
        info = yaml.safe_load(f)

    if gd.loglikes is None or len(gd.loglikes) != len(gd.samples):
        raise RuntimeError(
            f"GetDist sample {root} does not contain a usable log-likelihood column"
        )

    finite = np.isfinite(gd.loglikes)
    if not np.any(finite):
        raise RuntimeError(f"No finite log-likelihood samples found in {root}")

    candidate_rows = np.flatnonzero(finite)
    best_row = int(candidate_rows[np.argmin(gd.loglikes[finite])])

    names = gd.paramNames.list()
    col = {name: i for i, name in enumerate(names)}

    params = {}
    for name, spec in info["params"].items():
        if isinstance(spec, dict) and "prior" in spec:
            if name not in col:
                raise KeyError(f"Sampled parameter {name!r} not found in chain")
            params[name] = float(gd.samples[best_row, col[name]])
        elif isinstance(spec, dict) and "value" in spec:
            params[name] = spec["value"]
        else:
            params[name] = spec

    return info, params, best_row, float(gd.loglikes[best_row])


def load_model_at_best(root, gd):
    """Return likelihood, 1h/2h/total pieces, and GOF at the best chain sample."""
    info, params, best_row, chain_negloglike = best_chain_params(root, gd)

    # cobaya.get_model does not need run-control blocks such as sampler/output.
    # Keeping only model-defining blocks avoids the repeated "Ignored blocks"
    # warnings when reconstructing the six models for plotting.
    model_info = {
        key: info[key]
        for key in ("likelihood", "params", "theory")
        if key in info
    }
    model = cobaya.get_model(model_info)
    likelihoods = model.likelihood

    if len(likelihoods) != 1:
        raise ValueError(f"Expected one likelihood in {root}; got {list(likelihoods)}")

    lname = next(iter(likelihoods))
    likeli = likelihoods[lname]
    result = get_param_results(params, likelihoods)[lname]

    theory_1h = np.asarray(result["1h"], dtype=float)
    theory_2h = np.asarray(result["2h"], dtype=float)
    theory_tot = np.asarray(result["tot"], dtype=float)

    data = np.asarray(likeli.data.y, dtype=float)
    inv_cov = np.asarray(likeli.data.inv_cov, dtype=float)

    # Keep the same defensive offset convention used in Chad's plotting code.
    offset = len(theory_tot) - len(data)
    theory_fit = theory_tot[offset:]
    delta = data - theory_fit
    chi2 = float(delta @ inv_cov @ delta)

    # Match Chad's plot_chains.compute_confidence_corner_multi convention:
    # dof = number of fitted data points, rather than N - n_parameters.
    N = int(len(likeli.data.x))
    chi2_over_N = chi2 / N
    pte = float(chi2_dist.sf(chi2, N))

    stats = {
        "chi2": chi2,
        "N": N,
        "chi2_over_N": chi2_over_N,
        "pte": pte,
        "best_row": best_row,
        "chain_negloglike": chain_negloglike,
        "params": {name: params[name] for name in PARAMS},
    }

    return likeli, theory_1h, theory_2h, theory_tot, stats


def posterior_limits(roots, params=PARAMS, nsigma=3.0):
    """Union of mean +/- nsigma for all roots, clipped to hard prior bounds."""
    limits = {}
    for param in params:
        lows = []
        highs = []
        for gd in roots:
            names = gd.paramNames.list()
            if param not in names:
                raise KeyError(f"Parameter {param!r} not found in chain")
            i = names.index(param)
            mean = float(gd.getMeans()[i])
            std = float(np.sqrt(gd.getVars()[i]))
            lo = mean - nsigma * std
            hi = mean + nsigma * std

            prior_lo = gd.ranges.getLower(param)
            prior_hi = gd.ranges.getUpper(param)
            if prior_lo is not None:
                lo = max(lo, prior_lo)
            if prior_hi is not None:
                hi = min(hi, prior_hi)

            lows.append(lo)
            highs.append(hi)
        limits[param] = (min(lows), max(highs))
    return limits


def weighted_quantile(values, quantiles, weights):
    """Weighted quantiles of MCMC samples."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[good], weights[good]

    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights)
    return np.interp(quantiles, cdf, values)


def median_and_1sigma(gd, param):
    """Weighted median with 16th/84th percentiles (+/- 1 sigma)."""
    i = gd.paramNames.list().index(param)
    q16, q50, q84 = weighted_quantile(gd.samples[:, i], [0.16, 0.50, 0.84], gd.weights)
    return q16, q50, q84


def posterior_summary(gd, param):
    """Median +/- 1 sigma, or a one-sided limit if the posterior piles up at a prior edge.

    The one-tail vs two-tail decision uses GetDist's 68% marginalized limits,
    exactly as the original triangle_plot(title_limit=1) titles did.

    Returns a dict with "kind" in {"twotail", "upper", "lower", "none"}.
    """
    stats = gd.getMargeStats().parWithName(param)
    lim68 = stats.limits[0]
    lim_quote = stats.limits[UPPER_LIMIT_LEVEL]

    if lim68.onetail_upper:
        return {"kind": "upper", "value": lim_quote.upper}
    if lim68.onetail_lower:
        return {"kind": "lower", "value": lim_quote.lower}
    if lim68.twotail:
        q16, q50, q84 = median_and_1sigma(gd, param)
        return {"kind": "twotail", "q16": q16, "q50": q50, "q84": q84}
    return {"kind": "none"}


def summary_latex(summary):
    """Right-hand side of the diagonal title, e.g. '= 1.30^{+0.38}_{-0.28}' or '< 5.75'."""
    kind = summary["kind"]
    if kind == "twotail":
        q16, q50, q84 = summary["q16"], summary["q50"], summary["q84"]
        return rf"= {q50:.2f}^{{+{q84 - q50:.2f}}}_{{-{q50 - q16:.2f}}}"
    if kind == "upper":
        return rf"< {summary['value']:.2f}"
    if kind == "lower":
        return rf"> {summary['value']:.2f}"
    return r"\ {\rm unconstrained}"


def summary_text(summary):
    """Plain-text version for the terminal printout."""
    kind = summary["kind"]
    if kind == "twotail":
        q16, q50, q84 = summary["q16"], summary["q50"], summary["q84"]
        return f"{q50:.3g} +{q84 - q50:.3g} -{q50 - q16:.3g}"
    if kind == "upper":
        return f"< {summary['value']:.3g}"
    if kind == "lower":
        return f"> {summary['value']:.3g}"
    return "unconstrained"


def draw_one_corner(g, subfig, gd, color, limits, panel_title, margins):
    """Draw one GetDist corner plot into a subfigure, with median +/- 1 sigma titles."""
    axes = subfig.subplots(3, 3)
    subfig.suptitle(panel_title, fontsize=CORNER_BLOCK_TITLE_FONTSIZE, y=CORNER_BLOCK_TITLE_Y)

    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            if col > row:
                ax.set_visible(False)
                continue

            xparam = PARAMS[col]

            if row == col:
                g.plot_1d(gd, xparam, ax=ax, colors=[color], normalized=False)
                summary = posterior_summary(gd, xparam)

                if summary["kind"] == "twotail":
                    # Median (dashed) and 16th-84th percentile band.
                    ax.axvline(summary["q50"], color=color, lw=1.0, ls="--")
                    ax.axvspan(summary["q16"], summary["q84"], color=color, alpha=0.12, lw=0)
                elif summary["kind"] == "upper":
                    # Shade from the prior edge up to the limit; dotted line at the limit.
                    edge = gd.ranges.getLower(xparam)
                    ax.axvspan(edge, summary["value"], color=color, alpha=0.12, lw=0)
                    ax.axvline(summary["value"], color=color, lw=1.0, ls=":")
                elif summary["kind"] == "lower":
                    edge = gd.ranges.getUpper(xparam)
                    ax.axvspan(summary["value"], edge, color=color, alpha=0.12, lw=0)
                    ax.axvline(summary["value"], color=color, lw=1.0, ls=":")

                ax.set_ylim(0.0, 1.08)
                ax.set_yticks([])

                label = PARAM_LABELS[col].strip("$")
                ax.set_title(
                    rf"${label} {summary_latex(summary)}$",
                    fontsize=CORNER_VALUE_FONTSIZE,
                )
            else:
                yparam = PARAMS[row]
                g.plot_2d(
                    gd, xparam, yparam, ax=ax,
                    filled=True, colors=[color], add_legend_proxy=False,
                )
                ax.set_ylim(*limits[yparam])

            ax.set_xlim(*limits[xparam])
            ticksize(ax, CORNER_TICK_FONTSIZE)
            ax.grid(False)

            # Only the outer edges of each corner get labels.
            ax.set_xlabel(PARAM_LABELS[col] if row == 2 else "", fontsize=CORNER_LABEL_FONTSIZE)
            if row != 2:
                ax.tick_params(labelbottom=False)
            if col == 0 and row > 0:
                ax.set_ylabel(PARAM_LABELS[row], fontsize=CORNER_LABEL_FONTSIZE)
            else:
                ax.set_ylabel("")
                if row != col:
                    ax.tick_params(labelleft=False)

    subfig.subplots_adjust(**margins,
                           wspace=CORNER_PANEL_WSPACE, hspace=CORNER_PANEL_HSPACE)


def plot_corners(samples, out_dir, job_prefix):
    """Combine all six GetDist corner plots into one 2x3 figure.

    Top row: major axis. Bottom row: minor axis. Columns: stellar-mass bin.
    Major and minor share axis limits within each mass bin so they can be
    compared by eye.
    """
    out_path = out_dir / f"{job_prefix}_corners_2x3.pdf"

    # GetDist's plotter needs make_figure() to initialize its internal state
    # (line-width/font scaling etc.). We then draw into our own axes via ax=,
    # so its placeholder figure is closed straight away. subplot_size is
    # roughly the width of one corner cell in the final figure (inches).
    g = gdplt.get_subplot_plotter(subplot_size=CORNER_GETDIST_SIZE)
    g.make_figure(nplot=1)
    plt.close(g.fig)

    fig = plt.figure(figsize=CORNER_FIGSIZE)
    # Give the two rows different heights so the panels themselves come out
    # the same size, even though the rows reserve different title space.
    row_margins = [CORNER_MARGINS_TOP_ROW, CORNER_MARGINS_BOTTOM_ROW]
    height_ratios = [1.0 / (m["top"] - m["bottom"]) for m in row_margins]
    subfigs = fig.subfigures(
        2, 3, wspace=CORNER_BLOCK_WSPACE, hspace=CORNER_BLOCK_HSPACE,
        height_ratios=height_ratios,
    )

    for mass_bin in range(3):
        limits = posterior_limits(
            [samples[(mass_bin, s)] for s in SECTORS]
        )
        for row, sector in enumerate(SECTORS):
            # The mass-bin label goes on the top row only.
            if row == 0:
                title = f"{MASS_BIN_LABELS[mass_bin]}\n{sector.capitalize()} axis"
            else:
                title = f"{sector.capitalize()} axis"
            draw_one_corner(
                g,
                subfigs[row, mass_bin],
                samples[(mass_bin, sector)],
                SECTOR_COLORS[sector],
                limits,
                title,
                row_margins[row],
            )

    fig.savefig(out_path, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)
    return out_path


def plot_posteriors(samples, out_path):
    fig, axes = plt.subplots(3, 3, figsize=POSTERIOR_FIGSIZE, sharey=False)

    for col, mass_bin in enumerate(range(3)):
        for row, (param, plabel) in enumerate(zip(PARAMS, PARAM_LABELS)):
            ax = axes[row, col]
            for sector in SECTORS:
                gd = samples[(mass_bin, sector)]
                density = gd.get1DDensity(param)
                x = np.asarray(density.x)
                y = np.asarray(density.P)
                if np.max(y) > 0:
                    y = y / np.max(y)
                ax.plot(
                    x,
                    y,
                    color=SECTOR_COLORS[sector],
                    label=sector.capitalize(),
                )

            if row == 0:
                ax.set_title(MASS_BIN_LABELS[mass_bin], **fontsize(POSTERIOR_TITLE_FONTSIZE))
            if col == 0:
                ax.set_ylabel(f"{plabel}\nNormalized density",
                              **fontsize(POSTERIOR_LABEL_FONTSIZE))
            ax.set_xlabel(plabel, **fontsize(POSTERIOR_LABEL_FONTSIZE))
            ticksize(ax, POSTERIOR_TICK_FONTSIZE)
            ax.set_ylim(bottom=0)
            if row == 0 and col == 0:
                ax.legend(frameon=False, **fontsize(POSTERIOR_LEGEND_FONTSIZE))

    fig.tight_layout(pad=POSTERIOR_PAD, w_pad=POSTERIOR_W_PAD, h_pad=POSTERIOR_H_PAD)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)


def plot_profiles(samples, roots, out_path):
    # Build all six models before creating the figure. Loading Popik2026 imports
    # PlotsTables.py, which calls plt.style.use('dark_background') and would
    # otherwise turn the stats-box and legend text white mid-plot.
    models = {
        key: load_model_at_best(roots[key], samples[key])
        for key in roots
    }
    set_plot_style("default")

    fig, axes = plt.subplots(1, 3, figsize=PROFILE_FIGSIZE, sharey=True)

    all_stats = {}

    for mass_bin, ax in enumerate(axes):
        sector_colors = {}
        stat_lines = []

        for sector in SECTORS:
            likeli, theory_1h, theory_2h, theory_tot, stats = models[(mass_bin, sector)]
            all_stats[(mass_bin, sector)] = stats

            R = np.asarray(likeli.R.to_value("arcmin"), dtype=float)
            data = np.asarray(likeli.meas_data.value, dtype=float)
            cov = np.asarray(likeli.meas_cov.value, dtype=float)
            err = np.sqrt(np.diag(cov))

            # Offset major/minor data points slightly in theta_d so their
            # markers/error bars do not lie directly on top of one another.
            color = SECTOR_COLORS[sector]
            sector_colors[sector] = color
            R_data = R + POINT_X_OFFSETS[sector]

            ax.errorbar(
                R_data,
                data * 1e6,
                yerr=err * 1e6,
                fmt="o",
                ms=PROFILE_MARKER_SIZE,
                capsize=PROFILE_CAPSIZE,
                color=color,
                label=f"{sector.capitalize()} data",
            )

            ax.plot(
                R,
                theory_tot * 1e6,
                color=color,
                lw=2.0,
                ls="-",
                label=f"{sector.capitalize()} total",
            )
            ax.plot(
                R,
                theory_1h * 1e6,
                color=color,
                lw=1.6,
                ls="--",
                label=f"{sector.capitalize()} 1h",
            )
            ax.plot(
                R,
                theory_2h * 1e6,
                color=color,
                lw=1.6,
                ls=":",
                label=f"{sector.capitalize()} 2h",
            )

            stat_lines.append(
                rf"{sector.capitalize()}: $\chi^2={stats['chi2']:.2f}$"
            )

        ax.axhline(0.0, lw=0.8, ls="--", color="0.5")
        ax.set_title(MASS_BIN_LABELS[mass_bin], **fontsize(PROFILE_TITLE_FONTSIZE))
        ax.set_xlabel(r"$\theta_d\ [{\rm arcmin}]$", **fontsize(PROFILE_LABEL_FONTSIZE))
        if mass_bin == 0:
            ax.set_ylabel(r"$y_{\rm CAP}\ [10^{-6}\,{\rm arcmin}^2]$",
                          **fontsize(PROFILE_LABEL_FONTSIZE))
        ticksize(ax, PROFILE_TICK_FONTSIZE)

        ax.text(
            0.03,
            0.97,
            "\n".join(stat_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=PROFILE_STATS_FONTSIZE,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"),
        )

    # One figure-level legend keeps every panel clear of the chi^2/PTE box.
    # Marker colors identify major/minor; line styles identify total/1h/2h.
    sector_handles = [
        Line2D(
            [],
            [],
            color=SECTOR_COLORS[s],
            marker="o",
            ls="none",
            markersize=5.5,
            label=s.capitalize(),
        )
        for s in SECTORS
    ]
    component_handles = [
        Line2D([], [], color="0.25", ls="-", lw=2.0, label="Total"),
        Line2D([], [], color="0.25", ls="--", lw=1.6, label="1-halo"),
        Line2D([], [], color="0.25", ls=":", lw=1.8, label="2-halo"),
    ]
    fig.legend(
        handles=sector_handles + component_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, PROFILE_LEGEND_Y),
        ncol=5,
        frameon=False,
        fontsize=PROFILE_LEGEND_FONTSIZE,
        columnspacing=1.8,
        handlelength=2.4,
    )

    fig.tight_layout(rect=(0.0, PROFILE_LEGEND_SPACE, 1.0, 1.0),
                     pad=PROFILE_PAD, w_pad=PROFILE_W_PAD)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)

    return all_stats


def main():
    args = parse_args()
    set_plot_style("default")
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(OUTPUT_PATH) / f"{args.job_prefix}_combined"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    roots = {}
    samples = {}
    for mass_bin in range(3):
        for sector in SECTORS:
            root = chain_root(args.job_prefix, mass_bin, sector)
            roots[(mass_bin, sector)] = root
            if not Path(str(root) + ".input.yaml").exists():
                raise FileNotFoundError(
                    f"Missing completed/started chain root {root}.input.yaml"
                )
            samples[(mass_bin, sector)] = load_getdist(root, args.ignore_rows)

    posterior_path = out_dir / f"{args.job_prefix}_posteriors_3x3.pdf"
    profile_path = out_dir / f"{args.job_prefix}_profiles_1x3.pdf"

    plot_posteriors(samples, posterior_path)
    stats = plot_profiles(samples, roots, profile_path)
    corner_path = plot_corners(samples, out_dir, args.job_prefix)

    print(f"Saved {posterior_path}")
    print(f"Saved {profile_path}")
    print(f"Saved {corner_path}")

    level = ["68%", "95%"][UPPER_LIMIT_LEVEL]
    print(f"\nPosterior summaries (median +/- 1 sigma; one-sided limits at {level}):")
    for mass_bin in range(3):
        for sector in SECTORS:
            parts = []
            for param in PARAMS:
                summary = posterior_summary(samples[(mass_bin, sector)], param)
                parts.append(f"{param} {summary_text(summary)}")
            print(f"  bin {mass_bin} {sector:5s}: " + ", ".join(parts))

    print("\nMaximum-likelihood chain-sample goodness of fit:")
    for mass_bin in range(3):
        for sector in SECTORS:
            s = stats[(mass_bin, sector)]
            p = s["params"]
            print(
                f"  bin {mass_bin} {sector:5s}: "
                f"chi2={s['chi2']:.4f}, N={s['N']}, "
                f"chi2/N={s['chi2_over_N']:.4f}, PTE={s['pte']:.5f}; "
                f"P0={p['P0']:.5g}, beta_P={p['beta_P']:.5g}, "
                f"A_2h_P={p['A_2h_P']:.5g}"
            )


if __name__ == "__main__":
    main()
