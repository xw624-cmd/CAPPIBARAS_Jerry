#!/usr/bin/env python3
"""Make combined plots from all six independent CAPPIBARAS chains.

The six GetDist corner plots are combined into one 2x3 figure (major on top,
minor on bottom). Diagonal titles report marginalized posterior modes and 68% highest-posterior-
density (HPD) regions. Independent minor-minus-major posterior differences
are analyzed with Nick's KDE density-ordering significance convention.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml
import cobaya
import getdist.plots as gdplt
from matplotlib.lines import Line2D
from scipy.special import erfinv
from scipy.stats import gaussian_kde

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

# ---- Posterior summaries and independent-chain differences -------------------
HPD_MASS = 0.68  # 68% highest-posterior-density (not equal-tail) region
# Numerical KDE for differences. Smaller KDE fit sample keeps O(N^2) density
# evaluation reasonably fast while preserving the code's sample-based statistic.
DEFAULT_DIFFERENCE_DRAWS = 10000
DEFAULT_KDE_FIT_DRAWS = 3000
DEFAULT_RANDOM_SEED = 2026
DIFFERENCE_FIGSIZE = (12, 9)

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
CORNER_VALUE_FONTSIZE = 11          # mode + 68% HPD titles on the diagonal
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
    p.add_argument(
        "--difference-draws", type=int, default=DEFAULT_DIFFERENCE_DRAWS,
        help="Independent weighted minor/major posterior draws per parameter.",
    )
    p.add_argument(
        "--kde-fit-draws", type=int, default=DEFAULT_KDE_FIT_DRAWS,
        help="Max difference draws used to fit each 1D Gaussian KDE.",
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_RANDOM_SEED,
        help="Reproducible seed for posterior difference draws.",
    )
    args = p.parse_args()
    if args.difference_draws < 1000 or args.kde_fit_draws < 100:
        p.error("Need --difference-draws >= 1000 and --kde-fit-draws >= 100")
    return args


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

    return info, params


def load_model_at_best(root, gd):
    """Return component curves, exact forward-model total, and chi2."""
    info, params = best_chain_params(root, gd)

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

    # These separately projected components are useful for visualization.
    # The native beam convolution uses LOG-space interpolation before the
    # transform, so projecting 1h and 2h separately and adding them is not
    # mathematically identical to projecting their SUM.
    theory_1h = np.asarray(result["1h"], dtype=float)
    theory_2h = np.asarray(result["2h"], dtype=float)

    # Always plot and assess goodness-of-fit using the ACTUAL Cobaya
    # forward-model prediction, never a sum-of-components approximation.
    theory_tot = np.asarray(likeli.model(params).value, dtype=float)
    data = np.asarray(likeli.data.y, dtype=float)
    inv_cov = np.asarray(likeli.data.inv_cov, dtype=float)
    if theory_tot.shape != data.shape:
        raise ValueError(
            f"Theory/data vector mismatch for {root}: "
            f"{theory_tot.shape} versus {data.shape}"
        )
    if not all(np.all(np.isfinite(a)) for a in
               (theory_1h, theory_2h, theory_tot, data)):
        raise ValueError(f"Non-finite model or data values for {root}")
    if theory_1h.shape != data.shape or theory_2h.shape != data.shape:
        raise ValueError(f"Component/data vector mismatch for {root}")

    delta = data - theory_tot
    chi2 = float(delta @ inv_cov @ delta)

    # Diagnostic only: quantifies the non-additivity caused by the native
    # beam's log-interpolation. A discrepancy is NOT used in chi-square.
    sigma = np.sqrt(np.diag(np.asarray(likeli.meas_cov.value, dtype=float)))
    max_decomposition_residual_sigma = float(
        np.max(np.abs(theory_tot - theory_1h - theory_2h) / sigma)
    )
    if max_decomposition_residual_sigma > 0.05:
        print(
            f"Note: {root}: 1h + 2h curves differ from the native total "
            f"by up to {max_decomposition_residual_sigma:.3g} data-sigma "
            "(beam interpolation is not exactly additive)."
        )

    stats = {
        "chi2": chi2,
        "params": {name: params[name] for name in PARAMS},
    }

    return likeli, theory_1h, theory_2h, theory_tot, stats


def posterior_limits(roots, params=PARAMS):
    """Shared major/minor axis ranges from weighted posterior quantiles."""
    limits = {}
    for param in params:
        lows, highs = [], []
        for gd in roots:
            i = gd.paramNames.list().index(param)
            lo, hi = weighted_quantile(
                gd.samples[:, i], [0.00135, 0.99865], gd.weights
            )
            prior_lo = gd.ranges.getLower(param)
            prior_hi = gd.ranges.getUpper(param)
            if prior_lo is not None:
                lo = max(float(lo), float(prior_lo))
            if prior_hi is not None:
                hi = min(float(hi), float(prior_hi))
            lows.append(float(lo))
            highs.append(float(hi))

        lo, hi = min(lows), max(highs)
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            raise ValueError(f"Invalid posterior range for {param}: {lo}, {hi}")
        limits[param] = (lo, hi)
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


def hpd_regions(x, density, mass=HPD_MASS):
    """Continuous 1D HPD regions at probability `mass` from a smooth KDE.

    Finds a density threshold such that the integrated probability above it is
    `mass`. Intersections with the threshold are interpolated linearly. Returns
    all intervals, since genuinely multimodal posteriors have disjoint HPDs.
    """
    x = np.asarray(x, dtype=float)
    y = np.maximum(np.asarray(density, dtype=float), 0.0)
    if x.ndim != 1 or x.size != y.size or x.size < 3:
        raise ValueError("HPD requires matching 1D density grids of >= 3 points")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Nonfinite 1D density")
    if not np.all(np.diff(x) > 0) or not (0 < mass < 1):
        raise ValueError("HPD grid must increase and mass must be in (0,1)")
    area = float(np.trapezoid(y, x))
    if area <= 0:
        raise ValueError("Cannot compute HPD from zero-density curve")
    y = y / area

    def coverage(threshold):
        above0 = y[:-1] >= threshold
        above1 = y[1:] >= threshold
        widths = np.diff(x)
        intact = above0 & above1
        part = np.zeros_like(widths)
        part[intact] = 0.5 * (y[:-1][intact] + y[1:][intact]) * widths[intact]
        cross = above0 ^ above1
        if np.any(cross):
            frac = (threshold - y[:-1][cross]) / (y[1:][cross] - y[:-1][cross])
            selected_fraction = np.where(above0[cross], frac, 1.0 - frac)
            part[cross] = (threshold + np.where(
                above0[cross], y[:-1][cross], y[1:][cross]
            )) * selected_fraction * widths[cross] / 2.0
        return float(part.sum())

    lo = 0.0
    hi = float(y.max())
    for _ in range(65):
        middle = (lo + hi) / 2.0
        if coverage(middle) >= mass:
            lo = middle
        else:
            hi = middle
    level = 0.5 * (lo + hi)

    intervals = []
    active_start = float(x[0]) if y[0] >= level else None
    for i in range(x.size - 1):
        a, b = y[i] >= level, y[i + 1] >= level
        if a != b:
            xcross = float(x[i] + (level - y[i]) *
                           (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
            if not a and b:
                active_start = xcross
            elif a and not b and active_start is not None:
                intervals.append((active_start, xcross))
                active_start = None
    if active_start is not None:
        intervals.append((active_start, float(x[-1])))
    return intervals


def density_summary(x, density, mass=HPD_MASS):
    """1D marginalized mode and HPD region(s) from the plotted density."""
    x = np.asarray(x, dtype=float)
    density = np.asarray(density, dtype=float)
    if not np.all(np.isfinite(density)):
        raise ValueError("Nonfinite density")
    mode = float(x[np.argmax(density)])
    return {"mode": mode, "intervals": hpd_regions(x, density, mass)}


def posterior_summary(gd, param):
    """Use exactly the same GetDist 1D KDE as the plotted diagonal curve."""
    density = gd.get1DDensity(param)
    return density_summary(density.x, density.P)


def format_intervals(intervals, precision=3):
    return " U ".join(
        f"[{lo:.{precision}g}, {hi:.{precision}g}]"
        for lo, hi in intervals
    )


def summary_latex(summary):
    """Diagonal title: mode plus asymmetric 68% HPD errors if connected."""
    mode = summary["mode"]
    intervals = summary["intervals"]
    if len(intervals) == 1:
        low, high = intervals[0]
        return rf"= {mode:.2f}^{{+{high - mode:.2f}}}_{{-{mode - low:.2f}}}"
    # A multimodal HPD is a union; one error-bar pair would misrepresent it.
    return rf"= {mode:.2f}\ (\mathrm{{68\%\ HPD:multi}})"


def summary_text(summary):
    mode = summary["mode"]
    intervals = summary["intervals"]
    return f"mode={mode:.5g}; HPD68={format_intervals(intervals, 5)}"


def sample_parameter(gd, param, rng, size):
    """Draw from a GetDist weighted posterior; never ignore chain weights."""
    names = gd.paramNames.list()
    if param not in names:
        raise KeyError(f"Missing parameter {param} in chain")
    vals = np.asarray(gd.samples[:, names.index(param)], dtype=float)
    weights = np.asarray(gd.weights, dtype=float)
    good = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)
    vals, weights = vals[good], weights[good]
    if vals.size < 10:
        raise ValueError(f"Insufficient valid samples for {param}")
    return rng.choice(vals, size=size, replace=True, p=weights / weights.sum())


def difference_summary(major, minor, param, rng, draws, kde_fit_draws):
    """Compute Nick's KDE density-ordering significance for minor minus major.

    Assumes independent major/minor posteriors (product distribution). This is
    NOT a joint-sector likelihood; cross-sector covariance is not propagated.
    """
    a = sample_parameter(minor, param, rng, draws)
    b = sample_parameter(major, param, rng, draws)
    delta = a - b
    # A KDE on the full chain can be extremely expensive; select a random
    # subset for KDE fitting, then evaluate the density at ALL paired draws.
    nfit = min(kde_fit_draws, draws)
    indices = rng.choice(draws, nfit, replace=False)
    kde = gaussian_kde(delta[indices])
    density_zero = float(kde([0.0])[0])
    density_at_samples = np.asarray(kde(delta), dtype=float)

    # Nick's exact code concept: p = mean(kde(delta) > kde(0)).
    p = float(np.mean(density_at_samples > density_zero))
    p_positive = float(np.mean(delta > 0))

    # For p==1 the MC estimate is saturated; return a conservative floor in
    # absolute Gaussian sigma, rather than the meaningless 'infinite sigma'.
    limited = p >= 1.0
    p_for_inverse = min(p, 1.0 - 0.5 / draws)
    sigma_abs = float(np.sqrt(2.0) * erfinv(p_for_inverse))
    # A sign is NOT part of Nick's two-sided density-ranking statistic.
    # Supplement with posterior side probability to indicate direction.
    direction = 1 if p_positive > 0.5 else -1 if p_positive < 0.5 else 0
    sigma_signed = direction * sigma_abs

    # Evaluate KDE on a grid extending beyond sample extrema to capture tails.
    scale = float(np.std(delta, ddof=1))
    pad = max(0.15 * (float(delta.max()) - float(delta.min())), 2.5 * kde.factor * scale)
    low, high = float(delta.min() - pad), float(delta.max() + pad)
    if low > 0:
        low = 0.0 - pad
    if high < 0:
        high = 0.0 + pad
    x = np.linspace(low, high, 1200)
    y = np.asarray(kde(x), dtype=float)
    summary = density_summary(x, y)
    summary.update({
        "draws": delta,
        "x": x,
        "density": y,
        "prob_minor_gt_major": p_positive,
        "density_rank_mass": p,
        "sigma_signed": sigma_signed,
        "sigma_limited": limited,
    })
    return summary


def plot_differences(samples, out_path, rng, draws, kde_fit_draws):
    """Nine minor-minus-major distributions (3 params x 3 mass bins)."""
    fig, axes = plt.subplots(3, 3, figsize=DIFFERENCE_FIGSIZE)
    all_results = {}
    for mass_bin in range(3):
        for row, (param, label) in enumerate(zip(PARAMS, PARAM_LABELS)):
            result = difference_summary(
                samples[(mass_bin, "major")], samples[(mass_bin, "minor")],
                param, rng, draws, kde_fit_draws,
            )
            all_results[(mass_bin, param)] = result
            ax = axes[row, mass_bin]
            x, y = result["x"], result["density"]
            if y.max() > 0:
                y = y / y.max()
            ax.plot(x, y, color=CAP_MINOR_COLOR, lw=1.7)
            for lower, upper in result["intervals"]:
                mask = (x >= lower) & (x <= upper)
                ax.fill_between(x[mask], 0, y[mask], color=CAP_MINOR_COLOR, alpha=0.20)
            ax.axvline(0.0, color="black", ls=":", lw=1.1)
            ax.axvline(result["mode"], color=CAP_MINOR_COLOR, ls="--", lw=1.0)
            if result["sigma_limited"]:
                sigma_label = (
                    rf"$|n_\sigma|\geq {abs(result['sigma_signed']):.2f}$"
                )
            else:
                sigma_label = rf"$n_\sigma={result['sigma_signed']:+.2f}$"
            ax.text(
                0.97, 0.95,
                sigma_label + "\n"
                + rf"$P(\Delta>0)={result['prob_minor_gt_major']:.3f}$",
                ha="right", va="top", transform=ax.transAxes, fontsize=9,
            )
            ax.set_ylim(bottom=0)
            ax.set_yticks([])
            ax.set_xlabel(rf"$\Delta {label.strip('$')}$", fontsize=10)
            if mass_bin == 0:
                ax.set_ylabel("Density (peak normalized)", fontsize=9)
            if row == 0:
                ax.set_title(MASS_BIN_LABELS[mass_bin], fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)
    return all_results


def save_posterior_and_difference_tables(samples, differences, out_dir, job_prefix):
    """Machine-readable modes/HPD and minor-major KDE significances."""
    posterior_csv = out_dir / f"{job_prefix}_mode_HPD68.csv"
    with posterior_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mass_bin", "sector", "parameter", "mode", "HPD68_intervals",
                         "HPD68_low_if_connected", "HPD68_high_if_connected"])
        for bin_index in range(3):
            for sector in SECTORS:
                for param in PARAMS:
                    summary = posterior_summary(samples[(bin_index, sector)], param)
                    intervals = summary["intervals"]
                    writer.writerow([
                        bin_index, sector, param, summary["mode"],
                        format_intervals(intervals, 8),
                        intervals[0][0] if len(intervals) == 1 else "",
                        intervals[0][1] if len(intervals) == 1 else "",
                    ])
    delta_csv = out_dir / f"{job_prefix}_minor_minus_major.csv"
    with delta_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mass_bin", "parameter", "delta_convention", "mode",
                         "HPD68_intervals", "HPD68_low_if_connected",
                         "HPD68_high_if_connected", "P_minor_gt_major",
                         "KDE_density_rank_mass", "signed_gaussian_equivalent_sigma",
                         "sigma_is_lower_limit"])
        for bin_index in range(3):
            for param in PARAMS:
                summary = differences[(bin_index, param)]
                intervals = summary["intervals"]
                writer.writerow([
                    bin_index, param, "minor-major", summary["mode"],
                    format_intervals(intervals, 8),
                    intervals[0][0] if len(intervals) == 1 else "",
                    intervals[0][1] if len(intervals) == 1 else "",
                    summary["prob_minor_gt_major"],
                    summary["density_rank_mass"], summary["sigma_signed"],
                    summary["sigma_limited"],
                ])
    return posterior_csv, delta_csv


def draw_one_corner(g, subfig, gd, color, limits, panel_title, margins):
    """Draw one GetDist corner plot with mode and 68% HPD annotations."""
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

                # HPD regions may be disjoint; shade every connected component.
                for low, high in summary["intervals"]:
                    ax.axvspan(low, high, color=color, alpha=0.12, lw=0)
                ax.axvline(summary["mode"], color=color, lw=1.0, ls="--")

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

    # One figure-level legend keeps every panel clear of the chi^2 box.
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

    difference_path = out_dir / f"{args.job_prefix}_minor_minus_major_3x3.pdf"
    differences = plot_differences(
        samples, difference_path, np.random.default_rng(args.seed),
        args.difference_draws, args.kde_fit_draws,
    )
    posterior_csv, delta_csv = save_posterior_and_difference_tables(
        samples, differences, out_dir, args.job_prefix,
    )
    print(f"Saved {difference_path}")
    print(f"Saved {posterior_csv}")
    print(f"Saved {delta_csv}")

    print("\nMarginalized posterior mode and 68% HPD regions:")
    for mass_bin in range(3):
        for sector in SECTORS:
            for param in PARAMS:
                summary = posterior_summary(samples[(mass_bin, sector)], param)
                print(f"  bin {mass_bin} {sector:5s} {param}: {summary_text(summary)}")

    print("\nMinor-minus-major posteriors (independent-chain assumption):")
    print("  n_sigma: Nick's density-ranking Gaussian equivalent; ")
    print("  P(delta>0): fraction of paired posterior draws with minor > major.")
    for mass_bin in range(3):
        for param in PARAMS:
            summary = differences[(mass_bin, param)]
            sigma_text = (
                f"|n_sigma| >= {abs(summary['sigma_signed']):.3f}"
                if summary["sigma_limited"]
                else f"n_sigma = {summary['sigma_signed']:+.3f}"
            )
            print(
                f"  bin {mass_bin} {param}: {summary_text(summary)}; "
                f"{sigma_text}; "
                f"P(minor > major)={summary['prob_minor_gt_major']:.5f}"
            )

    print("\nBest-sampled-chain-point goodness of fit:")
    for mass_bin in range(3):
        for sector in SECTORS:
            s = stats[(mass_bin, sector)]
            p = s["params"]
            print(
                f"  {MASS_BIN_LABELS[mass_bin]} {sector:5s}: "
                f"chi2={s['chi2']:.4f}; "
                f"P0={p['P0']:.5g}, beta_P={p['beta_P']:.5g}, "
                f"A_2h_P={p['A_2h_P']:.5g}"
            )


if __name__ == "__main__":
    main()
