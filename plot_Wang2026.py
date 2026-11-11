import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import yaml
import cobaya
import getdist.plots as gdplt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.special import erfinv
from scipy.stats import gaussian_kde, chi2 as chi2_distribution
from config import OUTPUT_PATH, set_plot_style

PARAMS = ["P0", "beta_P", "A_2h_P"]
PARAM_LABELS = [r"$P_0$", r"$\beta_P$", r"$A_{2h,P}$"]
SECTORS = ["major", "minor"]

# Only these are shown in the corners; A_2h_P is still fitted and marginalized over.
CORNER_PARAMS = ["P0", "beta_P"]
CORNER_LABELS = [r"$P_0$", r"$\beta_P$"]

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
SAVE_PAD_INCHES = 0.02
HPD_MASS = 0.68
CONTOUR_LEVELS = np.array([0.68, 0.95])
UPPER_LIMIT_MASS = 0.95
CORNER_CONTOUR_PAD = 0.09
DEFAULT_DIFFERENCE_DRAWS = 10000
DEFAULT_RANDOM_SEED = 2026
DIFFERENCE_FIGSIZE = (15, 4.6)
DIFFERENCE_XLIMITS = [(-60, 60), (-5, 7.5), (-10, 20)]

# ---- Combined corner + profile figure layout (all lengths in inches) ----
CELL = 1.55                    # side length of one corner panel
LEFT_MARGIN = 1.05             # room for row labels, y tick labels, y label
RIGHT_MARGIN = 0.15
BOTTOM_MARGIN = 0.60           # room for bottom x tick labels and x labels
TOP_MARGIN = 0.55              # room for the mass-bin titles
COL_GAP = 0.85                 # gap between mass-bin columns
ROW_GAP = 0.65                 # gap between top of minor block (its profile) and bottom of major corner
PROFILE_PAD_LEFT = 0.85        # profile left edge, measured from the empty cell's left edge
PROFILE_PAD_BOTTOM = 0.75      # profile bottom edge, measured from the empty cell's bottom edge
PROFILE_W = 2.05               # profile panel width
PROFILE_H = 1.75               # profile panel height
PROFILE_SHARE_Y = True         # same y-range for major and minor profiles within a mass bin
PROFILE_SHOW_LEGEND = False     # legend in the first column's profile of each row
ROW_LABEL_X = 0.55             # x position of the "Major axis"/"Minor axis" row labels
CORNER_GETDIST_SIZE = 1.7
MASS_BIN_TITLE_FONTSIZE = 13
ROW_LABEL_FONTSIZE = 13
CORNER_VALUE_FONTSIZE = 10
CORNER_LABEL_FONTSIZE = 11
CORNER_TICK_FONTSIZE = 9
CORNER_XBINS = 6
COMBINED_PROFILE_LABEL_FONTSIZE = 9
COMBINED_PROFILE_TICK_FONTSIZE = 8
COMBINED_PROFILE_LEGEND_FONTSIZE = 9
COMBINED_PROFILE_MARKER_SIZE = 3.5
COMBINED_PROFILE_CAPSIZE = 2.0

POSTERIOR_FIGSIZE = (12, 9)
POSTERIOR_PAD = 0.2
POSTERIOR_W_PAD = 0.4
POSTERIOR_H_PAD = 0.4
POSTERIOR_TITLE_FONTSIZE = None
POSTERIOR_LABEL_FONTSIZE = None
POSTERIOR_TICK_FONTSIZE = None
POSTERIOR_LEGEND_FONTSIZE = None

# Residual figure
RESIDUAL_FIGSIZE = (16.5, 5.2)
RESIDUAL_PAD = 0.2
RESIDUAL_W_PAD = 0
RESIDUAL_LEGEND_SPACE = 0.08
RESIDUAL_LEGEND_Y = -0.015
RESIDUAL_LEGEND_FONTSIZE = 10
RESIDUAL_TITLE_FONTSIZE = 11
RESIDUAL_LABEL_FONTSIZE = 10
RESIDUAL_TICK_FONTSIZE = None
RESIDUAL_MARKER_SIZE = 4.5
RESIDUAL_CAPSIZE = 2.5


def fontsize(size):
    return {} if size is None else {"fontsize": size}


def ticksize(ax, size):
    if size is not None:
        ax.tick_params(labelsize=size)


def force_light_style():
    """White background and black text/axes for every figure, regardless of style defaults."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.transparent": False,
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "legend.facecolor": "white",
        "legend.edgecolor": "0.8",
    })


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job-prefix", default="Wang2026_gnfw_noradio_16")
    p.add_argument("--output-dir", default=None)
    p.add_argument(
        "--ignore-rows",
        type=float,
        default=0.3,
        help="Burn-in fraction removed from each GetDist chain before posterior summaries.",
    )
    p.add_argument(
        "--difference-draws", type=int, default=DEFAULT_DIFFERENCE_DRAWS,
        help="Independent weighted P0 minor/major posterior draws.",
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_RANDOM_SEED,
        help="Reproducible seed for posterior difference draws.",
    )
    args = p.parse_args()
    return args


def chain_root(job_prefix, mass_bin, sector):
    job = f"{job_prefix}_b{mass_bin}_{sector}"
    return Path(OUTPUT_PATH) / job / job


def load_getdist(root, ignore_rows):
    gd = cobaya.load_samples(str(root), to_getdist=True)
    if ignore_rows > 0:
        gd.removeBurnFraction(ignore_rows)
    gd.contours = CONTOUR_LEVELS.copy()
    return gd


def best_chain_params(root, gd):
    input_yaml = Path(str(root) + ".input.yaml")
    with open(input_yaml) as f:
        info = yaml.safe_load(f)
    finite = np.isfinite(gd.loglikes)
    candidate_rows = np.flatnonzero(finite)
    best_row = int(candidate_rows[np.argmin(gd.loglikes[finite])])
    names = gd.paramNames.list()
    col = {name: i for i, name in enumerate(names)}
    params = {}
    sampled = []
    for name, spec in info["params"].items():
        if isinstance(spec, dict) and "prior" in spec:
            params[name] = float(gd.samples[best_row, col[name]])
            sampled.append(name)
        elif isinstance(spec, dict) and "value" in spec:
            params[name] = spec["value"]
        else:
            params[name] = spec
    return info, params, sampled


def load_model_at_best(root, gd):
    info, params, sampled = best_chain_params(root, gd)
    model_info = {
        key: info[key]
        for key in ("likelihood", "params", "theory")
        if key in info
    }
    model = cobaya.get_model(model_info)
    likelihoods = model.likelihood
    likeli = likelihoods[next(iter(likelihoods))]
    theory_tot = np.asarray(likeli.model(params).value, dtype=float)
    data = np.asarray(likeli.data.y, dtype=float)
    inv_cov = np.asarray(likeli.data.inv_cov, dtype=float)
    delta = data - theory_tot
    chi2 = float(delta @ inv_cov @ delta)
    n_data = int(data.size)
    n_params = len(sampled)
    dof = n_data - n_params
    if dof <= 0:
        raise ValueError(
            f"{root}: {n_data} aperture radii but {n_params} fitted parameters "
            f"({sampled}); degrees of freedom = {dof}."
        )
    stats = {
        "chi2": chi2,
        "n_data": n_data,
        "n_params": n_params,
        "dof": dof,
        "reduced_chi2": chi2 / dof,
        "gaussian_pte": float(chi2_distribution.sf(chi2, df=dof)),
        "params": {name: params[name] for name in PARAMS},
    }
    return likeli, theory_tot, stats


def posterior_limits(roots, cache, params=PARAMS):
    extents = {param: [] for param in params}
    for key, gd in roots:
        for param in params:
            d1, summary = cache[(*key, param)]
            for low, high in hpd_regions(d1.x, d1.P, mass=UPPER_LIMIT_MASS):
                extents[param].append((low, high))
            if summary["kind"] in ("upper", "lower"):
                extents[param].append((summary["mode"], summary["limit"]))
        for i, xparam in enumerate(params):
            for yparam in params[i+1:]:
                d2 = gd.get2DDensityGridData(
                    xparam, yparam, num_plot_contours=2
                )
                threshold95 = float(d2.contours[1])
                density = np.asarray(d2.P)
                mask = density >= threshold95
                ys, xs = np.where(mask)
                x = np.asarray(d2.x, dtype=float)
                y = np.asarray(d2.y, dtype=float)
                extents[xparam].append((float(x[xs.min()]), float(x[xs.max()])))
                extents[yparam].append((float(y[ys.min()]), float(y[ys.max()])))
    limits = {}
    for param in params:
        lo = min(pair[0] for pair in extents[param])
        hi = max(pair[1] for pair in extents[param])
        pad = CORNER_CONTOUR_PAD * (hi - lo)
        lo, hi = lo - pad, hi + pad
        prior_lows = [gd.ranges.getLower(param) for _, gd in roots]
        prior_highs = [gd.ranges.getUpper(param) for _, gd in roots]
        low_bounds = [x for x in prior_lows if x is not None]
        high_bounds = [x for x in prior_highs if x is not None]
        if low_bounds:
            lo = max(lo, min(map(float, low_bounds)))
        if high_bounds:
            hi = min(hi, max(map(float, high_bounds)))
        limits[param] = (lo, hi)
    return limits


def weighted_quantile(values, quantiles, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[good], weights[good]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights)
    return np.interp(quantiles, cdf, values)


def hpd_regions(x, density, mass=HPD_MASS):
    x = np.asarray(x, dtype=float)
    y = np.maximum(np.asarray(density, dtype=float), 0.0)
    area = float(np.trapezoid(y, x))
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


def shortest_mode_interval(x, density, mass=HPD_MASS):
    """Narrowest connected interval containing the global mode and `mass` probability.

    This is NOT necessarily a highest-posterior-density (HPD) set when the
    marginalized posterior is multimodal. It covers 68% in one interval.
    """
    x = np.asarray(x, dtype=float)
    y = np.maximum(np.asarray(density, dtype=float), 0.0)
    cdf = np.r_[0.0, np.cumsum(0.5 * (y[:-1] + y[1:]) * np.diff(x))]
    cdf /= cdf[-1]
    mode = float(x[np.argmax(y)])

    # For each possible lower boundary, find the upper boundary at +68% mass.
    feasible = (x <= mode) & (cdf <= 1.0 - mass)
    lows = x[feasible]
    highs = np.interp(cdf[feasible] + mass, cdf, x)
    includes_mode = highs >= mode
    lows, highs = lows[includes_mode], highs[includes_mode]
    i = np.argmin(highs - lows)
    return float(lows[i]), float(highs[i])


def density_summary(x, density, mass=HPD_MASS):
    x = np.asarray(x, dtype=float)
    density = np.asarray(density, dtype=float)
    mode = float(x[np.argmax(density)])
    return {"mode": mode, "intervals": hpd_regions(x, density, mass)}


def posterior_summary(gd, param, density, marge_stats):
    summary = density_summary(density.x, density.P)
    lo_prior = gd.ranges.getLower(param)
    hi_prior = gd.ranges.getUpper(param)
    width = float(density.x[-1] - density.x[0])
    atol = max(width / max(1, len(density.x) - 1) * 1.5, 1e-10)
    hpd_lo = min(lower for lower, _ in summary["intervals"])
    hpd_hi = max(upper for _, upper in summary["intervals"])
    touches_lower = lo_prior is not None and hpd_lo <= float(lo_prior) + atol
    touches_upper = hi_prior is not None and hpd_hi >= float(hi_prior) - atol
    onetail68 = marge_stats.parWithName(param).limits[0]
    upper = bool(onetail68.onetail_upper) or (touches_lower and not touches_upper)
    lower = bool(onetail68.onetail_lower) or (touches_upper and not touches_lower)
    if upper != lower:
        index = gd.paramNames.list().index(param)
        quantile = UPPER_LIMIT_MASS if upper else 1.0 - UPPER_LIMIT_MASS
        val = weighted_quantile(gd.samples[:, index], [quantile], gd.weights)[0]
        summary.update(kind="upper" if upper else "lower", limit=float(val))
    else:
        summary["kind"] = "twotail"
    # For two-sided summaries, show a single contiguous 68% interval
    # around the marginal mode. The standard (possibly split) HPD remains
    # used above to identify prior-boundary contact.
    if summary["kind"] == "twotail":
        summary["intervals"] = [shortest_mode_interval(density.x, density.P)]
    return summary


def summarize(summary, latex=False):
    kind = summary.get("kind", "twotail")
    mode = summary["mode"]
    if kind in ("upper", "lower"):
        op = "<" if kind == "upper" else ">"
        if latex:
            return rf"{op} {summary['limit']:.2f}"
        return f"mode={mode:.5g}; {op} {summary['limit']:.5g} (95% {kind} limit)"
    intervals = summary["intervals"]
    low, high = intervals[0]
    if latex:
        return rf"= {mode:.2f}^{{+{high - mode:.2f}}}_{{-{mode - low:.2f}}}"
    return f"mode={mode:.5g}; contiguous68=[{low:.5g}, {high:.5g}]"


def prepare_posteriors(samples):
    cache = {}
    for (mass_bin, sector), gd in samples.items():
        marge_stats = gd.getMargeStats()
        for param in PARAMS:
            density = gd.get1DDensity(param)
            cache[(mass_bin, sector, param)] = (
                density, posterior_summary(gd, param, density, marge_stats)
            )
    return cache


def sample_parameter(gd, param, rng, size):
    names = gd.paramNames.list()
    vals = np.asarray(gd.samples[:, names.index(param)], dtype=float)
    weights = np.asarray(gd.weights, dtype=float)
    good = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)
    vals, weights = vals[good], weights[good]
    return rng.choice(vals, size=size, replace=True, p=weights / weights.sum())


def difference_summary(major, minor, param, rng, draws):
    a = sample_parameter(minor, param, rng, draws)
    b = sample_parameter(major, param, rng, draws)
    delta = a - b
    kde = gaussian_kde(delta)
    density_zero = float(kde([0.0])[0])
    density_at_samples = np.asarray(kde(delta), dtype=float)
    p = float(np.mean(density_at_samples > density_zero))
    limited = p >= 1.0
    sigma_abs = float(np.sqrt(2.0) * erfinv(p))
    sigma_signed = np.sign(np.mean(delta > 0) - 0.5) * sigma_abs
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
        "x": x,
        "density": y,
        "prob_minor_gt_major": p,
        "sigma_signed": sigma_signed,
        "sigma_limited": limited,
    })
    return summary


def plot_differences(samples, out_path, seed, draws):
    param = "P0"
    fig, axes = plt.subplots(1, 3, figsize=DIFFERENCE_FIGSIZE, sharey=False)
    all_results = {}
    for mass_bin, ax in enumerate(axes):
        rng = np.random.default_rng([seed, mass_bin])
        result = difference_summary(
            samples[(mass_bin, "major")], samples[(mass_bin, "minor")],
            param, rng, draws,
        )
        all_results[mass_bin] = result
        x, y = result["x"], result["density"]
        ax.plot(x, y, color=CAP_MINOR_COLOR, lw=1.7)
        for lower, upper in result["intervals"]:
            mask = (x >= lower) & (x <= upper)
            ax.fill_between(x[mask], 0, y[mask], color=CAP_MINOR_COLOR, alpha=0.20)
        ax.axvline(result["mode"], color=CAP_MINOR_COLOR, ls="--", lw=1.0)
        if result["sigma_limited"]:
            sigma_label = rf"$|n_\sigma|\geq {abs(result['sigma_signed']):.2f}$"
        else:
            sigma_label = rf"$n_\sigma={result['sigma_signed']:+.2f}$"
        ax.text(
            0.97, 0.95,
            sigma_label,
            ha="right", va="top", transform=ax.transAxes, fontsize=9,
        )
        ax.set_ylim(bottom=0)
        ax.set_xlim(*DIFFERENCE_XLIMITS[mass_bin])
        ax.set_xlabel(r"$\Delta P_0$", fontsize=10)
        ax.set_title(MASS_BIN_LABELS[mass_bin], fontsize=11)
        ax.set_ylabel(r"Probability density" if mass_bin == 0 else "", fontsize=10)
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)
    return all_results


# ---------------------------------------------------------------------------
# Combined corner + profile figure
# ---------------------------------------------------------------------------

def combined_layout():
    """Return figure size, block width, and corner origins (bottom-left, inches)."""
    n = len(CORNER_PARAMS)
    corner_h = n * CELL
    block_w = max(n * CELL, (n - 1) * CELL + PROFILE_PAD_LEFT + PROFILE_W)
    # Each row's block height includes the profile sticking out above its corner.
    block_h = max(corner_h, (n - 1) * CELL + PROFILE_PAD_BOTTOM + PROFILE_H)
    fig_w = LEFT_MARGIN + 3 * block_w + 2 * COL_GAP + RIGHT_MARGIN
    fig_h = BOTTOM_MARGIN + 2 * block_h + ROW_GAP + TOP_MARGIN
    y_minor = BOTTOM_MARGIN
    y_major = BOTTOM_MARGIN + block_h + ROW_GAP
    origins = {}
    for mass_bin in range(3):
        x0 = LEFT_MARGIN + mass_bin * (block_w + COL_GAP)
        origins[(mass_bin, "major")] = (x0, y_major)
        origins[(mass_bin, "minor")] = (x0, y_minor)
    return fig_w, fig_h, block_w, origins


def add_axes_inches(fig, fig_w, fig_h, left, bottom, width, height):
    return fig.add_axes([left / fig_w, bottom / fig_h, width / fig_w, height / fig_h])


def draw_corner(fig, fig_w, fig_h, g, gd, key, cache, color, limits, x0, y0):
    n = len(CORNER_PARAMS)
    for row in range(n):
        for col in range(row + 1):
            left = x0 + col * CELL
            bottom = y0 + (n - 1 - row) * CELL
            ax = add_axes_inches(fig, fig_w, fig_h, left, bottom, CELL, CELL)
            xparam = CORNER_PARAMS[col]
            if row == col:
                g.plot_1d(gd, xparam, ax=ax, colors=[color], normalized=False)
                summary = cache[(*key, xparam)][1]
                if summary["kind"] in ("upper", "lower"):
                    edge = gd.ranges.getLower(xparam) if summary["kind"] == "upper" \
                        else gd.ranges.getUpper(xparam)
                    if edge is not None:
                        ax.axvspan(edge, summary["limit"], color=color, alpha=0.12, lw=0)
                    ax.axvline(summary["limit"], color=color, lw=1.1, ls=":")
                else:
                    # One shaded interval containing exactly 68% probability.
                    # The interval is contiguous, not necessarily an HPD set.
                    for low, high in summary["intervals"]:
                        ax.axvspan(low, high, color=color, alpha=0.12, lw=0)
                if summary["kind"] == "twotail":
                    ax.axvline(summary["mode"], color=color, lw=1.0, ls="--")
                ax.set_ylim(0.0, 1.08)
                ax.set_yticks([])
                label = CORNER_LABELS[col].strip("$")
                ax.set_title(
                    rf"${label} {summarize(summary, latex=True)}$",
                    fontsize=CORNER_VALUE_FONTSIZE,
                )
            else:
                yparam = CORNER_PARAMS[row]
                g.plot_2d(
                    gd, xparam, yparam, ax=ax,
                    filled=True, colors=[color], add_legend_proxy=False,
                )
                ax.set_ylim(*limits[yparam])
            ax.set_xlim(*limits[xparam])
            # Drop the last x tick label on inner columns so it doesn't
            # collide with the first label of the next column.
            if col < n - 1:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=CORNER_XBINS, prune="upper"))
            ticksize(ax, CORNER_TICK_FONTSIZE)
            ax.grid(False)
            ax.set_xlabel(CORNER_LABELS[col] if row == n - 1 else "",
                          fontsize=CORNER_LABEL_FONTSIZE)
            if row != n - 1:
                ax.tick_params(labelbottom=False)
            if col == 0 and row > 0:
                ax.set_ylabel(CORNER_LABELS[row], fontsize=CORNER_LABEL_FONTSIZE)
            else:
                ax.set_ylabel("")
                if row != col:
                    ax.tick_params(labelleft=False)


def draw_profile(ax, models, mass_bin, sector, show_legend):
    """Data and best-fit total model for a single sector."""
    likeli, theory_tot, _ = models[(mass_bin, sector)]
    R = np.asarray(likeli.R.to_value("arcmin"), dtype=float)
    data = np.asarray(likeli.meas_data.value, dtype=float)
    cov = np.asarray(likeli.meas_cov.value, dtype=float)
    err = np.sqrt(np.diag(cov))
    color = SECTOR_COLORS[sector]
    handle = ax.errorbar(
        R,
        data * 1e6,
        yerr=err * 1e6,
        fmt="o",
        ms=COMBINED_PROFILE_MARKER_SIZE,
        capsize=COMBINED_PROFILE_CAPSIZE,
        color=color,
        label=f"{sector.capitalize()}-axis",
    )
    ax.plot(R, theory_tot * 1e6, color=color, lw=1.6, ls="-")
    ax.set_xlabel(r"$\theta_d\ [{\rm arcmin}]$", fontsize=COMBINED_PROFILE_LABEL_FONTSIZE)
    ax.set_ylabel(r"$y_{\rm CAP}\ [10^{-6}\,{\rm arcmin}^2]$",
                  fontsize=COMBINED_PROFILE_LABEL_FONTSIZE)
    ticksize(ax, COMBINED_PROFILE_TICK_FONTSIZE)
    ax.grid(False)
    if show_legend:
        ax.legend(
            handles=[handle],
            loc="upper left",
            frameon=True,
            fancybox=True,
            edgecolor="0.8",
            fontsize=COMBINED_PROFILE_LEGEND_FONTSIZE,
        )


def plot_combined(samples, cache, models, out_path):
    g = gdplt.get_subplot_plotter(subplot_size=CORNER_GETDIST_SIZE)
    g.settings.num_plot_contours = 2
    g.make_figure(nplot=1)
    plt.close(g.fig)

    fig_w, fig_h, block_w, origins = combined_layout()
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    n = len(CORNER_PARAMS)

    for mass_bin in range(3):
        limits = posterior_limits(
            [((mass_bin, s), samples[(mass_bin, s)]) for s in SECTORS],
            cache, params=CORNER_PARAMS,
        )
        profile_axes = []
        for sector in SECTORS:
            x0, y0 = origins[(mass_bin, sector)]
            draw_corner(
                fig, fig_w, fig_h, g,
                samples[(mass_bin, sector)],
                (mass_bin, sector), cache,
                SECTOR_COLORS[sector],
                limits, x0, y0,
            )
            # Profile: up and to the right of this corner's empty cell.
            prof_ax = add_axes_inches(
                fig, fig_w, fig_h,
                x0 + (n - 1) * CELL + PROFILE_PAD_LEFT,
                y0 + (n - 1) * CELL + PROFILE_PAD_BOTTOM,
                PROFILE_W, PROFILE_H,
            )
            draw_profile(
                prof_ax, models, mass_bin, sector,
                show_legend=PROFILE_SHOW_LEGEND and mass_bin == 0,
            )
            profile_axes.append(prof_ax)

        if PROFILE_SHARE_Y:
            lo = min(ax.get_ylim()[0] for ax in profile_axes)
            hi = max(ax.get_ylim()[1] for ax in profile_axes)
            for ax in profile_axes:
                ax.set_ylim(lo, hi)

        # Mass-bin title centered over the whole column.
        x0, _ = origins[(mass_bin, "major")]
        fig.text(
            (x0 + block_w / 2) / fig_w,
            (fig_h - TOP_MARGIN / 2) / fig_h,
            MASS_BIN_LABELS[mass_bin],
            ha="center", va="center", fontsize=MASS_BIN_TITLE_FONTSIZE,
        )

    # Row labels on the far left, centered on each corner.
    for sector in SECTORS:
        _, y0 = origins[(0, sector)]
        fig.text(
            ROW_LABEL_X / fig_w,
            (y0 + n * CELL / 2) / fig_h,
            f"{sector.capitalize()} axis",
            rotation=90, ha="center", va="center",
            fontsize=ROW_LABEL_FONTSIZE,
        )

    fig.savefig(out_path, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES,
                facecolor="white", transparent=False)
    plt.close(fig)
    return out_path


def plot_posteriors(cache, out_path):
    fig, axes = plt.subplots(3, 3, figsize=POSTERIOR_FIGSIZE, sharey=False)
    for col, mass_bin in enumerate(range(3)):
        for row, (param, plabel) in enumerate(zip(PARAMS, PARAM_LABELS)):
            ax = axes[row, col]
            for sector in SECTORS:
                density = cache[(mass_bin, sector, param)][0]
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


def print_fit_statistics_table(all_stats):
    """Print best-fit statistics with dof = (number of aperture radii) - (number of fitted parameters)."""
    mass_bins = ("11.0-11.4", "11.4-11.7", "11.7-12.0")
    headers = ("Mass bin (log10 M*/Msun)", "Sector", "N_R", "N_par", "dof",
               "Chi^2", "Reduced chi^2", "Gaussian PTE")
    rows = []
    for mass_bin, label in enumerate(mass_bins):
        for sector in SECTORS:
            stats = all_stats[(mass_bin, sector)]
            rows.append((
                label,
                sector.capitalize(),
                str(stats["n_data"]),
                str(stats["n_params"]),
                str(stats["dof"]),
                f"{stats['chi2']:.3f}",
                f"{stats['reduced_chi2']:.3f}",
                f"{stats['gaussian_pte']:.4g}",
            ))
    widths = [max(len(str(row[col])) for row in [headers, *rows]) for col in range(len(headers))]
    def format_row(row):
        return "  ".join(str(value).ljust(widths[col]) for col, value in enumerate(row))
    print("\nBest-fit profile statistics (dof = N_R - N_par):")
    print(format_row(headers))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))
    print("Gaussian PTE = Pr[chi^2(dof) >= observed chi^2]\n")


def plot_residuals(models, out_path):
    """Plot data minus best-fit prediction for all mass bins and sectors."""
    fig, axes = plt.subplots(1, 3, figsize=RESIDUAL_FIGSIZE, sharey=False)
    for mass_bin, ax in enumerate(axes):
        for sector in SECTORS:
            likeli, theory_tot, _ = models[(mass_bin, sector)]
            R = np.asarray(likeli.R.to_value("arcmin"), dtype=float)
            data = np.asarray(likeli.meas_data.value, dtype=float)
            cov = np.asarray(likeli.meas_cov.value, dtype=float)
            err = np.sqrt(np.diag(cov))
            residual = data - theory_tot
            ax.errorbar(
                R + POINT_X_OFFSETS[sector],
                residual * 1e6,
                yerr=err * 1e6,
                fmt="o",
                ms=RESIDUAL_MARKER_SIZE,
                capsize=RESIDUAL_CAPSIZE,
                color=SECTOR_COLORS[sector],
                label=sector.capitalize(),
            )
        ax.axhline(0.0, color="0.35", lw=1.0, ls="--")
        ax.set_title(MASS_BIN_LABELS[mass_bin], **fontsize(RESIDUAL_TITLE_FONTSIZE))
        ax.set_xlabel(r"$\theta_d\ [{\rm arcmin}]$", **fontsize(RESIDUAL_LABEL_FONTSIZE))
        if mass_bin == 0:
            ax.set_ylabel(
                r"$(y_{\rm data}-y_{\rm model})\ [10^{-6}\,{\rm arcmin}^2]$",
                **fontsize(RESIDUAL_LABEL_FONTSIZE),
            )
        ticksize(ax, RESIDUAL_TICK_FONTSIZE)
        ax.grid(False)
    fig.legend(
        handles=[
            Line2D([], [], color=SECTOR_COLORS[s], marker="o", ls="none",
                   markersize=5.5, label=s.capitalize())
            for s in SECTORS
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, RESIDUAL_LEGEND_Y),
        ncol=2,
        frameon=False,
        fontsize=RESIDUAL_LEGEND_FONTSIZE,
    )
    fig.tight_layout(rect=(0.0, RESIDUAL_LEGEND_SPACE, 1.0, 1.0),
                     pad=RESIDUAL_PAD, w_pad=RESIDUAL_W_PAD)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)


def main():
    args = parse_args()
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
            samples[(mass_bin, sector)] = load_getdist(root, args.ignore_rows)
    cache = prepare_posteriors(samples)
    models = {
        key: load_model_at_best(roots[key], samples[key])
        for key in roots
    }
    fit_stats = {key: model[2] for key, model in models.items()}

    # Set the style only after the models are built: importing the
    # likelihood/theory code during get_model can change matplotlib's style.
    set_plot_style("default")
    force_light_style()

    posterior_path = out_dir / f"{args.job_prefix}_posteriors_3x3.pdf"
    residual_path = out_dir / f"{args.job_prefix}_residuals_1x3.pdf"
    combined_path = out_dir / f"{args.job_prefix}_corners_profiles.pdf"
    difference_path = out_dir / f"{args.job_prefix}_P0_minor_minus_major_1x3.pdf"

    plot_posteriors(cache, posterior_path)
    plot_residuals(models, residual_path)
    print_fit_statistics_table(fit_stats)
    plot_combined(samples, cache, models, combined_path)
    plot_differences(samples, difference_path, args.seed, args.difference_draws)

if __name__ == "__main__":
    main()
