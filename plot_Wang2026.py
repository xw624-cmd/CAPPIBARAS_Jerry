import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import yaml
import cobaya
import getdist.plots as gdplt
from matplotlib.lines import Line2D
from scipy.special import erfinv
from scipy.stats import gaussian_kde, chi2 as chi2_distribution
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
SAVE_PAD_INCHES = 0.02
HPD_MASS = 0.68
CONTOUR_LEVELS = np.array([0.68, 0.95])
UPPER_LIMIT_MASS = 0.95
CORNER_CONTOUR_PAD = 0.09
DEFAULT_DIFFERENCE_DRAWS = 10000
DEFAULT_RANDOM_SEED = 2026
DIFFERENCE_FIGSIZE = (15, 4.6)
DIFFERENCE_XLIMITS = [(-60, 60), (-5, 7.5), (-10, 20)]
CORNER_FIGSIZE = (18.0, 12.0)
CORNER_BLOCK_WSPACE = 0.02
CORNER_BLOCK_HSPACE = 0.03
CORNER_PANEL_WSPACE = 0.0
CORNER_PANEL_HSPACE = 0.0
CORNER_MARGINS_TOP_ROW = dict(left=0.085, right=0.995, bottom=0.07, top=0.90)
CORNER_MARGINS_BOTTOM_ROW = dict(left=0.085, right=0.995, bottom=0.07, top=0.925)
CORNER_GETDIST_SIZE = 1.7
CORNER_BLOCK_TITLE_FONTSIZE = 13
CORNER_VALUE_FONTSIZE = 11
CORNER_LABEL_FONTSIZE = 11
CORNER_TICK_FONTSIZE = 9
CORNER_BLOCK_TITLE_Y = 1.0
POSTERIOR_FIGSIZE = (12, 9)
POSTERIOR_PAD = 0.2
POSTERIOR_W_PAD = 0.4
POSTERIOR_H_PAD = 0.4
POSTERIOR_TITLE_FONTSIZE = None
POSTERIOR_LABEL_FONTSIZE = None
POSTERIOR_TICK_FONTSIZE = None
POSTERIOR_LEGEND_FONTSIZE = None
PROFILE_FIGSIZE = (16.5, 5.2)
PROFILE_PAD = 0.2
PROFILE_W_PAD = 0
PROFILE_LEGEND_SPACE = 0.08
PROFILE_LEGEND_Y = -0.015
PROFILE_LEGEND_FONTSIZE = 10
PROFILE_TITLE_FONTSIZE = 11
PROFILE_LABEL_FONTSIZE = 10
PROFILE_TICK_FONTSIZE = None
PROFILE_MARKER_SIZE = 4.5
PROFILE_CAPSIZE = 2.5


def fontsize(size):
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
    for name, spec in info["params"].items():
        if isinstance(spec, dict) and "prior" in spec:
            params[name] = float(gd.samples[best_row, col[name]])
        elif isinstance(spec, dict) and "value" in spec:
            params[name] = spec["value"]
        else:
            params[name] = spec
    return info, params


def load_model_at_best(root, gd):
    info, params = best_chain_params(root, gd)
    model_info = {
        key: info[key]
        for key in ("likelihood", "params", "theory")
        if key in info
    }
    model = cobaya.get_model(model_info)
    likelihoods = model.likelihood
    lname = next(iter(likelihoods))
    likeli = likelihoods[lname]
    result = get_param_results(params, likelihoods)[lname]
    theory_1h = np.asarray(result["1h"], dtype=float)
    theory_2h = np.asarray(result["2h"], dtype=float)
    theory_tot = np.asarray(likeli.model(params).value, dtype=float)
    data = np.asarray(likeli.data.y, dtype=float)
    inv_cov = np.asarray(likeli.data.inv_cov, dtype=float)
    delta = data - theory_tot
    chi2 = float(delta @ inv_cov @ delta)
    stats = {
        "chi2": chi2,
        "reduced_chi2": chi2 / 4.0,
        "gaussian_pte": float(chi2_distribution.sf(chi2, df=4)),
        "params": {name: params[name] for name in PARAMS},
    }
    return likeli, theory_1h, theory_2h, theory_tot, stats


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
    if latex:
        if len(intervals) != 1:
            return rf"= {mode:.2f}\ (\mathrm{{68\\\\%\ HPD:multi}})"
        low, high = intervals[0]
        return rf"= {mode:.2f}^{{+{high - mode:.2f}}}_{{-{mode - low:.2f}}}"
    regions = " U ".join(f"[{lo:.5g}, {hi:.5g}]" for lo, hi in intervals)
    return f"mode={mode:.5g}; HPD68={regions}"


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
    # p_positive = float(np.mean(delta > 0))
    limited = p >= 1.0
    # p_for_inverse = min(p, 1.0 - 0.5 / draws)
    #sigma_abs = float(np.sqrt(2.0) * erfinv(p_for_inverse))
    sigma_abs = float(np.sqrt(2.0) * erfinv(p))
    #direction = 1 if p_positive > 0.5 else -1 if p_positive < 0.5 else 0
    sigma_signed =  sigma_abs #* direction 
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
        "prob_minor_gt_major": p, #p_positive,
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


def draw_one_corner(g, subfig, gd, key, cache, color, limits, panel_title, margins):
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
                summary = cache[(*key, xparam)][1]
                if summary["kind"] in ("upper", "lower"):
                    edge = gd.ranges.getLower(xparam) if summary["kind"] == "upper" \
                        else gd.ranges.getUpper(xparam)
                    if edge is not None:
                        ax.axvspan(edge, summary["limit"], color=color, alpha=0.12, lw=0)
                    ax.axvline(summary["limit"], color=color, lw=1.1, ls=":")
                else:
                    for low, high in summary["intervals"]:
                        ax.axvspan(low, high, color=color, alpha=0.12, lw=0)
                if summary["kind"] == "twotail":
                    ax.axvline(summary["mode"], color=color, lw=1.0, ls="--")
                ax.set_ylim(0.0, 1.08)
                ax.set_yticks([])
                label = PARAM_LABELS[col].strip("$")
                ax.set_title(
                    rf"${label} {summarize(summary, latex=True)}$",
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


def plot_corners(samples, cache, out_dir, job_prefix):
    out_path = out_dir / f"{job_prefix}_corners_2x3.pdf"
    g = gdplt.get_subplot_plotter(subplot_size=CORNER_GETDIST_SIZE)
    g.settings.num_plot_contours = 2
    g.make_figure(nplot=1)
    plt.close(g.fig)
    fig = plt.figure(figsize=CORNER_FIGSIZE)
    row_margins = [CORNER_MARGINS_TOP_ROW, CORNER_MARGINS_BOTTOM_ROW]
    height_ratios = [1.0 / (m["top"] - m["bottom"]) for m in row_margins]
    subfigs = fig.subfigures(
        2, 3, wspace=CORNER_BLOCK_WSPACE, hspace=CORNER_BLOCK_HSPACE,
        height_ratios=height_ratios,
    )
    for mass_bin in range(3):
        limits = posterior_limits(
            [((mass_bin, s), samples[(mass_bin, s)]) for s in SECTORS], cache
        )
        for row, sector in enumerate(SECTORS):
            if row == 0:
                title = f"{MASS_BIN_LABELS[mass_bin]}\n{sector.capitalize()} axis"
            else:
                title = f"{sector.capitalize()} axis"
            draw_one_corner(
                g,
                subfigs[row, mass_bin],
                samples[(mass_bin, sector)],
                (mass_bin, sector), cache,
                SECTOR_COLORS[sector],
                limits,
                title,
                row_margins[row],
            )
    fig.savefig(out_path, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
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


def plot_profiles(samples, roots, out_path, residual_path=None):
    models = {
        key: load_model_at_best(roots[key], samples[key])
        for key in roots
    }
    set_plot_style("default")
    fig, axes = plt.subplots(1, 3, figsize=PROFILE_FIGSIZE, sharey=True)
    all_stats = {}
    for mass_bin, ax in enumerate(axes):
        for sector in SECTORS:
            likeli, theory_1h, theory_2h, theory_tot, stats = models[(mass_bin, sector)]
            all_stats[(mass_bin, sector)] = stats
            R = np.asarray(likeli.R.to_value("arcmin"), dtype=float)
            data = np.asarray(likeli.meas_data.value, dtype=float)
            cov = np.asarray(likeli.meas_cov.value, dtype=float)
            err = np.sqrt(np.diag(cov))
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
        ax.set_title(MASS_BIN_LABELS[mass_bin], **fontsize(PROFILE_TITLE_FONTSIZE))
        ax.set_xlabel(r"$\theta_d\ [{\rm arcmin}]$", **fontsize(PROFILE_LABEL_FONTSIZE))
        if mass_bin == 0:
            ax.set_ylabel(r"$y_{\rm CAP}\ [10^{-6}\,{\rm arcmin}^2]$",
                          **fontsize(PROFILE_LABEL_FONTSIZE))
        ticksize(ax, PROFILE_TICK_FONTSIZE)
        ax.grid(False)
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
    if residual_path is not None:
        plot_residuals(models, residual_path)
    return all_stats


def print_fit_statistics_table(all_stats):
    """Print best-fit statistics assuming four chi-square degrees of freedom."""
    mass_bins = ("11.0-11.4", "11.4-11.7", "11.7-12.0")
    headers = ("Mass bin (log10 M*/Msun)", "Sector", "Chi^2", "Reduced chi^2", "Gaussian PTE")
    rows = []
    for mass_bin, label in enumerate(mass_bins):
        for sector in SECTORS:
            stats = all_stats[(mass_bin, sector)]
            rows.append((
                label,
                sector.capitalize(),
                f"{stats['chi2']:.3f}",
                f"{stats['reduced_chi2']:.3f}",
                f"{stats['gaussian_pte']:.4g}",
            ))
    widths = [max(len(str(row[col])) for row in [headers, *rows]) for col in range(len(headers))]
    def format_row(row):
        return "  ".join(str(value).ljust(widths[col]) for col, value in enumerate(row))
    print("\nBest-fit profile statistics (4 degrees of freedom):")
    print(format_row(headers))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))
    print("Gaussian PTE = Pr[chi^2(df=4) >= observed chi^2]\n")




def plot_residuals(models, out_path):
    """Plot data minus best-fit prediction for all mass bins and sectors."""
    fig, axes = plt.subplots(1, 3, figsize=PROFILE_FIGSIZE, sharey=False)
    for mass_bin, ax in enumerate(axes):
        for sector in SECTORS:
            likeli, _, _, theory_tot, _ = models[(mass_bin, sector)]
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
                ms=PROFILE_MARKER_SIZE,
                capsize=PROFILE_CAPSIZE,
                color=SECTOR_COLORS[sector],
                label=sector.capitalize(),
            )
        ax.axhline(0.0, color="0.35", lw=1.0, ls="--")
        ax.set_title(MASS_BIN_LABELS[mass_bin], **fontsize(PROFILE_TITLE_FONTSIZE))
        ax.set_xlabel(r"$\theta_d\ [{\rm arcmin}]$", **fontsize(PROFILE_LABEL_FONTSIZE))
        if mass_bin == 0:
            ax.set_ylabel(
                r"$(y_{\rm data}-y_{\rm model})\ [10^{-6}\,{\rm arcmin}^2]$",
                **fontsize(PROFILE_LABEL_FONTSIZE),
            )
        ticksize(ax, PROFILE_TICK_FONTSIZE)
        ax.grid(False)
    fig.legend(
        handles=[
            Line2D([], [], color=SECTOR_COLORS[s], marker="o", ls="none",
                   markersize=5.5, label=s.capitalize())
            for s in SECTORS
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, PROFILE_LEGEND_Y),
        ncol=2,
        frameon=False,
        fontsize=PROFILE_LEGEND_FONTSIZE,
    )
    fig.tight_layout(rect=(0.0, PROFILE_LEGEND_SPACE, 1.0, 1.0),
                     pad=PROFILE_PAD, w_pad=PROFILE_W_PAD)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)


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
            samples[(mass_bin, sector)] = load_getdist(root, args.ignore_rows)
    cache = prepare_posteriors(samples)
    posterior_path = out_dir / f"{args.job_prefix}_posteriors_3x3.pdf"
    profile_path = out_dir / f"{args.job_prefix}_profiles_1x3.pdf"
    residual_path = out_dir / f"{args.job_prefix}_residuals_1x3.pdf"
    plot_posteriors(cache, posterior_path)
    fit_stats = plot_profiles(samples, roots, profile_path, residual_path=residual_path)
    print_fit_statistics_table(fit_stats)
    plot_corners(samples, cache, out_dir, args.job_prefix)
    difference_path = out_dir / f"{args.job_prefix}_P0_minor_minus_major_1x3.pdf"
    plot_differences(
        samples, difference_path, args.seed, args.difference_draws,
    )


if __name__ == "__main__":
    main()
