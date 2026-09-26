#!/usr/bin/env python3
"""Make combined plots from all six independent CAPPIBARAS chains.

Produces, by default:
  1) <prefix>_posteriors_3x3.pdf
     Three mass-bin columns x three fitted-parameter rows, with major/minor
     1D posteriors overlaid in every mass-bin panel.

  2) <prefix>_profiles_1x3.pdf
     Three mass-bin panels with major/minor data plus the posterior-mean
     total, 1-halo, and 2-halo model components.

  3) <prefix>_corner_b{0,1,2}_{major,minor}.pdf
     Six individual three-parameter GetDist corner plots.

  4) <prefix>_corner_b{0,1,2}_major_minor.pdf
     Three overlaid GetDist corner plots comparing major and minor sectors
     within each stellar-mass bin.

The profile panels also report chi^2, chi^2/N, and PTE for the plotted
maximum-likelihood chain sample.  The PTE convention matches Chad's
plot_chains.compute_confidence_corner_multi: N = number of fitted data points
is used as the chi-square degrees of freedom.
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job-prefix", default="radio6_gnfw")
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


def make_corner_plot(
    roots,
    out_path,
    legend_labels=None,
    title=None,
    contour_colors=None,
):
    """Save one GetDist triangle plot for one or more chains."""
    roots = list(roots)
    g = gdplt.get_subplot_plotter(subplot_size=2.35)
    g.settings.title_limit_fontsize = 12
    g.triangle_plot(
        roots,
        PARAMS,
        filled=True,
        legend_labels=legend_labels,
        contour_colors=contour_colors,
        title_limit=1,
        title_fmt=".2f",
        param_limits=posterior_limits(roots),
    )
    if title:
        g.fig.suptitle(title, y=1.02)
    g.export(str(out_path))
    plt.close(g.fig)


def plot_corners(samples, out_dir, job_prefix):
    """Write six individual corners plus three major/minor overlay corners."""
    saved = []

    for mass_bin in range(3):
        for sector in SECTORS:
            out_path = out_dir / f"{job_prefix}_corner_b{mass_bin}_{sector}.pdf"
            make_corner_plot(
                [samples[(mass_bin, sector)]],
                out_path,
                legend_labels=[sector.capitalize()],
                title=f"{MASS_BIN_LABELS[mass_bin]}: {sector.capitalize()}",
                contour_colors=[SECTOR_COLORS[sector]],
            )
            saved.append(out_path)

        out_path = out_dir / f"{job_prefix}_corner_b{mass_bin}_major_minor.pdf"
        make_corner_plot(
            [samples[(mass_bin, "major")], samples[(mass_bin, "minor")]],
            out_path,
            legend_labels=["Major", "Minor"],
            title=f"{MASS_BIN_LABELS[mass_bin]}: Major vs Minor",
            contour_colors=[CAP_MAJOR_COLOR, CAP_MINOR_COLOR],
        )
        saved.append(out_path)

    return saved

def plot_posteriors(samples, out_path):
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharey=False)

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
                ax.set_title(MASS_BIN_LABELS[mass_bin])
            if col == 0:
                ax.set_ylabel(f"{plabel}\nNormalized density")
            ax.set_xlabel(plabel)
            ax.set_ylim(bottom=0)
            if row == 0 and col == 0:
                ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
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

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), sharey=True)
    
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
                ms=4.5,
                capsize=2.5,
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
                rf"{sector.capitalize()}: $\chi^2={stats['chi2']:.2f}$, "
                rf"$\chi^2/N={stats['chi2_over_N']:.2f}$, "
                rf"PTE$={stats['pte']:.3f}$"
            )

        ax.axhline(0.0, lw=0.8, ls="--", color="0.5")
        ax.set_title(MASS_BIN_LABELS[mass_bin])
        ax.set_xlabel(r"$\theta_d\ [{\rm arcmin}]$")
        if mass_bin == 0:
            ax.set_ylabel(r"$y_{\rm CAP}\ [10^{-6}\,{\rm arcmin}^2]$")

        ax.text(
            0.03,
            0.97,
            "\n".join(stat_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
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
        bbox_to_anchor=(0.5, -0.015),
        ncol=5,
        frameon=False,
        fontsize=9.5,
        columnspacing=1.8,
        handlelength=2.4,
    )

    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(out_path, bbox_inches="tight")
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
    corner_paths = plot_corners(samples, out_dir, args.job_prefix)

    print(f"Saved {posterior_path}")
    print(f"Saved {profile_path}")
    for path in corner_paths:
        print(f"Saved {path}")
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
