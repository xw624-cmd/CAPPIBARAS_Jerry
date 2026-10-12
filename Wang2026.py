"""
CAPPIBARAS data/observable adapters for oriented tSZ profiles.
"""

from pathlib import Path

import numpy as np
import astropy.units as u

from Models.OldModules.SHMRs import Kravtsov2018 as ChadKravtsov2018


class Measurement:
    """One major- or minor-axis ring-ring CAP profile from the stacking NPZ.
    """

    def __init__(self, profile_file, mass_bin, sector, **kwargs):
        self.profile_file = Path(profile_file).expanduser()
        self.mass_bin = int(mass_bin)
        self.sector = str(sector).lower()

        if self.mass_bin not in (0, 1, 2):
            raise ValueError(f"mass_bin must be 0, 1, or 2; got {mass_bin!r}")
        if self.sector not in ("major", "minor"):
            raise ValueError(f"sector must be 'major' or 'minor'; got {sector!r}")
        if not self.profile_file.exists():
            raise FileNotFoundError(self.profile_file)

        self._load()

    def _load(self):
        with np.load(self.profile_file, allow_pickle=False) as d:
            prefix = f"bin{self.mass_bin}"
            required = [
                "theta_arcmin",
                f"{prefix}_{self.sector}_mean",
                f"{prefix}_{self.sector}_covariance",
            ]
            missing = [k for k in required if k not in d]
            if missing:
                raise KeyError(f"Missing keys in {self.profile_file}: {missing}")

            R = np.asarray(d["theta_arcmin"], dtype=float)
            y = np.asarray(d[f"{prefix}_{self.sector}_mean"], dtype=float)
            cov = np.asarray(
                d[f"{prefix}_{self.sector}_covariance"], dtype=float
            )

            if y.shape != R.shape:
                raise ValueError(f"R/y shape mismatch: {R.shape} vs {y.shape}")
            if cov.shape != (R.size, R.size):
                raise ValueError(
                    f"Covariance has shape {cov.shape}; expected {(R.size, R.size)}"
                )

            self.mass_lo = float(np.asarray(d[f"{prefix}_mass_lo"]).item())
            self.mass_hi = float(np.asarray(d[f"{prefix}_mass_hi"]).item())

        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(cov)):
            raise ValueError("Non-finite values found in measurement or covariance")
        if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-14):
            raise ValueError("Measurement covariance is not symmetric")

        self.R = R * u.arcmin
        self.y_data = y * u.arcmin**2
        self.y_cov = cov * u.arcmin**4
        self.y_err = np.sqrt(np.clip(np.diag(cov), 0.0, None)) * u.arcmin**2


class TargetData:
    """Galaxy sample underlying one paired major/minor mass-bin profile.
    """

    def __init__(self, profile_file, mass_bin, **kwargs):
        self.profile_file = Path(profile_file).expanduser()
        self.mass_bin = int(mass_bin)

        if self.mass_bin not in (0, 1, 2):
            raise ValueError(f"mass_bin must be 0, 1, or 2; got {mass_bin!r}")
        if not self.profile_file.exists():
            raise FileNotFoundError(self.profile_file)

        prefix = f"bin{self.mass_bin}"
        with np.load(self.profile_file, allow_pickle=False) as d:
            self.cat_zs = np.asarray(d[f"{prefix}_redshift"], dtype=float)
            self.cat_logMs = np.asarray(
                d[f"{prefix}_log10_stellar_mass"], dtype=float
            )
            self.cat_weights = np.asarray(
                d.get(f"{prefix}_stack_weight", np.ones_like(self.cat_zs)),
                dtype=float,
            )
            self.mass_lo = float(np.asarray(d[f"{prefix}_mass_lo"]).item())
            self.mass_hi = float(np.asarray(d[f"{prefix}_mass_hi"]).item())

        if not (
            self.cat_zs.shape == self.cat_logMs.shape == self.cat_weights.shape
        ):
            raise ValueError("z, stellar-mass, and stack-weight arrays are misaligned")

        good = (
            np.isfinite(self.cat_zs)
            & np.isfinite(self.cat_logMs)
            & np.isfinite(self.cat_weights)
            & (self.cat_weights > 0)
        )
        self.cat_zs = self.cat_zs[good]
        self.cat_logMs = self.cat_logMs[good]
        self.cat_weights = self.cat_weights[good]
        if self.cat_zs.size < 2:
            raise ValueError("Too few valid galaxies in TargetData")

        shmr = ChadKravtsov2018(model="B13", mdef="200c", scatter="S")
        self.cat_logMh = np.asarray(shmr.HSMR(self.cat_logMs)(), dtype=float)

        good = np.isfinite(self.cat_logMh)
        if not np.all(good):
            self.cat_logMh = self.cat_logMh[good]
            self.cat_zs = self.cat_zs[good]
            self.cat_logMs = self.cat_logMs[good]
            self.cat_weights = self.cat_weights[good]
        if self.cat_logMh.size < 2:
            raise ValueError("Too few valid Kravtsov-inferred halo masses")

        self.zMin = float(np.min(self.cat_zs))
        self.zMax = float(np.max(self.cat_zs))
        self.zMean = float(np.average(self.cat_zs, weights=self.cat_weights))
        self.logMsMin = float(np.min(self.cat_logMs))
        self.logMsMax = float(np.max(self.cat_logMs))
        self.logMsMean = float(
            np.average(self.cat_logMs, weights=self.cat_weights)
        )
        self.logMhMin = float(np.min(self.cat_logMh))
        self.logMhMax = float(np.max(self.cat_logMh))
        self.logMhMean = float(
            np.average(self.cat_logMh, weights=self.cat_weights)
        )

    @staticmethod
    def _edges(qmin, qmax, qnum=None, dq=None):
        if qnum is None:
            if dq is None:
                raise ValueError("Specify qnum or dq")
            qnum = int(np.ceil((qmax - qmin) / dq)) + 1
        qnum = int(qnum)
        if qnum < 2:
            raise ValueError("Distribution grid needs at least two edges")
        return np.linspace(qmin, qmax, qnum)

    def make_zbins(self, zMin=None, zMax=None, zNum=None, dz=None, **kwargs):
        zMin = self.zMin if zMin is None else float(zMin)
        zMax = self.zMax if zMax is None else float(zMax)
        edges = self._edges(zMin, zMax, zNum, dz)
        z = 0.5 * (edges[1:] + edges[:-1])
        dz_eff = z[1] - z[0] if z.size > 1 else edges[1] - edges[0]
        return edges, z, dz_eff

    def dNdz(self, zMin=None, zMax=None, zNum=None, dz=None, **kwargs):
        edges, z, dz_eff = self.make_zbins(zMin, zMax, zNum, dz)
        counts = np.histogram(self.cat_zs, bins=edges, weights=self.cat_weights)[0]
        return z, counts / dz_eff

    def make_logMsbins(
        self, logMsMin=None, logMsMax=None, logMsNum=None, dlogMs=None, **kwargs
    ):
        logMsMin = self.logMsMin if logMsMin is None else float(logMsMin)
        logMsMax = self.logMsMax if logMsMax is None else float(logMsMax)
        edges = self._edges(logMsMin, logMsMax, logMsNum, dlogMs)
        q = 0.5 * (edges[1:] + edges[:-1])
        dq = q[1] - q[0] if q.size > 1 else edges[1] - edges[0]
        return edges, q, dq

    def dNdlogMs(
        self, logMsMin=None, logMsMax=None, logMsNum=None, dlogMs=None, **kwargs
    ):
        edges, q, dq = self.make_logMsbins(
            logMsMin, logMsMax, logMsNum, dlogMs
        )
        counts = np.histogram(
            self.cat_logMs, bins=edges, weights=self.cat_weights
        )[0]
        return q, counts / dq

    def make_logMhbins(
        self, logMhMin=None, logMhMax=None, logMhNum=None, dlogMh=None, **kwargs
    ):
        logMhMin = self.logMhMin if logMhMin is None else float(logMhMin)
        logMhMax = self.logMhMax if logMhMax is None else float(logMhMax)
        edges = self._edges(logMhMin, logMhMax, logMhNum, dlogMh)
        q = 0.5 * (edges[1:] + edges[:-1])
        dq = q[1] - q[0] if q.size > 1 else edges[1] - edges[0]
        return edges, q, dq

    def dNdlogMh(
        self, logMhMin=None, logMhMax=None, logMhNum=None, dlogMh=None, **kwargs
    ):
        edges, q, dq = self.make_logMhbins(
            logMhMin, logMhMax, logMhNum, dlogMh
        )
        counts = np.histogram(
            self.cat_logMh, bins=edges, weights=self.cat_weights
        )[0]
        return q, counts / dq


class MapData:
    """Read the two-column ACT ILC beam transfer-function file."""

    def __init__(self, beam_file, **kwargs):
        self.beam_file = Path(beam_file).expanduser()
        if not self.beam_file.exists():
            raise FileNotFoundError(self.beam_file)

        arr = np.loadtxt(self.beam_file, comments="#")
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] < 2:
            raise ValueError(
                f"Expected at least two numeric columns in {self.beam_file}"
            )

        self.beam_ells = arr[:, 0]
        self.beam_data = arr[:, 1]
        order = np.argsort(self.beam_ells)
        self.beam_ells = self.beam_ells[order]
        self.beam_data = self.beam_data[order]

        good = np.isfinite(self.beam_ells) & np.isfinite(self.beam_data)
        self.beam_ells = self.beam_ells[good]
        self.beam_data = self.beam_data[good]
        if self.beam_ells.size < 2:
            raise ValueError("Beam file does not contain enough valid samples")


class CAP:
    """Continuous ring-ring version of the stacking aperture filter.

    For each aperture theta_d and fixed inner radius theta_0, integrate

        theta_0 < theta < theta_d

    and subtract the equal-area outer ring

        theta_d < theta < sqrt(2 theta_d^2 - theta_0^2).

    Popik2026.CAP itself remains untouched.
    """

    def __init__(
        self,
        Rs,
        inner_radius_arcmin=1.0,
        N_RCAP=1000,
        **kwargs,
    ):
        self.Rs = u.Quantity(Rs).to(u.arcmin)
        self.R0 = float(inner_radius_arcmin) * u.arcmin
        self.N_RCAP = int(N_RCAP)
        if self.N_RCAP < 20:
            raise ValueError("N_RCAP is too small for stable ring-ring integration")
        if np.any(self.Rs <= self.R0):
            raise ValueError("Every aperture radius must exceed the inner radius")

        self.Rout = np.sqrt(2.0 * self.Rs**2 - self.R0**2)

        self.Rs_disk = np.array(
            [
                np.linspace(self.R0.value, R.value, self.N_RCAP + 1)
                for R in self.Rs
            ]
        ) * u.arcmin
        self.Rs_ring = np.array(
            [
                np.linspace(R.value, Rout.value, self.N_RCAP + 1)
                for R, Rout in zip(self.Rs, self.Rout)
            ]
        ) * u.arcmin

    @staticmethod
    def _radial_integral(r_arcmin, y):
        return np.trapezoid(2.0 * np.pi * r_arcmin * y, x=r_arcmin)

    def disk_ring_CAP(self, Rsprof, prof2D_beam):
        Rprof = u.Quantity(Rsprof).to_value(u.arcmin)
        prof_unit = getattr(prof2D_beam, "unit", u.dimensionless_unscaled)
        yprof = np.asarray(
            prof2D_beam.value if hasattr(prof2D_beam, "value") else prof2D_beam,
            dtype=float,
        )

        out = np.empty(len(self.Rs), dtype=float)
        for i in range(len(self.Rs)):
            rin = self.Rs_disk[i].to_value(u.arcmin)
            rout = self.Rs_ring[i].to_value(u.arcmin)
            yin = np.interp(rin, Rprof, yprof, right=0.0)
            yout = np.interp(rout, Rprof, yprof, right=0.0)
            inner = self._radial_integral(rin, yin)
            outer = self._radial_integral(rout, yout)
            out[i] = inner - outer

        return out * prof_unit * u.arcmin**2
