from pathlib import Path
import numpy as np
import astropy.units as u
from Models.OldModules.SHMRs import Kravtsov2018
from Models.Papers.Coulton2024 import MapData as ACTDR6MapData
from Models.Papers.RiedGuachalla2025 import TargetData as CatalogDistributions


class Measurement:
    def __init__(self, profile_file, mass_bin, sector, **kwargs):
        with np.load(profile_file, allow_pickle=False) as d:
            prefix = f"bin{int(mass_bin)}_{sector.lower()}"
            self.R = d["theta_arcmin"] * u.arcmin
            self.y_data = d[f"{prefix}_mean"] * u.arcmin**2
            cov = d[f"{prefix}_covariance"]
            self.y_cov = cov * u.arcmin**4
            self.y_err = np.sqrt(np.diag(cov)) * u.arcmin**2


class TargetData(CatalogDistributions):
    # Do not inherit Ried Guachalla's sample-specific metadata.
    area = None
    v_rms = None

    def __init__(self, profile_file, mass_bin, **kwargs):
        with np.load(profile_file, allow_pickle=False) as d:
            prefix = f"bin{int(mass_bin)}"
            self.cat_zs = np.asarray(d[f"{prefix}_redshift"], dtype=float)
            self.cat_logMs = np.asarray(d[f"{prefix}_log10_stellar_mass"], dtype=float)
            self.mass_lo = float(d[f"{prefix}_mass_lo"].item())
            self.mass_hi = float(d[f"{prefix}_mass_hi"].item())

        shmr = Kravtsov2018(model="B13", mdef="200c", scatter="S")
        self.cat_logMh = np.asarray(shmr.HSMR(self.cat_logMs)(), dtype=float)

        for name, catalog in (("z", self.cat_zs),
                              ("logMs", self.cat_logMs),
                              ("logMh", self.cat_logMh)):
            setattr(self, f"{name}Min", float(np.min(catalog)))
            setattr(self, f"{name}Max", float(np.max(catalog)))
            setattr(self, f"{name}Mean", float(np.mean(catalog)))


class MapData(ACTDR6MapData):
    def __init__(self, beam_file, **kwargs):
        self.path = Path(beam_file).expanduser().parent
        super().__init__(**kwargs)


class CAP:
    def __init__(self, Rs, inner_radius_arcmin=1.0, N_RCAP=1000, **kwargs):
        self.Rs = u.Quantity(Rs).to(u.arcmin)
        self.R0 = float(inner_radius_arcmin) * u.arcmin
        self.N_RCAP = int(N_RCAP)
        self.Rout = np.sqrt(2.0 * self.Rs**2 - self.R0**2)
        self.Rs_disk = np.array([
            np.linspace(self.R0.value, R.value, self.N_RCAP + 1)
            for R in self.Rs
        ]) * u.arcmin
        self.Rs_ring = np.array([
            np.linspace(R.value, Rout.value, self.N_RCAP + 1)
            for R, Rout in zip(self.Rs, self.Rout)
        ]) * u.arcmin

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
            out[i] = self._radial_integral(rin, yin) - self._radial_integral(rout, yout)

        return out * prof_unit * u.arcmin**2
