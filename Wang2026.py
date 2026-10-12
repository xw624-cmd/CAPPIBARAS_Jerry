from pathlib import Path

import numpy as np
import astropy.units as u

from Models.OldModules.SHMRs import Kravtsov2018
from Models.Papers.Coulton2024 import MapData as ACTDR6MapData
from Models.Papers.RiedGuachalla2025 import TargetData as CatalogDistributions
from Models.Papers.Popik2026 import CAP as PopikCAP


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


class CAP(PopikCAP):
    def __init__(self, Rs, inner_radius_arcmin=1.0, N_RCAP=10000,
                 f_ring=np.sqrt(2), **kwargs):
        Rs = u.Quantity(Rs, u.arcmin).to_value(u.arcmin)
        R0 = float(inner_radius_arcmin)
        super().__init__(Rs=Rs, f_ring=f_ring, R_0=R0, N_RCAP=N_RCAP, **kwargs)

        Rout = np.sqrt((f_ring * Rs)**2 - R0**2)
        self.intfac_disk = 2 * np.pi * (Rs - R0)[:, None] / N_RCAP * self.Rs_disk
        self.intfac_ring = 2 * np.pi * (Rout - R0)[:, None] / N_RCAP * self.Rs_ring
