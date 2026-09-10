"""
data_curves.py
==============
Yield-curve and inflation data layer (Stage 1 data + Stage 4 discounting).

Design intent
-------------
The base-date term structure is stored as a smooth Nelson-Siegel curve whose
parameters are calibrated to the *shape* of the Bank of England GLC nominal and
real spot curves at 31 March 2026. This keeps the repo self-contained and
reproducible while remaining swappable: drop a real BoE GLC file into
data/raw/ and `load_boe_spot_curve()` will use it instead.

All rates are annually-compounded spot rates expressed as decimals.

Curves provided
---------------
    nominal_spot(t)   nominal gilt spot rate for term t (years)
    real_spot(t)      real (index-linked) gilt spot rate
    implied_rpi(t)    breakeven RPI = nominal - real
    cpi(t)            implied RPI less the RPI/CPI wedge

Discount-basis curves (Stage 4) are built in valuation.py from these.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATA_RAW, RPI_CPI_WEDGE, VALUATION_DATE


# --------------------------------------------------------------------------
# Nelson-Siegel spot-rate function
# --------------------------------------------------------------------------
def _nelson_siegel(t: np.ndarray, beta0: float, beta1: float,
                   beta2: float, tau: float) -> np.ndarray:
    """Nelson-Siegel spot rate. t may be scalar or array (years, > 0)."""
    t = np.asarray(t, dtype=float)
    t = np.where(t <= 0, 1e-6, t)          # guard t=0
    x = t / tau
    term = (1.0 - np.exp(-x)) / x
    return beta0 + beta1 * term + beta2 * (term - np.exp(-x))


# Parameters calibrated to the shape of the BoE GLC curves at 31 Mar 2026.
# beta0 = long-run level, beta1 = short-end deviation (slope), beta2 = hump,
# tau   = location of the hump. Documented so a reviewer can see the shape.
_NS_NOMINAL = dict(beta0=0.0475, beta1=-0.0030, beta2=0.0090, tau=4.5)
#   -> ~4.5% short, gentle hump ~4.75%, long level ~4.75%
_NS_REAL = dict(beta0=0.0135, beta1=-0.0035, beta2=0.0040, tau=4.5)
#   -> ~1.0% short rising to ~1.35% long (index-linked gilts)


class YieldCurves:
    """Holds the base-date nominal and real spot curves and derived series."""

    def __init__(self, nominal_params=None, real_params=None,
                 wedge: float = RPI_CPI_WEDGE, source: str = "synthetic-NS"):
        self.nominal_params = nominal_params or dict(_NS_NOMINAL)
        self.real_params = real_params or dict(_NS_REAL)
        self.wedge = wedge
        self.source = source
        self.valuation_date = VALUATION_DATE

        # Optional real-data override tables (term -> rate), if loaded.
        self._nominal_table: pd.Series | None = None
        self._real_table: pd.Series | None = None

    # -- spot curves -------------------------------------------------------
    def nominal_spot(self, t):
        if self._nominal_table is not None:
            return self._interp(self._nominal_table, t)
        return _nelson_siegel(t, **self.nominal_params)

    def real_spot(self, t):
        if self._real_table is not None:
            return self._interp(self._real_table, t)
        return _nelson_siegel(t, **self.real_params)

    def implied_rpi(self, t):
        """Breakeven RPI inflation = nominal - real (Fisher, additive approx)."""
        return self.nominal_spot(t) - self.real_spot(t)

    def cpi(self, t):
        """Implied CPI = implied RPI less the RPI/CPI wedge, floored at 0."""
        return np.maximum(self.implied_rpi(t) - self.wedge, 0.0)

    # -- discount factors --------------------------------------------------
    def discount_factor(self, t, spread=0.0):
        """v(t) using nominal spot + a flat spread (annually compounded)."""
        r = self.nominal_spot(t) + spread
        t = np.asarray(t, dtype=float)
        return (1.0 + r) ** (-t)

    # -- real-data override ------------------------------------------------
    @staticmethod
    def _interp(table: pd.Series, t):
        t = np.asarray(t, dtype=float)
        return np.interp(t, table.index.values, table.values)

    def load_override(self, nominal: pd.Series | None = None,
                      real: pd.Series | None = None, source: str | None = None):
        if nominal is not None:
            self._nominal_table = nominal.sort_index()
        if real is not None:
            self._real_table = real.sort_index()
        if source:
            self.source = source
        return self


# --------------------------------------------------------------------------
# Real BoE GLC loader (used only if a file is present)
# --------------------------------------------------------------------------
def load_boe_spot_curve(curves: YieldCurves | None = None) -> YieldCurves:
    """
    If data/raw/boe_nominal_spot.csv and boe_real_spot.csv exist (columns:
    term_years, spot_rate with rates as decimals or percent), load and use them.
    Otherwise return the synthetic Nelson-Siegel curves unchanged.
    """
    curves = curves or YieldCurves()
    nom_path = DATA_RAW / "boe_nominal_spot.csv"
    real_path = DATA_RAW / "boe_real_spot.csv"

    def _read(path):
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        term = df["term_years"].astype(float)
        rate = df["spot_rate"].astype(float)
        if rate.max() > 1.0:                     # percent -> decimal
            rate = rate / 100.0
        return pd.Series(rate.values, index=term.values).sort_index()

    loaded = {}
    if nom_path.exists():
        loaded["nominal"] = _read(nom_path)
    if real_path.exists():
        loaded["real"] = _read(real_path)
    if loaded:
        curves.load_override(nominal=loaded.get("nominal"),
                             real=loaded.get("real"),
                             source="BoE GLC 31-Mar-2026 (loaded)")
    return curves


# Module-level singleton used across the model.
CURVES = load_boe_spot_curve(YieldCurves())


if __name__ == "__main__":
    terms = np.array([1, 2, 5, 10, 15, 20, 30, 50])
    print(f"Base date: {CURVES.valuation_date}  source: {CURVES.source}\n")
    df = pd.DataFrame({
        "term": terms,
        "nominal_%": CURVES.nominal_spot(terms) * 100,
        "real_%": CURVES.real_spot(terms) * 100,
        "impliedRPI_%": CURVES.implied_rpi(terms) * 100,
        "CPI_%": CURVES.cpi(terms) * 100,
    })
    print(df.round(3).to_string(index=False))
