"""
valuation.py  (Stage 4)
=======================
Discount the projected cashflows on three bases and report value + duration.

Basis            Discount rate                              Purpose
---------------  -----------------------------------------  ----------------------------
Technical prov.  gilts + 1.0% stepping to +0.5% over 15y    Statutory funding target
Low dependency   gilts + 0.5% flat                          2024 Funding Code, sig. maturity
Buy-out (proxy)  gilts flat, strengthened mortality (x0.95) Insurer price proxy
                 + 4% expense/profit loading

Mid-year convention: year-t cashflow discounted at t - 0.5 (monthly-in-advance proxy).

The TP and low-dependency bases use best-estimate mortality; the buy-out basis
uses strengthened mortality (lighter -> longer lives -> higher cost), plus an
explicit loading. Buy-out is a proxy - insurer pricing is not public - calibrated
to sit ~8-12% above low dependency, and stress-tested in the sensitivity suite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import BASIS, MID_YEAR_CONVENTION, MORTALITY, PROJECTION_YEARS
from data_curves import CURVES
from cashflows import project_cashflows

T = PROJECTION_YEARS


# --------------------------------------------------------------------------
# Basis spread term structures
# --------------------------------------------------------------------------
def tp_spread(t: np.ndarray) -> np.ndarray:
    """TP spread: +1.0% at t=0 stepping linearly to +0.5% by year 15, flat after."""
    t = np.asarray(t, dtype=float)
    frac = np.clip(t / BASIS.tp_step_years, 0.0, 1.0)
    return BASIS.tp_spread_start + (BASIS.tp_spread_end - BASIS.tp_spread_start) * frac


def ld_spread(t: np.ndarray) -> np.ndarray:
    return np.full_like(np.asarray(t, dtype=float), BASIS.ld_spread)


def buyout_spread(t: np.ndarray) -> np.ndarray:
    return np.zeros_like(np.asarray(t, dtype=float))    # gilts flat


# --------------------------------------------------------------------------
# Discount factors (mid-year) and PV
# --------------------------------------------------------------------------
def discount_factors(spread_fn, curve=CURVES, mid_year: float = MID_YEAR_CONVENTION,
                     n: int = T, level_shift: float = 0.0) -> np.ndarray:
    """
    v(t) for t=1..n using nominal spot + basis spread, discounted at t - mid_year.
    `level_shift` shifts the whole nominal curve (used for duration/sensitivity).
    """
    t = np.arange(1, n + 1, dtype=float)
    disc_t = t - mid_year
    spot = curve.nominal_spot(disc_t) + level_shift
    rate = spot + spread_fn(disc_t)
    return (1.0 + rate) ** (-disc_t)


def present_value(cashflows: np.ndarray, spread_fn, curve=CURVES,
                  level_shift: float = 0.0) -> float:
    v = discount_factors(spread_fn, curve=curve, n=len(cashflows), level_shift=level_shift)
    return float(np.sum(cashflows * v))


def effective_duration(cashflows: np.ndarray, spread_fn, curve=CURVES,
                       bump: float = 0.0001) -> float:
    """Modified/effective duration via a +/- bump to the whole curve."""
    pv_up = present_value(cashflows, spread_fn, curve, level_shift=+bump)
    pv_dn = present_value(cashflows, spread_fn, curve, level_shift=-bump)
    pv0 = present_value(cashflows, spread_fn, curve)
    return -(pv_up - pv_dn) / (2 * bump * pv0)


# --------------------------------------------------------------------------
# Cashflow variants by mortality basis (cached)
# --------------------------------------------------------------------------
_CF_CACHE: dict = {}


def cashflows_best_estimate() -> np.ndarray:
    if "be" not in _CF_CACHE:
        _CF_CACHE["be"] = project_cashflows(
            lt_rate=MORTALITY.lt_improvement, scaling=MORTALITY.scheme_scaling)["total"]
    return _CF_CACHE["be"]


def cashflows_buyout() -> np.ndarray:
    if "bo" not in _CF_CACHE:
        scaling = MORTALITY.scheme_scaling * BASIS.buyout_mortality_scaling
        _CF_CACHE["bo"] = project_cashflows(
            lt_rate=MORTALITY.lt_improvement, scaling=scaling)["total"]
    return _CF_CACHE["bo"]


# --------------------------------------------------------------------------
# The three liability values
# --------------------------------------------------------------------------
def value_all_bases(curve=CURVES) -> pd.DataFrame:
    cf_be = cashflows_best_estimate()
    cf_bo = cashflows_buyout()

    tp = present_value(cf_be, tp_spread, curve)
    ld = present_value(cf_be, ld_spread, curve)
    bo_raw = present_value(cf_bo, buyout_spread, curve)     # gilts flat, strengthened
    bo = bo_raw * (1.0 + BASIS.buyout_expense_loading)      # + expense/profit loading

    dur_tp = effective_duration(cf_be, tp_spread, curve)
    dur_ld = effective_duration(cf_be, ld_spread, curve)
    dur_bo = effective_duration(cf_bo, buyout_spread, curve)

    rows = [
        ("Technical provisions", tp, dur_tp),
        ("Low dependency", ld, dur_ld),
        ("Buy-out (proxy)", bo, dur_bo),
    ]
    df = pd.DataFrame(rows, columns=["basis", "liability", "duration"])
    df["liability_m"] = df["liability"] / 1e6
    df["vs_low_dep_%"] = 100 * (df["liability"] / ld - 1.0)
    return df


def liability_on_basis(basis: str, curve=CURVES, level_shift: float = 0.0) -> float:
    """Single liability value for a named basis with an optional curve shift."""
    if basis == "tp":
        return present_value(cashflows_best_estimate(), tp_spread, curve, level_shift)
    if basis == "ld":
        return present_value(cashflows_best_estimate(), ld_spread, curve, level_shift)
    if basis == "buyout":
        raw = present_value(cashflows_buyout(), buyout_spread, curve, level_shift)
        return raw * (1.0 + BASIS.buyout_expense_loading)
    raise ValueError(basis)


if __name__ == "__main__":
    df = value_all_bases()
    print("Liability values at 31-Mar-2026:\n")
    show = df[["basis", "liability_m", "duration", "vs_low_dep_%"]].copy()
    show.columns = ["Basis", "Liability (£m)", "Duration (yrs)", "vs LD (%)"]
    print(show.round(2).to_string(index=False))
    undisc = cashflows_best_estimate().sum()
    tp = df.loc[df.basis == "Technical provisions", "liability"].iloc[0]
    print(f"\nUndiscounted / TP liability ratio: {undisc/tp:.2f}x  (target 2.5-3.5x)")
    df.to_csv("../tables/liability_three_bases.csv", index=False)
