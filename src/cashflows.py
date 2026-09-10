"""
cashflows.py  (Stage 3)
=======================
Deterministic expected benefit-outgo projection, year by year, for ~70 years.

For each member-tranche and future year t the expected outgo is:

  member's own pension:
      pension_t = tranche_pension x increase_factor(t) x tPx

  contingent spouse's pension (NOT additive to the member's own - it is
  conditional on the member having died):
      spouse_t = tranche_pension x spouse_fraction x prop_married
                 x increase_factor(t) x P(member died by t) x P(spouse alive at t)

Traps handled explicitly
------------------------
 * Increases COMPOUND: (1+i_1)(1+i_2)...  not 1 + sum(i).
 * Caps bind ANNUALLY: LPI(5%) caps each year's increase, not the cumulative.
 * Deferreds revalued ONCE to 65, then increased in payment - never both from t=0.
 * Cohort survival (mortality.survival_prob).
 * Mid-year discounting handled later (valuation.py): cashflows here are the
   expected amount paid during projection year t.

Increases use forward CPI/RPI implied by the base-date curve (deterministic).
Under stochastic projection the cashflow vector is held fixed and re-discounted
with the rolled curve (documented simplification in the write-up).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (NORMAL_RETIREMENT_AGE, PROJECTION_YEARS, PROP_MARRIED,
                    SPOUSE_AGE_DIFF, SPOUSE_FRACTION, TRANCHES, MORTALITY)
from data_curves import CURVES
from membership import load_membership
from mortality import survival_prob

T = PROJECTION_YEARS
_OPP = {"M": "F", "F": "M"}


# --------------------------------------------------------------------------
# Forward inflation from the base-date curve
# --------------------------------------------------------------------------
def forward_cpi(n_years: int = T) -> np.ndarray:
    """One-year forward CPI rates for years 1..n_years from the CPI spot curve."""
    ts = np.arange(0, n_years + 1)
    cpi_spot = np.concatenate([[0.0], CURVES.cpi(ts[1:])])   # spot at t>=1
    cum = (1.0 + cpi_spot) ** ts                              # cumulative index
    fwd = cum[1:] / cum[:-1] - 1.0                            # forward year t
    return fwd


def forward_rpi(n_years: int = T) -> np.ndarray:
    ts = np.arange(0, n_years + 1)
    rpi_spot = np.concatenate([[0.0], CURVES.implied_rpi(ts[1:])])
    cum = (1.0 + rpi_spot) ** ts
    return cum[1:] / cum[:-1] - 1.0


# --------------------------------------------------------------------------
# Increase-rate vectors by tranche
# --------------------------------------------------------------------------
def _annual_rate(rule: dict, fwd_cpi: np.ndarray, fwd_rpi: np.ndarray) -> np.ndarray:
    """Vector of annual increase rates (len T) for a given increase/revaluation rule."""
    kind = rule["kind"]
    if kind == "fixed":
        return np.full(T, rule["rate"])
    # inflation-linked: choose index (CPI for revaluation/LPI-in-payment here)
    infl = fwd_cpi                     # scheme increases are CPI-linked in this model
    cap = rule.get("cap", np.inf)
    floor = rule.get("floor", 0.0)
    if kind == "statutory_cpi":        # statutory deferred revaluation ~ CPI capped 5%
        cap = rule.get("cap", 0.05)
    return np.clip(infl, floor, cap)


def build_increase_factors():
    """
    Returns two dicts keyed by tranche:
      Fpay[tr]  : pensioner-basis cumulative in-payment factor, Fpay[t]=prod_{k<t}(1+i_k)
                  (length T, Fpay[0]=1 meaning year-1 payment is at the current rate)
      Frev[tr]  : cumulative revaluation factor in deferment, Frev[d]=prod_{k<=d}(1+r_k)
                  (length T+1, Frev[0]=1)
      inpay_rate[tr], reval_rate[tr] kept for inspection.
    """
    fwd_cpi = forward_cpi()
    fwd_rpi = forward_rpi()
    Fpay, Frev, inpay_rate, reval_rate = {}, {}, {}, {}
    for tr, spec in TRANCHES.items():
        i_pay = _annual_rate(spec["increase_in_payment"], fwd_cpi, fwd_rpi)
        i_rev = _annual_rate(spec["revaluation"], fwd_cpi, fwd_rpi)
        inpay_rate[tr] = i_pay
        reval_rate[tr] = i_rev
        # in-payment cumulative factor: year-1 payment at current rate -> Fpay[0]=1
        fpay = np.concatenate([[1.0], np.cumprod(1.0 + i_pay[:-1])])   # len T
        Fpay[tr] = fpay
        # revaluation cumulative factor over d years -> Frev[0]=1, Frev[d]=prod_{k=1..d}
        frev = np.concatenate([[1.0], np.cumprod(1.0 + i_rev)])        # len T+1
        Frev[tr] = frev
    return Fpay, Frev, inpay_rate, reval_rate


# --------------------------------------------------------------------------
# Survival caches (keyed by sex, age) - cohort basis
# --------------------------------------------------------------------------
def _survival_cache(df: pd.DataFrame, lt_rate: float, scaling: float):
    """Precompute tPx vectors (t=1..T) for every (sex, age) appearing in df, plus spouses."""
    ts = np.arange(1, T + 1)
    cache = {}
    # members
    for sex in ("M", "F"):
        ages = sorted(df.loc[df.sex == sex, "age"].unique())
        for a in ages:
            cache[("mem", sex, int(a))] = survival_prob(int(a), sex, ts,
                                                         lt_rate=lt_rate, scaling=scaling)
    # spouses: spouse of a member (sex s, age a) has opposite sex and age a+diff
    for sex in ("M", "F"):
        ssex = _OPP[sex]
        diff = SPOUSE_AGE_DIFF[sex]
        ages = sorted(df.loc[df.sex == sex, "age"].unique())
        for a in ages:
            sage = int(np.clip(a + diff, 20, 95))
            key = ("sp", sex, int(a))
            if key not in cache:
                cache[key] = survival_prob(sage, ssex, ts, lt_rate=lt_rate, scaling=scaling)
    return cache


# --------------------------------------------------------------------------
# Main projection
# --------------------------------------------------------------------------
def project_cashflows(df: pd.DataFrame | None = None, lt_rate: float | None = None,
                      scaling: float | None = None, include_spouse: bool = True):
    """
    Project expected total benefit outgo per year (len T) and its components.

    Returns dict:
      own    : member's own pension cashflow vector (T,)
      spouse : contingent spouse's pension vector (T,)
      total  : own + spouse (T,)
      years  : 1..T
    """
    df = load_membership() if df is None else df
    lt_rate = MORTALITY.lt_improvement if lt_rate is None else lt_rate
    scaling = MORTALITY.scheme_scaling if scaling is None else scaling

    Fpay, Frev, _, _ = build_increase_factors()
    cache = _survival_cache(df, lt_rate, scaling)

    own = np.zeros(T)
    spouse = np.zeros(T)
    years_idx = np.arange(T)         # 0-based: projection year t = idx+1

    for row in df.itertuples(index=False):
        tr = row.tranche
        p0 = row.tranche_pension
        sex = row.sex
        age = int(row.age)
        fpay = Fpay[tr]              # (T,)
        frev = Frev[tr]             # (T+1,)
        mem_surv = cache[("mem", sex, age)]      # tPx for t=1..T
        sp_surv = cache[("sp", sex, age)]

        if row.status == "pensioner":
            # in payment now: payment year t = p0 * Fpay[t-1] * tPx
            pay_amount = p0 * fpay                        # (T,) escalation
            own += pay_amount * mem_surv
            # spouse contingent pension, in payment escalation, conditional on member death
            if include_spouse:
                base = p0 * SPOUSE_FRACTION * PROP_MARRIED[sex]
                sp_amount = base * fpay
                spouse += sp_amount * (1.0 - mem_surv) * sp_surv
        else:
            # deferred: revalue to 65 once, then increase in payment from 65
            d = max(0, NORMAL_RETIREMENT_AGE - age)       # whole years to NRA
            if d >= T:
                continue
            reval = frev[min(d, T)]                        # revaluation factor to 65
            pension_at_65 = p0 * reval
            # payments for projection years t = d+1 .. T (0-based idx d..T-1)
            idx = np.arange(d, T)
            years_in_pay = idx - d                         # 0 at first payment
            escalation = fpay[years_in_pay]                # in-payment escalation
            amt = pension_at_65 * escalation
            own[idx] += amt * mem_surv[idx]
            if include_spouse:
                base = pension_at_65 * SPOUSE_FRACTION * PROP_MARRIED[sex]
                sp_amt = base * escalation
                spouse[idx] += sp_amt * (1.0 - mem_surv[idx]) * sp_surv[idx]

    total = own + spouse
    return {"own": own, "spouse": spouse, "total": total,
            "years": np.arange(1, T + 1)}


def cashflow_frame(res: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "year": res["years"],
        "own_pension": res["own"],
        "spouse_pension": res["spouse"],
        "total": res["total"],
    })


if __name__ == "__main__":
    res = project_cashflows()
    cf = cashflow_frame(res)
    peak_year = int(cf.year[cf.total.idxmax()])
    print(f"Projected benefit outgo (best-estimate mortality):")
    print(f"  Year 1 total:      £{cf.total.iloc[0]/1e6:6.2f}m "
          f"(own £{cf.own_pension.iloc[0]/1e6:.2f}m, spouse £{cf.spouse_pension.iloc[0]/1e6:.2f}m)")
    print(f"  Peak year:         {peak_year}  at £{cf.total.max()/1e6:.2f}m")
    print(f"  Year 30:           £{cf.total.iloc[29]/1e6:6.2f}m")
    print(f"  Year 55:           £{cf.total.iloc[54]/1e6:6.2f}m")
    print(f"  Undiscounted total:£{cf.total.sum()/1e6:,.0f}m")
    cf.to_csv("../tables/deterministic_cashflows.csv", index=False)
    print("  saved tables/deterministic_cashflows.csv")
