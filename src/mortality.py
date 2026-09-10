"""
mortality.py  (Stage 2)
=======================
Base mortality + CMI-style improvements on a COHORT basis.

Base table
----------
A Makeham law  q(x) = 1 - exp(-(A + B c^x))  is fitted to the shape of the ONS
National Life Tables (England & Wales). Parameters are documented below. This
is a transparent proxy for the CMI S3PA self-administered scheme tables, which
are not fully open. A scheme-experience scaling factor of 0.92 is then applied
(pension scheme members are wealthier and lighter than the national population).

Improvements
------------
A CMI-style constant long-term improvement rate (default 1.25% p.a.) is applied,
tapering linearly to zero between age 90 and 110. Improvements are applied on a
COHORT basis: as we project a life forward, both age and calendar year advance
together, so improvement factors accumulate along the diagonal, not across a
single row. Getting period-vs-cohort right is the classic CS2 trap.

Key function
------------
    survival_prob(age, sex, t, ...)  ->  probability a life aged `age` at the
    valuation date (2026) survives a further `t` years (cohort basis).
"""
from __future__ import annotations

import numpy as np

from config import MORTALITY

# --------------------------------------------------------------------------
# Makeham parameters calibrated to ONS ELT (England & Wales) shape.
#   A: age-independent (accident) term
#   B, c: Gompertz senescent term  mu = A + B c^x
# Calibrated so period q65 ~ 1.3% (M) / 0.9% (F) and q85 ~ 9.7% / 7.3%,
# matching recent ELT pensioner-age mortality.
# --------------------------------------------------------------------------
_MAKEHAM = {
    "M": {"A": 0.00030, "B": 1.79e-5, "c": 1.1064},
    "F": {"A": 0.00020, "B": 1.126e-5, "c": 1.1080},
}

MAX_AGE = 120


def base_qx(age, sex: str) -> np.ndarray:
    """Period mortality rate q_x from the Makeham base table (before scaling)."""
    p = _MAKEHAM[sex]
    age = np.asarray(age, dtype=float)
    mu = p["A"] + p["B"] * p["c"] ** age
    q = 1.0 - np.exp(-mu)
    return np.clip(q, 0.0, 1.0)


def scheme_qx(age, sex: str, scaling: float | None = None) -> np.ndarray:
    """Base q_x scaled to scheme experience (ONS x 0.92 by default)."""
    scaling = MORTALITY.scheme_scaling if scaling is None else scaling
    return np.clip(base_qx(age, sex) * scaling, 0.0, 1.0)


def _improvement_rate(age, lt_rate: float) -> np.ndarray:
    """
    Annual mortality improvement rate by age. Flat at the long-term rate up to
    the taper age, then linearly to zero by the zero age. Simple, defensible,
    CMI-style (not the full CMI core projection).
    """
    age = np.asarray(age, dtype=float)
    taper = MORTALITY.improvement_taper_age
    zero = MORTALITY.improvement_zero_age
    frac = np.clip((zero - age) / (zero - taper), 0.0, 1.0)
    rate = np.where(age <= taper, lt_rate, lt_rate * frac)
    return np.clip(rate, 0.0, lt_rate)


def qx_cohort(age, sex: str, years_from_base, lt_rate: float | None = None,
              scaling: float | None = None) -> np.ndarray:
    """
    Mortality rate for a life currently aged `age` (at valuation date), `years_from_base`
    years later, on a cohort basis. At that point the life is aged (age + years_from_base)
    in calendar year (base_year + years_from_base), so improvements have compounded
    over `years_from_base` years at the improvement rate *for that attained age*.
    """
    lt_rate = MORTALITY.lt_improvement if lt_rate is None else lt_rate
    age = np.asarray(age, dtype=float)
    k = np.asarray(years_from_base, dtype=float)
    attained = age + k
    q0 = scheme_qx(attained, sex, scaling)
    imp = _improvement_rate(attained, lt_rate)
    factor = (1.0 - imp) ** k                  # cohort: compound over k years
    return np.clip(q0 * factor, 0.0, 1.0)


def survival_prob(age, sex: str, t, lt_rate: float | None = None,
                  scaling: float | None = None) -> np.ndarray:
    """
    t p_x on a cohort basis: probability a life aged `age` at the valuation date
    survives a further `t` complete years.

    `age` and `t` may be scalars or broadcastable arrays. Vectorised over a grid
    of future years internally.
    """
    age_arr = np.atleast_1d(np.asarray(age, dtype=float))
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    t_max = int(np.ceil(t_arr.max())) if t_arr.size else 0
    if t_max <= 0:
        out = np.ones((age_arr.size, t_arr.size))
        return _squeeze(out, age, t)

    # yearly survival probabilities for k = 0..t_max-1 (age x sim grid)
    ks = np.arange(t_max)                                   # (t_max,)
    # attained-age grid: (n_age, t_max)
    q_grid = np.empty((age_arr.size, t_max))
    for i, a in enumerate(age_arr):
        q_grid[i, :] = qx_cohort(a, sex, ks, lt_rate=lt_rate, scaling=scaling)
    p_year = 1.0 - q_grid
    cum = np.cumprod(p_year, axis=1)                        # tp_x for t=1..t_max
    cum = np.concatenate([np.ones((age_arr.size, 1)), cum], axis=1)  # prepend t=0

    # gather at requested t (integer years; linear interp for fractional t)
    out = np.empty((age_arr.size, t_arr.size))
    grid_t = np.arange(t_max + 1)
    for j, tt in enumerate(t_arr):
        out[:, j] = _interp_rows(cum, grid_t, tt)
    return _squeeze(out, age, t)


def _interp_rows(cum, grid_t, tt):
    if tt <= 0:
        return np.ones(cum.shape[0])
    if tt >= grid_t[-1]:
        return cum[:, -1]
    lo = int(np.floor(tt))
    frac = tt - lo
    return cum[:, lo] * (1 - frac) + cum[:, lo + 1] * frac


def _squeeze(out, age, t):
    scalar_age = np.ndim(age) == 0
    scalar_t = np.ndim(t) == 0
    if scalar_age and scalar_t:
        return float(out[0, 0])
    if scalar_age:
        return out[0, :]
    if scalar_t:
        return out[:, 0]
    return out


def life_expectancy(age, sex: str, cohort: bool = True, lt_rate: float | None = None,
                    scaling: float | None = None, curtate: bool = False) -> float:
    """
    Complete expectation of life e_x. Cohort (default) uses survival_prob with
    improvements; period sets improvements to zero (lt_rate=0 flat). Uses the
    sum of tp_x with a +0.5 completion term for the complete expectation.
    """
    max_t = MAX_AGE - int(np.floor(age))
    ts = np.arange(1, max_t + 1)
    if cohort:
        tp = survival_prob(age, sex, ts, lt_rate=lt_rate, scaling=scaling)
    else:
        # period: freeze the table at the base year (no improvement compounding)
        q = scheme_qx(age + np.arange(0, max_t), sex, scaling)
        tp = np.cumprod(1.0 - q[:-1]) if max_t > 1 else np.array([1.0])
        tp = np.concatenate([tp, [tp[-1] * (1 - q[-1])]]) if max_t >= 1 else tp
        tp = tp[:len(ts)]
    ex = tp.sum()
    return ex if curtate else ex + 0.5


def life_expectancy_at(target_age: int, sex: str, calendar_offset: int = 0,
                       lt_rate: float | None = None, scaling: float | None = None) -> float:
    """
    Complete cohort expectation of life at `target_age`, for a person who reaches
    `target_age` `calendar_offset` years after the base year. A 45-year-old today
    reaches 65 in 20 years, so their mortality at 65+ carries 20 extra years of
    improvement (calendar_offset=20). This is what shows the cohort effect.
    """
    lt_rate = MORTALITY.lt_improvement if lt_rate is None else lt_rate
    max_t = MAX_AGE - target_age
    ks = np.arange(0, max_t)
    attained = target_age + ks
    q0 = scheme_qx(attained, sex, scaling)
    imp = _improvement_rate(attained, lt_rate)
    q = np.clip(q0 * (1.0 - imp) ** (calendar_offset + ks), 0.0, 1.0)
    tp = np.cumprod(1.0 - q)
    return float(tp.sum()) + 0.5


if __name__ == "__main__":
    print("Period life expectancy at 65 (no improvements):")
    print(f"  Male:   {life_expectancy(65, 'M', cohort=False):.1f}")
    print(f"  Female: {life_expectancy(65, 'F', cohort=False):.1f}")
    print("\nCohort life expectancy at 65 (LT improvement 1.25%):")
    print(f"  Male:   {life_expectancy(65, 'M', cohort=True):.1f}")
    print(f"  Female: {life_expectancy(65, 'F', cohort=True):.1f}")
    print("\nSample survival probs, male aged 65:")
    for t in (5, 10, 20, 30):
        print(f"  {t}p65 = {survival_prob(65, 'M', t):.4f}")
