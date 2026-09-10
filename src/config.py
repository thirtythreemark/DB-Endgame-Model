"""
config.py
=========
Single source of truth for every assumption in the DB Endgame model.

Every number a pensions actuary would question lives here, with a one-line
justification, mirroring the assumptions table in the trustee note (Section 5
of the build guide). Nothing downstream should hard-code an assumption; it
should import it from here so the model is auditable and reproducible.

Mark Cheung, autumn 2026.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
# Outputs are written at the project root (flat layout): charts/, tables/,
# validation/, and reports (docx/pdf/xlsx) directly in the root.
OUT = ROOT
OUT_CHARTS = ROOT / "charts"
OUT_TABLES = ROOT / "tables"
OUT_VALID = ROOT / "validation"

for _p in (DATA_RAW, DATA_PROC, OUT_CHARTS, OUT_TABLES, OUT_VALID):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 20260331          # fixed seed -> frozen membership + reproducible sims

# --------------------------------------------------------------------------
# Valuation basis
# --------------------------------------------------------------------------
VALUATION_DATE = _dt.date(2026, 3, 31)   # single base date used everywhere
NORMAL_RETIREMENT_AGE = 65               # NRA for deferred members

PROJECTION_YEARS = 70          # deterministic cashflow horizon (near run-off)
SIM_PROJECTION_YEARS = 20      # stochastic endgame horizon
N_SIMS = 5_000                 # Monte Carlo paths
MID_YEAR_CONVENTION = 0.5      # discount year-t cashflow at t - 0.5 (monthly-in-advance proxy)

# --------------------------------------------------------------------------
# Membership target profile (Stage 1)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MembershipConfig:
    n_pensioners: int = 1_200
    n_deferreds: int = 1_800

    # Age distributions (truncated normals)
    pensioner_age_mean: float = 73.0
    pensioner_age_sd: float = 8.0
    pensioner_age_min: int = 60
    pensioner_age_max: int = 95

    deferred_age_mean: float = 54.0
    deferred_age_sd: float = 6.0
    deferred_age_min: int = 40
    deferred_age_max: int = 64

    # Sex split: overall ~60/40 M/F, pensioners skew more male
    male_share_pensioner: float = 0.68
    male_share_deferred: float = 0.55

    # Pension amounts: lognormal with a long right tail.
    # NOTE (documented modelling choice): the build-guide gives a ~£4,500 median
    # AND a £15-20m year-1 outgo target. In year 1 only pensioners are paid
    # (deferreds get nothing until 65), so a single £4,500 median cannot deliver
    # both. We therefore apply the £4,500 median to DEFERREDS and give PENSIONERS
    # a higher median (~£9,000) - realistic, as pensions in payment are larger and
    # have had years of increases - calibrated so year-1 outgo lands at £15-20m.
    pension_median_pensioner: float = 9_000.0
    pension_median_deferred: float = 4_500.0
    pension_sigma: float = 0.85     # log-scale sd -> long right tail

    # Tranche split by value (pre-97 / 97-05 / post-05) for a scheme closed ~2010
    tranche_split: tuple = (0.40, 0.30, 0.30)


MEMBERSHIP = MembershipConfig()

# --------------------------------------------------------------------------
# Benefit increase / revaluation rules by tranche
# --------------------------------------------------------------------------
# Increase in payment and revaluation in deferment differ by accrual period.
# 'kind' drives the increase_factor logic in cashflows.py.
TRANCHES = {
    "pre97": {
        "label": "Pre-1997",
        "increase_in_payment": {"kind": "fixed", "rate": 0.0},          # typically nil
        "revaluation": {"kind": "statutory_cpi", "cap": 0.05},          # statutory in deferment
    },
    "9705": {
        "label": "1997-2005",
        "increase_in_payment": {"kind": "lpi", "cap": 0.05, "floor": 0.0},   # LPI(5%)
        "revaluation": {"kind": "cpi", "cap": 0.05, "floor": 0.0},           # CPI capped 5%
    },
    "post05": {
        "label": "Post-2005",
        "increase_in_payment": {"kind": "lpi", "cap": 0.025, "floor": 0.0},  # LPI(2.5%)
        "revaluation": {"kind": "cpi", "cap": 0.025, "floor": 0.0},          # CPI capped 2.5%
    },
}

# --------------------------------------------------------------------------
# Contingent spouse's pension
# --------------------------------------------------------------------------
SPOUSE_FRACTION = 0.50            # spouse gets 50% of member's pension
PROP_MARRIED = {"M": 0.80, "F": 0.65}   # proportion married at retirement
SPOUSE_AGE_DIFF = {              # spouse age relative to member (years)
    "M": -3,                     # male members: wife 3 years younger
    "F": +3,                     # female members: husband 3 years older
}

# --------------------------------------------------------------------------
# Economic base-date assumptions
# --------------------------------------------------------------------------
# CPI derived from market-implied RPI less a wedge (post-2030 RPI reform assumed).
RPI_CPI_WEDGE = 0.007            # 0.70% p.a.

# --------------------------------------------------------------------------
# Discount bases (Stage 4)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class BasisConfig:
    # Technical provisions: gilts + 1.0% stepping down to +0.5% over 15 years
    tp_spread_start: float = 0.010
    tp_spread_end: float = 0.005
    tp_step_years: int = 15

    # Low dependency: gilts + 0.5% flat (2024 DB Funding Code at significant maturity)
    ld_spread: float = 0.005

    # Buy-out proxy: low-dependency cashflows on gilts-flat + loading + longevity margin.
    # Calibrated so buy-out sits ~8-12% above low dependency (typical market range).
    # Gilts-flat alone lifts ~7% over the ~15yr duration, so the loading + longevity
    # margin are kept modest to stay in range; loading is stress-tested in Stage 7.
    buyout_expense_loading: float = 0.030    # 3.0% expense/profit loading
    buyout_mortality_scaling: float = 0.97   # strengthen mortality (multiply rates)


BASIS = BasisConfig()

# --------------------------------------------------------------------------
# Mortality (Stage 2)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MortalityConfig:
    scheme_scaling: float = 0.92        # ONS ELT x 0.92 (proxy for S3PA; lighter lives)
    lt_improvement: float = 0.0125      # CMI-style long-term rate 1.25% p.a.
    lt_improvement_low: float = 0.010   # sensitivity
    lt_improvement_high: float = 0.015  # sensitivity
    improvement_taper_age: int = 90     # improvements taper to zero by this age
    improvement_zero_age: int = 110     # no improvement at/after
    base_year: int = 2026               # base table year (aligned to valuation date)


MORTALITY = MortalityConfig()

# --------------------------------------------------------------------------
# Assets & ESG (Stage 5)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AssetConfig:
    growth_weight: float = 0.25         # 25% growth assets
    matching_weight: float = 0.75       # 75% matching (gilts + buy-and-maintain credit)
    hedge_ratio: float = 0.80           # 80% IR+inflation hedge on TP liabilities

    growth_excess_return: float = 0.035  # 3.5% expected return above gilts
    growth_vol: float = 0.15             # 15% annual volatility
    matching_duration: float = 14.0      # duration of matching portfolio (yrs)
    credit_spread: float = 0.004         # buy-and-maintain credit spread over gilts,
                                         # NET of expected defaults/downgrades


ASSETS = AssetConfig()

@dataclass(frozen=True)
class ESGConfig:
    # Yield level shocks (correlated normal on the gilt curve level)
    yield_vol: float = 0.011            # ~1.1% annual vol on yields
    yield_mean_reversion: float = 0.10  # Vasicek-style pull toward long-run level
    yield_long_run: float = 0.045       # long-run nominal yield level

    # Inflation: mean-reverting to 2.5% CPI, 1% vol
    infl_long_run: float = 0.025
    infl_mean_reversion: float = 0.30
    infl_vol: float = 0.010

    # Correlations
    corr_yield_infl: float = 0.30       # yields and inflation positively correlated
    corr_yield_growth: float = -0.10    # growth vs rates mildly negative


ESG = ESGConfig()

# --------------------------------------------------------------------------
# Endgame strategy (Stage 6)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EndgameConfig:
    surplus_release_margin: float = 0.05     # release above low-dependency + 5%
    buyout_target_funding: float = 1.00      # buy-out achieved at 100% on buy-out basis
    ld_floor: float = 1.00                   # downside threshold: 100% low dependency
    # Growth-asset allocations to sweep for the risk/return frontier
    frontier_growth_weights: tuple = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)


ENDGAME = EndgameConfig()

# --------------------------------------------------------------------------
# Starting funding position (calibrated so TP funding is broadly PPF-7800-like)
# --------------------------------------------------------------------------
# Assets set as a ratio of the TP liability at the base date; ~1.05 keeps the
# scheme a little above 100% TP, in the region of the PPF 7800 index in 2026.
START_TP_FUNDING = 1.05
