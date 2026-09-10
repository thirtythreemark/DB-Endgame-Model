
from __future__ import annotations

import numpy as np

from config import (ASSETS, BASIS, ENDGAME, ESG, N_SIMS, SEED,
                    SIM_PROJECTION_YEARS, START_TP_FUNDING, MID_YEAR_CONVENTION,
                    PROJECTION_YEARS)
from data_curves import CURVES
from valuation import (cashflows_best_estimate, cashflows_buyout,
                       tp_spread, ld_spread, buyout_spread, value_all_bases)

T = PROJECTION_YEARS
Y = SIM_PROJECTION_YEARS

INFL_LINKED_SHARE = 0.60      # ~60% of liabilities are inflation-linked (post-97 tranches)
RECOVERY_YEARS = 7            # deficit-repair recovery period on the TP basis
_BUYOUT_LOAD = 1.0 + BASIS.buyout_expense_loading

# --------------------------------------------------------------------------
# Precomputed discounting grids (term k -> mid-year time k-0.5)
# --------------------------------------------------------------------------
_times = np.arange(1, T + 1) - MID_YEAR_CONVENTION
_spot = CURVES.nominal_spot(_times)
_spread = {"tp": tp_spread(_times), "ld": ld_spread(_times), "buyout": buyout_spread(_times)}
_cf = {"tp": cashflows_best_estimate(), "ld": cashflows_best_estimate(),
       "buyout": cashflows_buyout()}


def _liab(basis: str, elapsed: int, shift: np.ndarray) -> np.ndarray:
    """Liability on `basis` after `elapsed` whole years paid, for shift array (nsims,)."""
    cf = _cf[basis][elapsed:]
    m = cf.shape[0]
    if m == 0:
        return np.zeros_like(shift)
    times = _times[:m]
    r0 = _spot[:m] + _spread[basis][:m]
    base = np.clip(1.0 + r0[None, :] + shift[:, None], 1e-6, None)
    pv = (cf[None, :] * base ** (-times[None, :])).sum(axis=1)
    return pv * _BUYOUT_LOAD if basis == "buyout" else pv


# --------------------------------------------------------------------------
# Economic scenario generator
# --------------------------------------------------------------------------
def _correlated_shocks(rng, nsims, nyears):
    corr = np.array([
        [1.0, ESG.corr_yield_infl, ESG.corr_yield_growth],
        [ESG.corr_yield_infl, 1.0, 0.0],
        [ESG.corr_yield_growth, 0.0, 1.0],
    ])
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((nsims, nyears, 3))
    return z @ L.T


# --------------------------------------------------------------------------
# Main projection
# --------------------------------------------------------------------------
def run_projection(n_sims: int = N_SIMS, n_years: int = Y, seed: int = SEED,
                   growth_weight: float | None = None, hedge_ratio: float | None = None,
                   start_assets: float | None = None,
                   surplus_release_margin: float | None = None,
                   release_surplus: bool = False, deficit_repair: bool = True):
    """Run the ALM Monte Carlo. Returns dict of (n_sims, n_years+1) funding arrays."""
    rng = np.random.default_rng(seed)
    gw = ASSETS.growth_weight if growth_weight is None else growth_weight
    mw = 1.0 - gw
    hr = ASSETS.hedge_ratio if hedge_ratio is None else hedge_ratio
    margin = ENDGAME.surplus_release_margin if surplus_release_margin is None else surplus_release_margin

    base = value_all_bases()
    L_tp0 = base.loc[base.basis == "Technical provisions", "liability"].iloc[0]
    L_ld0 = base.loc[base.basis == "Low dependency", "liability"].iloc[0]
    L_bo0 = base.loc[base.basis == "Buy-out (proxy)", "liability"].iloc[0]
    A0 = START_TP_FUNDING * L_tp0 if start_assets is None else start_assets

    shift = np.zeros(n_sims)
    A_g = np.full(n_sims, A0 * gw)
    A_m = np.full(n_sims, A0 * mw)

    fund_tp = np.zeros((n_sims, n_years + 1))
    fund_ld = np.zeros((n_sims, n_years + 1))
    fund_bo = np.zeros((n_sims, n_years + 1))
    assets_rec = np.zeros((n_sims, n_years + 1))
    surplus_rec = np.zeros((n_sims, n_years + 1))
    cum_surplus = np.zeros(n_sims)

    fund_tp[:, 0], fund_ld[:, 0], fund_bo[:, 0] = A0 / L_tp0, A0 / L_ld0, A0 / L_bo0
    assets_rec[:, 0] = A0

    shocks = _correlated_shocks(rng, n_sims, n_years)
    infl_dev = np.zeros(n_sims)
    short0 = CURVES.nominal_spot(1.0)

    for y in range(1, n_years + 1):
        eb = y - 1
        z_rate, z_infl, z_growth = shocks[:, y - 1, 0], shocks[:, y - 1, 1], shocks[:, y - 1, 2]

        short = short0 + shift + ASSETS.credit_spread          # running yield on matching
        # 1. roll the curve (mean-reverting level shift)
        shift_new = shift + ESG.yield_mean_reversion * (0.0 - shift) + ESG.yield_vol * z_rate
        # inflation AR(1) deviation from long-run; one-year surprise
        infl_dev = (1 - ESG.infl_mean_reversion) * infl_dev + ESG.infl_vol * z_infl
        infl_prop = INFL_LINKED_SHARE * infl_dev               # proportional liability lift

        # 2. rate P&L on TP liabilities (remaining cashflows, before roll-off)
        L_tp_old = _liab("tp", eb, shift)
        L_tp_new = _liab("tp", eb, shift_new)
        dL_tp_rate = L_tp_new - L_tp_old
        infl_pnl_tp = L_tp_new * infl_prop

        # 3. asset returns
        # growth: lognormal so the median carries a realistic volatility drag,
        # while the arithmetic expected return stays at gilts + excess.
        g_mean = short0 + shift + ASSETS.growth_excess_return
        sig = ASSETS.growth_vol
        g_factor = (1.0 + g_mean) * np.exp(sig * z_growth - 0.5 * sig * sig)
        A_g_new = A_g * g_factor
        hedge_pnl = hr * (dL_tp_rate + infl_pnl_tp)            # matching hedge offsets hr
        A_m_new = A_m * (1.0 + short) + hedge_pnl
        A = A_g_new + A_m_new

        # 4. pay the year's benefit outgo
        A = A - _cf["tp"][y - 1]

        # liabilities after roll-off, with full inflation lift (denominator carries all,
        # assets carried hr of it -> net (1-hr) unhedged, which is the intent)
        infl_factor = 1.0 + infl_prop
        L_tp = _liab("tp", y, shift_new) * infl_factor
        L_ld = _liab("ld", y, shift_new) * infl_factor
        L_bo = _liab("buyout", y, shift_new) * infl_factor

        # 5. deficit-repair contributions on TP basis
        if deficit_repair:
            A = A + np.maximum(L_tp - A, 0.0) / RECOVERY_YEARS

        # Route B: release surplus above low-dependency + margin
        if release_surplus:
            excess = np.maximum(A - L_ld * (1.0 + margin), 0.0)
            A = A - excess
            cum_surplus += excess
            surplus_rec[:, y] = excess

        A = np.maximum(A, 1e-6)
        A_g, A_m = A * gw, A * mw

        # 6. record
        fund_tp[:, y] = A / np.maximum(L_tp, 1e-6)
        fund_ld[:, y] = A / np.maximum(L_ld, 1e-6)
        fund_bo[:, y] = A / np.maximum(L_bo, 1e-6)
        assets_rec[:, y] = A
        shift = shift_new

    return dict(fund_tp=fund_tp, fund_ld=fund_ld, fund_bo=fund_bo,
                assets=assets_rec, surplus=surplus_rec, cum_surplus=cum_surplus,
                L_tp0=L_tp0, L_ld0=L_ld0, L_bo0=L_bo0, A0=A0,
                growth_weight=gw, hedge_ratio=hr)


if __name__ == "__main__":
    import time
    t0 = time.time()
    res = run_projection()
    dt = time.time() - t0
    print(f"Ran {N_SIMS:,} sims x {Y} years in {dt:.1f}s\n")
    for name, arr in [("TP", res["fund_tp"]), ("Low dependency", res["fund_ld"]),
                      ("Buy-out", res["fund_bo"])]:
        final = arr[:, -1]
        print(f"  {name:<15} funding at yr {Y}: median {np.median(final):.1%}  "
              f"5th {np.percentile(final,5):.1%}  95th {np.percentile(final,95):.1%}")
    print(f"\n  Start funding: TP {res['A0']/res['L_tp0']:.1%}, "
          f"LD {res['A0']/res['L_ld0']:.1%}, BO {res['A0']/res['L_bo0']:.1%}")
