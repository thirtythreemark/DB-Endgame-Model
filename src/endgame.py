
from __future__ import annotations

import numpy as np
import pandas as pd

from config import ENDGAME, N_SIMS, SIM_PROJECTION_YEARS
from projection import run_projection

Y = SIM_PROJECTION_YEARS


def time_to_buyout(res: dict | None = None) -> dict:
    res = run_projection(release_surplus=False) if res is None else res
    fbo = res["fund_bo"]                                     # (nsims, Y+1)
    reached = fbo >= ENDGAME.buyout_target_funding

    n_sims = fbo.shape[0]
    first = np.full(n_sims, Y + 1)
    any_reached = reached.any(axis=1)
    first[any_reached] = reached[any_reached].argmax(axis=1)
    reached_mask = first <= Y
    times = first[reached_mask]

    def pct(p):
        return float(np.percentile(times, p)) if times.size else np.nan

    summary = {
        "median": pct(50), "p25": pct(25), "p75": pct(75),
        "p_within_5": float((first <= 5).mean()),
        "p_within_10": float((first <= 10).mean()),
        "p_within_15": float((first <= 15).mean()),
        "p_never_20": float((first > Y).mean()),
        "first_year": first,
        "start_bo_funding": res["A0"] / res["L_bo0"],
    }
    return summary


def run_on(res: dict | None = None) -> dict:
    res = run_projection(release_surplus=True) if res is None else res
    fld = res["fund_ld"]                                     # low-dependency funding path
    cum_surplus = res["cum_surplus"]
    # probability of falling below 100% LD at any point (years 1..Y)
    below = (fld[:, 1:] < ENDGAME.ld_floor).any(axis=1)
    worst = fld[:, 1:].min(axis=1)                           # worst LD funding per sim
    summary = {
        "expected_surplus": float(cum_surplus.mean()),
        "median_surplus": float(np.median(cum_surplus)),
        "p_below_ld": float(below.mean()),
        "worst_5pct_funding": float(np.percentile(worst, 5)),
        "cum_surplus": cum_surplus,
        "L_ld0": res["L_ld0"],
    }
    return summary



def frontier(growth_weights=None, hedge_ratio=None) -> pd.DataFrame:
    weights = growth_weights or ENDGAME.frontier_growth_weights
    rows = []
    for gw in weights:
        res = run_projection(growth_weight=gw, hedge_ratio=hedge_ratio,
                             release_surplus=True)
        fld = res["fund_ld"][:, 1:]
        below = (fld < ENDGAME.ld_floor).any(axis=1)
        below95 = (fld < 0.95).any(axis=1)
        worst = fld.min(axis=1)
        exp_surplus = res["cum_surplus"].mean()
        # also record buy-out time under this allocation (info)
        tb = time_to_buyout(run_projection(growth_weight=gw, hedge_ratio=hedge_ratio,
                                            release_surplus=False))
        rows.append(dict(
            growth_weight=gw,
            expected_surplus_m=exp_surplus / 1e6,
            prob_below_ld=below.mean(),
            prob_below_95_ld=below95.mean(),
            worst_5pct_funding=np.percentile(worst, 5),
            median_time_to_buyout=tb["median"],
            p_buyout_10=tb["p_within_10"],
        ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 64)
    print("ROUTE A - Buy-out: time-to-buy-out distribution")
    print("=" * 64)
    a = time_to_buyout()
    print(f"  Start buy-out funding: {a['start_bo_funding']:.1%}")
    print(f"  Median time to buy-out:   {a['median']:.0f} years")
    print(f"  25th / 75th percentile:   {a['p25']:.0f} / {a['p75']:.0f} years")
    print(f"  P(buy-out within 5 yrs):  {a['p_within_5']:.1%}")
    print(f"  P(buy-out within 10 yrs): {a['p_within_10']:.1%}")
    print(f"  P(buy-out within 15 yrs): {a['p_within_15']:.1%}")
    print(f"  Never within 20 yrs:      {a['p_never_20']:.1%}")

    print("\n" + "=" * 64)
    print("ROUTE B - Run-on with surplus release")
    print("=" * 64)
    b = run_on()
    print(f"  Expected surplus released (20y): £{b['expected_surplus']/1e6:,.1f}m")
    print(f"  Median surplus released:         £{b['median_surplus']/1e6:,.1f}m")
    print(f"  P(fall below 100% low dep):      {b['p_below_ld']:.1%}")
    print(f"  Worst-case (5th pct) LD funding: {b['worst_5pct_funding']:.1%}")

    print("\n" + "=" * 64)
    print("RISK/RETURN FRONTIER (growth-asset allocation sweep)")
    print("=" * 64)
    fr = frontier()
    print(fr.round(3).to_string(index=False))
    fr.to_csv("../tables/endgame_frontier.csv", index=False)
    print("\n  saved tables/endgame_frontier.csv")
