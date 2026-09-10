
from __future__ import annotations

import time
import numpy as np
import pandas as pd

from config import OUT_TABLES, N_SIMS, SIM_PROJECTION_YEARS


def main():
    t0 = time.time()
    print("=" * 66)
    print("DB SCHEME FUNDING & ENDGAME MODEL - full pipeline")
    print("=" * 66)

    # Stage 1 ---------------------------------------------------------------
    from membership import save_membership, summarise
    df = save_membership()
    print("\n[Stage 1] Synthetic scheme")
    print(summarise(df))

    # Stage 3/4 -------------------------------------------------------------
    from valuation import value_all_bases
    liab = value_all_bases()
    print("\n[Stage 4] Liability on three bases")
    print(liab[["basis", "liability_m", "duration", "vs_low_dep_%"]].round(2).to_string(index=False))
    liab.to_csv(OUT_TABLES / "liability_three_bases.csv", index=False)

    # Stage 5 ---------------------------------------------------------------
    from projection import run_projection
    res = run_projection(release_surplus=False)
    print("\n[Stage 5] Stochastic funding at year", SIM_PROJECTION_YEARS)
    for nm, arr in [("TP", res["fund_tp"]), ("LD", res["fund_ld"]), ("Buy-out", res["fund_bo"])]:
        f = arr[:, -1]
        print(f"  {nm:<8} median {np.median(f):.0%}  5th {np.percentile(f,5):.0%}  95th {np.percentile(f,95):.0%}")

    # Stage 6 ---------------------------------------------------------------
    from endgame import time_to_buyout, run_on, frontier
    a = time_to_buyout(res)
    b = run_on()
    fr = frontier()
    print("\n[Stage 6] Endgame")
    print(f"  Route A - median time to buy-out {a['median']:.0f}y "
          f"(25th/75th {a['p25']:.0f}/{a['p75']:.0f}y); "
          f"{a['p_within_10']:.0%} within 10y; {a['p_never_20']:.0%} never in 20y")
    print(f"  Route B - expected surplus £{b['expected_surplus']/1e6:.0f}m; "
          f"P(<95% LD) at 25% growth = {fr.loc[fr.growth_weight==0.25,'prob_below_95_ld'].iloc[0]:.0%}; "
          f"1-in-20 funding {fr.loc[fr.growth_weight==0.25,'worst_5pct_funding'].iloc[0]:.0%}")
    fr.to_csv(OUT_TABLES / "endgame_frontier.csv", index=False)

    # Charts ----------------------------------------------------------------
    from plotting import make_all
    print("\n[Charts]")
    make_all()

    # Stage 7 ---------------------------------------------------------------
    from validation import run_all as validate
    print("\n[Stage 7] Validation")
    validate()

    # Sensitivities ---------------------------------------------------------
    import sensitivities as S
    S.mortality_improvement_sensitivity().to_csv(OUT_TABLES / "sens_mortality.csv", index=False)
    S.buyout_loading_sensitivity().to_csv(OUT_TABLES / "sens_buyout_loading.csv", index=False)
    print("\n[Sensitivities] written to outputs/tables/")

    print(f"\nDONE in {time.time()-t0:.1f}s. Outputs in outputs/.")


if __name__ == "__main__":
    main()
