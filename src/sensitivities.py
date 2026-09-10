
from __future__ import annotations

import numpy as np
import pandas as pd

from config import BASIS, MORTALITY
from valuation import (present_value, tp_spread, ld_spread, buyout_spread,
                       cashflows_best_estimate, liability_on_basis)
from cashflows import project_cashflows


def mortality_improvement_sensitivity():
    rows = []
    for lt in (MORTALITY.lt_improvement_low, MORTALITY.lt_improvement,
               MORTALITY.lt_improvement_high):
        cf = project_cashflows(lt_rate=lt)["total"]
        tp = present_value(cf, tp_spread)
        ld = present_value(cf, ld_spread)
        rows.append(dict(lt_improvement=lt, tp_m=tp/1e6, ld_m=ld/1e6))
    df = pd.DataFrame(rows)
    base_tp = df.loc[df.lt_improvement == MORTALITY.lt_improvement, "tp_m"].iloc[0]
    df["tp_vs_base_%"] = 100 * (df.tp_m / base_tp - 1.0)
    return df


def buyout_loading_sensitivity():
    scaling = MORTALITY.scheme_scaling * BASIS.buyout_mortality_scaling
    cf_bo = project_cashflows(lt_rate=MORTALITY.lt_improvement, scaling=scaling)["total"]
    bo_raw = present_value(cf_bo, buyout_spread)
    ld = liability_on_basis("ld")
    rows = []
    for load in (0.00, 0.02, 0.03, 0.04, 0.06):
        bo = bo_raw * (1 + load)
        rows.append({"loading": load, "buyout_m": bo/1e6, "vs_ld_%": 100*(bo/ld - 1)})
    return pd.DataFrame(rows)


def discount_sensitivity():
    rows = []
    for shift in (-0.005, 0.0, +0.005, +0.01):
        rows.append(dict(rate_shift=shift,
                         tp_m=liability_on_basis("tp", level_shift=shift)/1e6,
                         ld_m=liability_on_basis("ld", level_shift=shift)/1e6,
                         buyout_m=liability_on_basis("buyout", level_shift=shift)/1e6))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from config import OUT_TABLES
    mi = mortality_improvement_sensitivity()
    bl = buyout_loading_sensitivity()
    ds = discount_sensitivity()
    print("Mortality improvement sensitivity:")
    print(mi.round(2).to_string(index=False))
    print("\nBuy-out loading sensitivity:")
    print(bl.round(2).to_string(index=False))
    print("\nDiscount-rate sensitivity:")
    print(ds.round(2).to_string(index=False))
    with pd.ExcelWriter(OUT_TABLES / "sensitivities.xlsx") as xw:
        mi.to_excel(xw, sheet_name="mortality_improvement", index=False)
        bl.to_excel(xw, sheet_name="buyout_loading", index=False)
        ds.to_excel(xw, sheet_name="discount_rate", index=False)
    mi.to_csv(OUT_TABLES / "sens_mortality.csv", index=False)
    bl.to_csv(OUT_TABLES / "sens_buyout_loading.csv", index=False)
    ds.to_csv(OUT_TABLES / "sens_discount.csv", index=False)
    print("\nsaved outputs/tables/sensitivities.xlsx (+ csvs)")
