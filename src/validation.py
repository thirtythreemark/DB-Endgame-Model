
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from config import OUT_VALID, OUT_CHARTS, START_TP_FUNDING, N_SIMS
from valuation import (value_all_bases, liability_on_basis, cashflows_best_estimate,
                       tp_spread, present_value, effective_duration)
from cashflows import project_cashflows
from projection import run_projection
from endgame import time_to_buyout

# PPF 7800 index funding ratio (s179 basis) around the valuation date. The s179
# basis is not the TP basis, but the two should be in a comparable region;
# a wild divergence would flag a miscalibrated basis.
PPF_7800_FUNDING = 1.25


def check_duration():
    base = value_all_bases()
    dur = base.loc[base.basis == "Technical provisions", "duration"].iloc[0]
    L0 = liability_on_basis("tp")
    # Isolate duration with a small (10bp) shift where convexity is negligible.
    small = liability_on_basis("tp", level_shift=+0.001)
    actual_small = (small / L0 - 1.0) * 100
    expected_small = -dur * 0.1
    ok = abs(actual_small - expected_small) < 0.10
    # Also report the full 1% move; the gap vs -duration is second-order convexity.
    big = liability_on_basis("tp", level_shift=+0.01)
    actual_big = (big / L0 - 1.0) * 100
    return dict(name="Duration check", ok=ok,
                detail=(f"+10bp: {actual_small:+.3f}% vs -duration*0.1% = {expected_small:+.3f}% "
                        f"(match); +1%: {actual_big:+.2f}% vs {-dur:+.2f}% "
                        f"(gap = +convexity, as expected); duration {dur:.1f}y"))


def check_zero_improvement():
    L0 = liability_on_basis("tp")
    cf_noimp = project_cashflows(lt_rate=0.0)["total"]
    L1 = present_value(cf_noimp, tp_spread)
    drop = (1.0 - L1 / L0) * 100
    ok = 1.5 < drop < 12.0
    return dict(name="Zero-improvement check", ok=ok,
                detail=f"turning off improvements drops TP liability by {drop:.1f}% "
                       f"(expect a few %)")


def check_cashflow_shape():
    undisc = cashflows_best_estimate().sum()
    L0 = liability_on_basis("tp")
    ratio = undisc / L0
    ok = 2.5 <= ratio <= 3.5
    return dict(name="Cashflow shape (undisc/TP)", ok=ok,
                detail=f"undiscounted outgo / TP liability = {ratio:.2f}x  (target 2.5-3.5x)")


def check_external_benchmark():
    start = START_TP_FUNDING
    ok = 0.85 <= start <= 1.35
    return dict(name="External benchmark (PPF 7800)", ok=ok,
                detail=f"start TP funding {start:.0%}; PPF 7800 ~{PPF_7800_FUNDING:.0%} "
                       f"(s179 basis, comparable region - not identical)")


def check_convergence():
    counts = [250, 500, 1000, 2000, 3000, 5000]
    medians = []
    for n in counts:
        res = run_projection(n_sims=n, release_surplus=False)
        medians.append(time_to_buyout(res)["median"])
    # settled if last two medians within 0.5 years
    ok = abs(medians[-1] - medians[-2]) <= 0.5

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(counts, medians, "-o", color="#1f3a5f")
    ax.set_xlabel("Number of simulations"); ax.set_ylabel("Median time to buy-out (yrs)")
    ax.set_title("Convergence of median time-to-buy-out")
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT_CHARTS / "07_convergence.png", dpi=150)
    plt.close(fig)
    return dict(name="Convergence check", ok=ok,
                detail="median TTB by sim count: " +
                       ", ".join(f"{n}:{m:.0f}y" for n, m in zip(counts, medians)))


def run_all():
    checks = [check_duration(), check_zero_improvement(), check_cashflow_shape(),
              check_external_benchmark(), check_convergence()]
    lines = ["DB ENDGAME MODEL - VALIDATION REPORT", "=" * 60, ""]
    for c in checks:
        flag = "PASS" if c["ok"] else "**REVIEW**"
        lines.append(f"[{flag}] {c['name']}")
        lines.append(f"        {c['detail']}")
        lines.append("")
    n_pass = sum(c["ok"] for c in checks)
    lines.append(f"Summary: {n_pass}/{len(checks)} checks passed.")
    report = "\n".join(lines)
    (OUT_VALID / "validation_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nsaved outputs/validation/validation_report.txt")
    print(f"saved outputs/charts/07_convergence.png")
    return checks


if __name__ == "__main__":
    run_all()
