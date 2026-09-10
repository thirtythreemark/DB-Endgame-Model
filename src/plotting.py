
from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from config import OUT_CHARTS, SIM_PROJECTION_YEARS
import mortality as mort
from cashflows import project_cashflows, cashflow_frame
from valuation import value_all_bases
from projection import run_projection
from endgame import time_to_buyout, frontier


mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.titleweight": "bold", "axes.titlesize": 12.5,
    "figure.autolayout": True,
})
NAVY, TEAL, AMBER, RED, GREY = "#1f3a5f", "#2a9d8f", "#e9a23b", "#c1414c", "#8a8f98"


def _save(fig, name):
    path = OUT_CHARTS / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# 1 -------------------------------------------------------------------------
def chart_life_expectancy():
    fig, ax = plt.subplots(figsize=(7, 4.4))
    labels = ["Male", "Female"]
    now65 = [mort.life_expectancy_at(65, s, calendar_offset=0) for s in ("M", "F")]
    now45 = [mort.life_expectancy_at(65, s, calendar_offset=20) for s in ("M", "F")]
    x = np.arange(2)
    w = 0.36
    b1 = ax.bar(x - w/2, now65, w, label="Aged 65 now", color=NAVY)
    b2 = ax.bar(x + w/2, now45, w, label="Aged 45 now (at 65)", color=TEAL)
    for b in list(b1) + list(b2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.15,
                f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Cohort life expectancy at 65 (years)")
    ax.set_title("Longevity: a 45-year-old is expected to live longer at 65\nthan a 65-year-old today")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(now45) + 3)
    return _save(fig, "01_life_expectancy_at_65.png")


# 2 -------------------------------------------------------------------------
def chart_cashflows():
    res = project_cashflows()
    cf = cashflow_frame(res)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.fill_between(cf.year, 0, cf.own_pension/1e6, color=NAVY, alpha=0.85, label="Member pensions")
    ax.fill_between(cf.year, cf.own_pension/1e6, cf.total/1e6, color=AMBER, alpha=0.9, label="Contingent spouse")
    peak = int(cf.year[cf.total.idxmax()])
    ax.axvline(peak, color=GREY, ls="--", lw=1)
    ax.text(peak+1, cf.total.max()/1e6*0.9, f"peak ~yr {peak}", color=GREY)
    ax.set_xlabel("Projection year"); ax.set_ylabel("Expected benefit outgo (£m p.a.)")
    ax.set_title("Projected benefit outgo: rises as deferreds retire, then runs off")
    ax.legend(frameon=False); ax.set_xlim(0, 70)
    return _save(fig, "02_cashflow_projection.png")


def chart_liabilities():
    df = value_all_bases()
    fig, ax = plt.subplots(figsize=(7, 4.4))
    colors = [NAVY, TEAL, AMBER]
    bars = ax.bar(df.basis, df.liability/1e6, color=colors, width=0.6)
    for b, d in zip(bars, df.duration):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+2,
                f"£{b.get_height():.0f}m\ndur {d:.1f}y", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Liability value (£m)")
    ax.set_title("Liability on three bases at 31 Mar 2026")
    ax.set_ylim(0, df.liability.max()/1e6 * 1.18)
    return _save(fig, "03_liability_three_bases.png")

def chart_funding_fan(res=None):
    res = run_projection(release_surplus=False) if res is None else res
    fbo = res["fund_bo"] * 100
    yrs = np.arange(fbo.shape[1])
    pcts = {p: np.percentile(fbo, p, axis=0) for p in (5, 25, 50, 75, 95)}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.fill_between(yrs, pcts[5], pcts[95], color=NAVY, alpha=0.15, label="5th-95th pct")
    ax.fill_between(yrs, pcts[25], pcts[75], color=NAVY, alpha=0.30, label="25th-75th pct")
    ax.plot(yrs, pcts[50], color=NAVY, lw=2, label="Median")
    ax.axhline(100, color=RED, ls="--", lw=1.2, label="Buy-out target (100%)")
    ax.set_xlabel("Projection year"); ax.set_ylabel("Buy-out funding level (%)")
    ax.set_title("Buy-out funding cone (5,000 simulations)")
    ax.legend(frameon=False, ncol=2, fontsize=9); ax.set_xlim(0, SIM_PROJECTION_YEARS)
    return _save(fig, "04_funding_fan_chart.png")


def chart_time_to_buyout(a=None):
    a = time_to_buyout() if a is None else a
    first = a["first_year"]
    Y = SIM_PROJECTION_YEARS
    reached = first[first <= Y]
    never = (first > Y).sum()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bins = np.arange(0, Y+2) - 0.5
    ax.hist(reached, bins=bins, color=TEAL, alpha=0.9, edgecolor="white")
    ax.axvline(a["median"], color=NAVY, lw=2, label=f"Median {a['median']:.0f}y")
    ax.axvline(a["p75"], color=AMBER, lw=2, ls="--", label=f"75th pct {a['p75']:.0f}y")
    ax.set_xlabel("Year buy-out first affordable"); ax.set_ylabel("Number of simulations")
    ax.set_title(f"Time to buy-out distribution  ({a['p_never_20']:.0%} never within {Y}y)")
    ax.legend(frameon=False); ax.set_xlim(-0.5, Y+0.5)
    return _save(fig, "05_time_to_buyout_hist.png")


def chart_frontier(fr=None):
    fr = frontier() if fr is None else fr
    fig, ax = plt.subplots(figsize=(8, 5))
    x = fr.prob_below_95_ld * 100
    y = fr.expected_surplus_m
    ax.plot(x, y, "-o", color=NAVY, lw=1.5, zorder=2)
    for _, r in fr.iterrows():
        note = f"{r.growth_weight:.0%} growth\n1-in-20: {r.worst_5pct_funding:.0%}"
        ax.annotate(note, (r.prob_below_95_ld*100, r.expected_surplus_m),
                    textcoords="offset points", xytext=(9, -6), fontsize=8, color=GREY)
    ax.set_xlabel("Downside risk:  P(funding falls materially below low dependency, <95%)  (%)")
    ax.set_ylabel("Reward:  expected surplus released over 20y (£m)")
    ax.set_title("Endgame risk/return frontier\n(labels: growth allocation & 1-in-20 worst-case LD funding)")
    ax.margins(x=0.14)
    return _save(fig, "06_endgame_frontier.png")


def make_all():
    paths = []
    print("Building charts...")
    paths.append(chart_life_expectancy()); print("  1 life expectancy")
    paths.append(chart_cashflows()); print("  2 cashflows")
    paths.append(chart_liabilities()); print("  3 liabilities")
    res = run_projection(release_surplus=False)
    paths.append(chart_funding_fan(res)); print("  4 funding fan")
    paths.append(chart_time_to_buyout(time_to_buyout(res))); print("  5 time to buy-out")
    paths.append(chart_frontier()); print("  6 frontier")
    for p in paths:
        print("   ->", p.name)
    return paths


if __name__ == "__main__":
    make_all()
