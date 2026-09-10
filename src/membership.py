"""
membership.py  (Stage 1)
========================
Generate a synthetic, closed, mature DB scheme and freeze it to CSV.

~3,000 members: ~1,200 pensioners (ages 60-95, mean ~73) and ~1,800 deferreds
(ages 40-64, mean ~54). Overall ~60/40 male/female, pensioners skewing more male.
Pension amounts are lognormal with a long right tail. Each member's pension is
split into three tranches by accrual period (pre-97 / 1997-2005 / post-2005,
roughly 40/30/30 by value) because tranches increase differently.

Output: data/processed/membership.csv, one row PER MEMBER PER TRANCHE, frozen
with a fixed seed so the whole model is reproducible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATA_PROC, MEMBERSHIP, SEED, TRANCHES


def _truncated_normal(rng, mean, sd, lo, hi, n):
    """Sample n integers from a normal truncated to [lo, hi] (rejection)."""
    out = np.empty(0, dtype=int)
    while out.size < n:
        draw = rng.normal(mean, sd, size=(n - out.size) * 2)
        draw = np.round(draw).astype(int)
        draw = draw[(draw >= lo) & (draw <= hi)]
        out = np.concatenate([out, draw])
    return out[:n]


def generate_membership(seed: int = SEED) -> pd.DataFrame:
    """Build the member-tranche table (not yet written to disk)."""
    rng = np.random.default_rng(seed)
    cfg = MEMBERSHIP
    rows = []

    def _add_group(n, status, age_mean, age_sd, age_lo, age_hi,
                   male_share, pension_median):
        ages = _truncated_normal(rng, age_mean, age_sd, age_lo, age_hi, n)
        sexes = np.where(rng.random(n) < male_share, "M", "F")
        # lognormal pensions: median = exp(mu) -> mu = ln(median)
        mu = np.log(pension_median)
        pensions = rng.lognormal(mean=mu, sigma=cfg.pension_sigma, size=n)
        for i in range(n):
            rows.append(dict(status=status, sex=sexes[i], age=int(ages[i]),
                             pension_total=float(pensions[i])))

    _add_group(cfg.n_pensioners, "pensioner", cfg.pensioner_age_mean,
               cfg.pensioner_age_sd, cfg.pensioner_age_min, cfg.pensioner_age_max,
               cfg.male_share_pensioner, cfg.pension_median_pensioner)
    _add_group(cfg.n_deferreds, "deferred", cfg.deferred_age_mean,
               cfg.deferred_age_sd, cfg.deferred_age_min, cfg.deferred_age_max,
               cfg.male_share_deferred, cfg.pension_median_deferred)

    members = pd.DataFrame(rows)
    members.insert(0, "member_id", np.arange(1, len(members) + 1))

    # Split each member's pension into tranches by value, then explode to rows.
    split = np.array(cfg.tranche_split)
    split = split / split.sum()
    trecs = []
    tranche_keys = list(TRANCHES.keys())
    for _, m in members.iterrows():
        for key, w in zip(tranche_keys, split):
            trecs.append(dict(
                member_id=int(m.member_id),
                status=m.status,
                sex=m.sex,
                age=int(m.age),
                tranche=key,
                tranche_label=TRANCHES[key]["label"],
                tranche_pension=float(m.pension_total * w),
            ))
    df = pd.DataFrame(trecs)
    return df


def save_membership(df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = generate_membership() if df is None else df
    path = DATA_PROC / "membership.csv"
    df.to_csv(path, index=False)
    return df


def load_membership() -> pd.DataFrame:
    path = DATA_PROC / "membership.csv"
    if not path.exists():
        return save_membership()
    return pd.read_csv(path)


def summarise(df: pd.DataFrame) -> str:
    """Human-readable summary used for the Stage 1 sanity check."""
    per_member = df.groupby(["member_id", "status", "sex", "age"], as_index=False)[
        "tranche_pension"].sum().rename(columns={"tranche_pension": "pension"})
    lines = []
    lines.append(f"Members:   {per_member.member_id.nunique():,}")
    lines.append(f"Rows:      {len(df):,} (member x tranche)")
    for status in ["pensioner", "deferred"]:
        sub = per_member[per_member.status == status]
        lines.append(
            f"  {status:<10} n={len(sub):>5}  "
            f"mean age={sub.age.mean():4.1f}  "
            f"male%={100*(sub.sex=='M').mean():4.1f}  "
            f"median pension=£{sub.pension.median():,.0f}  "
            f"mean pension=£{sub.pension.mean():,.0f}")
    overall_male = 100 * (per_member.sex == "M").mean()
    lines.append(f"Overall male share: {overall_male:.1f}%")
    yr1 = per_member[per_member.status == "pensioner"].pension.sum()
    lines.append(f"Year-1 pensioner outgo (pre-increase): £{yr1/1e6:,.1f}m")
    by_tranche = df.groupby("tranche_label")["tranche_pension"].sum()
    tot = by_tranche.sum()
    lines.append("Tranche split by value: " +
                 ", ".join(f"{k} {100*v/tot:.0f}%" for k, v in by_tranche.items()))
    return "\n".join(lines)


if __name__ == "__main__":
    df = save_membership()
    print("Saved:", DATA_PROC / "membership.csv")
    print("-" * 60)
    print(summarise(df))
