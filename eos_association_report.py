"""Does a token's projection on the value axis predict P(end-of-turn) there?

Input: results/eos_association.npz (modal_app.py::eos_association_main).

The steering result is causal but blunt: it pushes the residual stream far
off-distribution, and the seven-seed random band in steering_logits_report.py
shows random directions of the same norm move the end-of-turn logit almost as
much. This is the observational counterpart, on natural text, with no
intervention at all. For every assistant token:

    proj = cos(h_t at layer 21, unit axis)
    eos  = log P(<|im_end|> | tokens up to t)

THE CONFOUND IS POSITION. Both quantities rise toward the end of a turn, so a raw
correlation over all tokens is close to meaningless -- it would be produced by any
direction that happens to track position. The load-bearing number is therefore
the association computed WITHIN relative-position bins: at a fixed distance from
the end of the turn, does a higher projection still mean a higher probability of
stopping?

Every statistic is also computed for the shipped axis and for 8 random unit
directions, so the axis can be read against the spread of what random directions
give rather than against zero.

Usage: python eos_association_report.py [--bins 5]
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=5)
    args = ap.parse_args()

    d = np.load(ROOT / "results" / "eos_association.npz", allow_pickle=True)
    proj, eos, rel = d["proj"], d["eos"], d["rel_pos"]
    names = [str(x) for x in d["dir_names"]]
    ci = int(names.index("corrected"))
    si = int(names.index("shipped"))
    ri = [i for i, n in enumerate(names) if n.startswith("random")]

    print(f"{len(eos):,} assistant tokens over "
          f"{len(np.unique(d['conv_id']))} conversations, layer 21")
    print(f"log P(end-of-turn): mean {eos.mean():+.2f}, "
          f"range [{eos.min():+.1f}, {eos.max():+.1f}] (natural log)\n")

    def rho(idx, m=None):
        m = np.ones(len(eos), bool) if m is None else m
        if m.sum() < 30:
            return float("nan")
        r, _ = spearmanr(proj[m, idx], eos[m])
        return r

    # ---- raw association (confounded by position; reported to show the gap) ---
    print("=" * 74)
    print("RAW association over all tokens  --  CONFOUNDED BY POSITION")
    print("=" * 74)
    rr = np.array([rho(i) for i in ri])
    print(f"  corrected axis   Spearman rho = {rho(ci):+.3f}")
    print(f"  shipped axis                   {rho(si):+.3f}")
    print(f"  random x{len(ri)}      mean {rr.mean():+.3f}  sd {rr.std(ddof=1):.3f}  "
          f"range [{rr.min():+.3f}, {rr.max():+.3f}]")

    # ---- position-controlled -------------------------------------------------
    edges = np.linspace(0, 1, args.bins + 1)
    print("\n" + "=" * 74)
    print("WITHIN relative-position bins  --  the load-bearing number")
    print("=" * 74)
    print(f"  {'position in turn':<18} {'n':>8} {'corrected':>11} {'shipped':>9} "
          f"{'random mean':>12} {'random sd':>10} {'sd out':>7}")
    per_bin = []
    for a, b in zip(edges, edges[1:]):
        m = (rel >= a) & (rel < b if b < 1 else rel <= b)
        rc = rho(ci, m)
        rs = rho(si, m)
        rb = np.array([rho(i, m) for i in ri])
        k = (rc - rb.mean()) / rb.std(ddof=1) if np.isfinite(rc) else float("nan")
        per_bin.append((a, b, m.sum(), rc, rs, rb))
        print(f"  {f'{a:.1f}-{b:.1f}':<18} {m.sum():>8,} {rc:>+11.3f} {rs:>+9.3f} "
              f"{rb.mean():>+12.3f} {rb.std(ddof=1):>10.3f} {k:>+7.1f}")

    allrc = np.array([p[3] for p in per_bin])
    allrb = np.array([p[5].mean() for p in per_bin])
    print(f"\n  mean over bins: corrected {allrc.mean():+.3f}  "
          f"random {allrb.mean():+.3f}")
    print("\n  A positive corrected column that the random band does not reach is")
    print("  the observational form of the closure claim: at a FIXED distance")
    print("  from the end of the turn, tokens that project higher on the axis are")
    print("  tokens where the model is more likely to stop.")

    # ---- binned means, the shape behind the correlation -----------------------
    print("\n" + "=" * 74)
    print("MEAN log P(end-of-turn) BY PROJECTION DECILE (position-matched)")
    print("  Tokens are z-scored within their position bin first, so the deciles")
    print("  compare like with like rather than early tokens against late ones.")
    print("=" * 74)
    z = np.full(len(eos), np.nan)
    ez = np.full(len(eos), np.nan)
    for a, b in zip(edges, edges[1:]):
        m = (rel >= a) & (rel < b if b < 1 else rel <= b)
        if m.sum() < 30:
            continue
        p_, e_ = proj[m, ci], eos[m]
        z[m] = (p_ - p_.mean()) / (p_.std() + 1e-9)
        ez[m] = (e_ - e_.mean()) / (e_.std() + 1e-9)
    ok = np.isfinite(z)
    qs = np.quantile(z[ok], np.linspace(0, 1, 11))
    print(f"  {'decile of projection':<24} {'n':>8} {'mean z(log P(end))':>20}")
    for i in range(10):
        m = ok & (z >= qs[i]) & (z <= qs[i + 1] if i == 9 else z < qs[i + 1])
        print(f"  {f'{i+1}':<24} {m.sum():>8,} {ez[m].mean():>+20.3f}")


if __name__ == "__main__":
    main()
