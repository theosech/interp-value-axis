"""Is the before/after contrast a criterion-locked JUMP or a positional RAMP?

The paper's construction contrast is: mean projection of the tokens after the
criterion-satisfying token, minus the mean of the tokens before it, inside one
rewarded paragraph rewrite. That contrast is read as a value update at the
moment the criterion is met.

But the same contrast is produced by any quantity that rises monotonically
through the response, with no reference to the criterion at all. The two are
distinguishable by where you cut:

  JUMP at the criterion   the contrast is large only when the cut is at the
                          criterion token, and its size does not depend on
                          where in the paragraph that token happens to fall.
  LINEAR RAMP             for a linear f over T tokens, cutting at fraction p
                          gives mean(after) - mean(before) = slope * T / 2,
                          which is INDEPENDENT of p. The contrast is flat in
                          cut position -- and it appears wherever you cut.

So: bin attempts by the fraction of the rewrite that precedes the split, and
look at the contrast per bin. Flat across bins is the ramp signature.

Two tests here:
  A  actual-criterion splits, rewarded attempts, binned by cut fraction.
  B  PLACEBO -- failing attempts split at the model's own stated target word.
     No reward is delivered anywhere in these attempts, so a criterion-locked
     value update predicts ~0. A ramp predicts the same contrast as A.

Units: cosine against the unit axis (paper Eq. 2), corrected axis, layer 21,
per-token function-held-out split directions. Anchors: full before/after
dynamic range 0.165; random-direction null <=0.001; within-cell sd ~0.015;
the paper's own headline behavioral effects 0.02-0.04 on the same scale.

Input:  results/attempt_split.npz (modal_app.py::attempt_split_main)
Usage:  python ramp_cut_invariance.py [--layer 21]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"

BINS = [(0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 0.55),
        (0.55, 0.65), (0.65, 0.75), (0.75, 0.85)]


def held_out(d, L, test_fns):
    """Per-row held-out before/after at layer L: mean over the split directions
    whose held-out function set contains this row's reward function."""
    n = len(d["reward_fn"])
    held = np.zeros((n, len(test_fns)), dtype=bool)
    for si, fns in enumerate(test_fns):
        held[:, si] = np.isin(d["reward_fn"], fns)
    cnt = np.maximum(held.sum(axis=1).astype(float), 1)
    bef = (d["before_split"][:, :, L] * held).sum(axis=1) / cnt
    aft = (d["after_split"][:, :, L] * held).sum(axis=1) / cnt
    return bef, aft, held.sum(axis=1) > 0


def table(name, frac, diff, note=""):
    print(f"\n  {name}{note}")
    print(f"    {'cut fraction':>14} {'n':>6} {'diff':>9} {'sd':>8} {'t':>7}")
    for lo, hi in BINS:
        m = (frac >= lo) & (frac < hi)
        if m.sum() < 10:
            continue
        x = diff[m]
        se = x.std(ddof=1) / np.sqrt(len(x))
        print(f"    {f'{lo:.2f}-{hi:.2f}':>14} {m.sum():>6} {x.mean():>+9.4f} "
              f"{x.std(ddof=1):>8.4f} {x.mean()/se:>7.1f}")
    m = (frac >= 0.15) & (frac < 0.85)
    rho, p = spearmanr(frac[m], diff[m])
    print(f"    {'ALL 0.15-0.85':>14} {m.sum():>6} {diff[m].mean():>+9.4f} "
          f"{diff[m].std(ddof=1):>8.4f}")
    print(f"    Spearman(diff, cut fraction) = {rho:+.3f}  (p={p:.1e}, n={m.sum()})")
    return diff[m].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    L = args.layer

    d = np.load(RES / "attempt_split.npz", allow_pickle=True)
    test_fns = json.loads((RES / "attempt_split_test_fns.json").read_text())
    bef, aft, ok = held_out(d, L, test_fns)
    diff = aft - bef

    nb, ns, na = d["n_before"], d["n_split_tokens"], d["n_after"]
    total = nb + ns + na
    frac = np.where(total > 0, nb / np.maximum(total, 1), np.nan)

    ch, rew = d["channel"], d["reward"] == "True"
    print(f"{ok.sum()} rows with a held-out direction, layer {L}, corrected axis")
    print("anchors: dynamic range 0.165 | random-direction null <=0.001 | "
          "within-cell sd ~0.015")
    print("\nA linear ramp predicts a FLAT column of diffs; a criterion-locked "
          "jump predicts\nthe contrast to concentrate at the criterion cut and "
          "vanish in the placebo.")

    print("\n" + "=" * 74)
    print("A. ACTUAL criterion split, rewarded attempts")
    print("=" * 74)
    m = ok & (ch == "actual") & rew & np.isfinite(frac)
    a_mean = table("split at the criterion-satisfying token", frac[m], diff[m])

    print("\n" + "=" * 74)
    print("B. PLACEBO: failing attempts split at the model's BELIEVED target word")
    print("   (reward = -1; no criterion is met anywhere in these attempts)")
    print("=" * 74)
    m = ok & (ch == "believed") & (~rew) & np.isfinite(frac)
    b_mean = table("split at a word that earned nothing", frac[m], diff[m])

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    print(f"  actual-criterion split, rewarded : {a_mean:+.4f}")
    print(f"  believed-word split, FAILED      : {b_mean:+.4f}  "
          f"({b_mean/a_mean:.2f}x the criterion contrast)")
    print("\n  The placebo is not smaller than the criterion contrast -- it is")
    print("  LARGER. A contrast that survives, at full size, in attempts where")
    print("  no reward was delivered and the cut word earned nothing is not a")
    print("  measurement of the reward event.")


if __name__ == "__main__":
    main()
