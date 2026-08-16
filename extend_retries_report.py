"""Extended-retry experiment: what does the retry-depth climb actually track?

Input: results/extended_retries.npz (extend_retries_build.py ->
       modal run modal_app.py::extended_retries_main)

In the released corpus the projection at the assistant header climbs with retry
depth within a paragraph. Depth only runs to ~5 there, which is too short to
tell the candidate explanations apart. This experiment splices scripted failure
sequences out of the shared paragraph pool and runs depth to 20, in two arms:

  diverse    attempts 6-20 are 15 distinct rewrites
  duplicate  attempts 6-20 are verbatim cycles of the first 5

Measured at the 7-token assistant header, which is byte-identical everywhere, so
token identity is controlled by construction. `prev` records what preceded the
header ("minus1" after a -1, "paragraph" after a Paragraph message), because
left context shifts the level by ~0.05 and must not be mixed.

What each hypothesis predicts for the climb:

  value / solvability inference   should REVERSE once enough failures pile up:
                                  evidence accumulates that this paragraph is
                                  unsolvable, so expected value falls.
  information gain                the duplicate arm delivers zero new
                                  information after attempt 5, so its climb
                                  should FLATTEN relative to diverse.
  bare attempt counter            the climb should not care what the prompt
                                  says the attempt ceiling is.
  closure / settledness           climb, then saturate; no reversal; duplicate
                                  at least as high as diverse.

Units: cosine (paper Eq. 2), corrected axis, layer 21. Anchors: full before/
after dynamic range 0.165; random-direction null <=0.001; within-cell sd ~0.015.

Usage: python extend_retries_report.py [--layer 21]
"""

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def paired_by_paragraph(d, cos, mask_a, mask_b):
    """Mean difference b - a, paired within paragraph_id."""
    pid = d["paragraph_id"]
    acc_a, acc_b = {}, {}
    for m, acc in ((mask_a, acc_a), (mask_b, acc_b)):
        for p, v in zip(pid[m], cos[m]):
            acc.setdefault(p, []).append(v)
    common = sorted(set(acc_a) & set(acc_b))
    if len(common) < 3:
        return float("nan"), float("nan"), len(common)
    diff = np.array([np.mean(acc_b[p]) - np.mean(acc_a[p]) for p in common])
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    return diff.mean(), diff.mean() / se if se > 0 else float("nan"), len(diff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=21)
    args = ap.parse_args()
    L = args.layer

    d = np.load(ROOT / "results" / "extended_retries.npz", allow_pickle=True)
    cos = d["hdr_cos"][:, L]
    ship = d["hdr_shipped"][:, L]
    depth, arm, prev = d["depth"], d["arm"], d["prev"]
    rnd = d["hdr_random21"].mean(axis=1)

    # Only headers that follow a "-1" are retry headers; the rest are
    # paragraph-initial and belong to a different cell.
    retry = prev == "minus1"

    print(f"{len(depth)} headers, {len(set(d['paragraph_id']))} paragraphs, "
          f"2 arms, depths {depth.min()}-{depth.max()}, layer {L}")
    print(f"random-direction control mean {np.nanmean(rnd):+.4f}  "
          f"(anchors: dynamic range 0.165, within-cell sd ~0.015)\n")

    print("=" * 74)
    print("CLIMB BY DEPTH (retry headers only, i.e. those following a -1)")
    print("=" * 74)
    print(f"  {'depth':>6} {'diverse':>10} {'duplicate':>11} {'both':>9} "
          f"{'shipped':>9} {'n':>5}")
    for dep in sorted(set(depth[retry].tolist())):
        m = retry & (depth == dep)
        md = m & (arm == "diverse")
        mu = m & (arm == "duplicate")
        print(f"  {dep:>6} {np.mean(cos[md]) if md.any() else np.nan:>10.4f} "
              f"{np.mean(cos[mu]) if mu.any() else np.nan:>11.4f} "
              f"{np.mean(cos[m]):>9.4f} {np.mean(ship[m]):>9.4f} {m.sum():>5}")

    print("\n" + "=" * 74)
    print("TEST 1 - does the climb REVERSE?  (value / solvability inference)")
    print("=" * 74)
    bands = [(2, 5), (6, 10), (11, 15), (16, 20)]
    lvl = {}
    for lo, hi in bands:
        m = retry & (depth >= lo) & (depth <= hi)
        lvl[(lo, hi)] = np.mean(cos[m])
        print(f"  depth {lo:>2}-{hi:<2}  mean {np.mean(cos[m]):>+8.4f}  n={m.sum()}")
    for (a, b) in zip(bands, bands[1:]):
        ma = retry & (depth >= a[0]) & (depth <= a[1])
        mb = retry & (depth >= b[0]) & (depth <= b[1])
        mu, t, n = paired_by_paragraph(d, cos, ma, mb)
        print(f"  step {a[0]}-{a[1]} -> {b[0]}-{b[1]}: d={mu:>+8.4f}  "
              f"t={t:>+7.1f}  n={n} paragraphs")
    print("\n  A value/solvability reading needs a NEGATIVE step late. Saturation")
    print("  toward zero with no sign change is the closure/settledness pattern.")

    print("\n" + "=" * 74)
    print("TEST 2 - does zero new information flatten the climb?  (info gain)")
    print("=" * 74)
    print("  Attempts 6-20 of the duplicate arm are verbatim repeats: no new")
    print("  information about the criterion is available after attempt 5.")
    for lo, hi in bands[1:]:
        ma = retry & (depth >= lo) & (depth <= hi) & (arm == "duplicate")
        mb = retry & (depth >= lo) & (depth <= hi) & (arm == "diverse")
        mu, t, n = paired_by_paragraph(d, cos, ma, mb)
        print(f"  depth {lo:>2}-{hi:<2}  diverse - duplicate = {mu:>+8.4f}  "
              f"t={t:>+7.1f}  n={n} paragraphs")
    print("\n  Information gain predicts diverse > duplicate (positive). A")
    print("  negative sign means the zero-information arm sits HIGHER, which no")
    print("  information-accumulation account produces.")


if __name__ == "__main__":
    main()
