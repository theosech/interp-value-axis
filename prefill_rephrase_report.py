"""Rephrase-robustness report for the continuation-prefill experiment.

Input: results/prefill_rephrase.npz. Conditions: {arm}_v{1,2,3}_{t1,t2}.

The claim under test: on the byte-identical tail, complete_nosucc (give-up)
projects ABOVE correct_nocomplete (found-it-more-coming) -- the completion
reading. Robust if it holds for every phrasing pair and both tails.

Units: cosine (paper Eq. 2), corrected axis. Anchors: full before/after
dynamic range 0.165 @L21; random-direction null <=0.001; original single-
phrasing tail effect -0.0087 (65/65 conversations).

Usage: python prefill_rephrase_report.py [--layer 21]
"""

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=21)
    args = ap.parse_args()
    L = args.layer

    d = np.load(ROOT / "results" / "prefill_rephrase.npz", allow_pickle=True)
    cos = d["cos"][:, L]
    cond, seg, base = d["cond"], d["segment"], d["base_conv"]

    def cell(arm, v, t, s):
        m = (cond == f"{arm}_v{v}_{t}") & (seg == s)
        acc = {}
        for c, x in zip(base[m], cos[m]):
            acc.setdefault(c, []).append(x)
        return {c: float(np.mean(x)) for c, x in acc.items()}

    print(f"\nTAIL (matched tokens) paired diff: correct_nocomplete - complete_nosucc")
    print(f"corrected axis @L{L}; negative = completion reading wins")
    print(f"(anchors: dynamic range 0.165, random null <=0.001, original -0.0087)\n")
    for t in ("t1", "t2"):
        print(f"  tail {t}:")
        print(f"    {'':>12} " + " ".join(f"{'give-up v'+str(j):>16}" for j in (1, 2, 3)))
        for i in (1, 2, 3):
            row = f"    {'found-it v' + str(i):>12} "
            for j in (1, 2, 3):
                a = cell("correct_nocomplete", i, t, "tail")
                b = cell("complete_nosucc", j, t, "tail")
                common = sorted(set(a) & set(b))
                diff = np.array([a[c] - b[c] for c in common])
                se = diff.std(ddof=1) / np.sqrt(len(diff))
                row += f" {diff.mean():+.4f}({np.mean(diff < 0):3.0%})"
            print(row)
        print()

    # pooled per phrasing pair (both tails), sign consistency
    print("  pooled across tails, diagonal pairs:")
    for v in (1, 2, 3):
        diffs = []
        for t in ("t1", "t2"):
            a = cell("correct_nocomplete", v, t, "tail")
            b = cell("complete_nosucc", v, t, "tail")
            for c in set(a) & set(b):
                diffs.append(a[c] - b[c])
        diffs = np.array(diffs)
        se = diffs.std(ddof=1) / np.sqrt(len(diffs))
        print(f"    v{v}: d={diffs.mean():+.4f}  t={diffs.mean()/se:+.1f}  "
              f"completion-signed {np.mean(diffs < 0):.1%} of {len(diffs)}")

    # thinking/body for reference (lexically confounded)
    print("\n  thinking / body segments (condition-different text, for reference):")
    for s in ("thinking", "body"):
        diffs = []
        for v in (1, 2, 3):
            for t in ("t1", "t2"):
                a = cell("correct_nocomplete", v, t, s)
                b = cell("complete_nosucc", v, t, s)
                for c in set(a) & set(b):
                    diffs.append(a[c] - b[c])
        diffs = np.array(diffs)
        print(f"    {s:<9} mean diff {diffs.mean():+.4f}, "
              f"completion-signed {np.mean(diffs < 0):.1%}")


if __name__ == "__main__":
    main()
