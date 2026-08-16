"""Length-free confidence readout under value-axis steering.

Input: results/steering_logits.jsonl (modal_app.py::steering_logits_main).

`steering_probe_report.py` shows that steering moves response LENGTH (per-prefix
Spearman -0.96 on the corrected axis) and not the verbalized confidence RATING
(t~0). The objection to that result is that the rating was read out of generated
text, so a wrap-up push could truncate the response before the rating is settled,
or bias the parse. This report removes generation from the readout: every number
below comes from ONE forward pass, read at the single position where the answer
token would go.

Four channels, per (prefix, direction, alpha):
  yes_minus_no  logit(Yes) - logit(No) on a forced one-word probe. Signed,
                graded, length-free. THE confidence readout.
  exp_rating    expected value of the 0-9 digit distribution at the same
                position, renormalized over the digit tokens. The length-free
                analogue of the parsed rating in steering_probe. ("10" is two
                tokens, so a 0-10 probe would merge rating 10 into rating 1.)
  p_yes         P(Yes)/(P(Yes)+P(No)) at the same position. Bounded, so an
                overall change in logit scale cannot drag it.
  closure       log P(<|im_end|>), same position. The channel a closure
                controller should move.

A RANDOM unit direction is steered at the same alphas as a control. Pushing the
residual stream by |alpha|=75 in ANY direction takes it off-distribution, and an
off-distribution state degrades every readout. A channel that moves under the
value axis AND under the random direction is measuring steering damage; only a
channel that separates the two is measuring the axis's content. Read the random
column first.

Predictions:
  confidence-encoding  yes_minus_no / p_yes / exp_rating rise with alpha, and
                       do NOT do so under the random control
  closure controller   closure rises with alpha beyond the random control; the
                       confidence channels do not separate from it

Dose-response is a per-prefix Spearman across alphas, then a t-test of those
per-prefix rhos against zero, matching steering_probe_report.py.

Usage: python steering_logits_report.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent

CHANNELS = [
    ("yes_minus_no", "binary", "logit(Yes)-logit(No)"),
    ("p_yes", "binary", "P(Yes|Yes or No)"),
    ("exp_rating", "digit", "E[rating] over 0-9"),
    ("closure", "binary", "log P(im_end)"),
]
DIRECTIONS = ("shipped", "corrected", "random")


def slopes(rows, dname, key, probe):
    """Per-prefix Spearman(value, alpha); alpha=0 is shared across directions."""
    out = []
    keys = {(r["conv_id"], r["state"]) for r in rows}
    for k in keys:
        pts = [(r["alpha"], r[key]) for r in rows
               if (r["conv_id"], r["state"]) == k and r["probe"] == probe
               and (r["direction"] == dname or r["alpha"] == 0)
               and key in r and np.isfinite(r[key])]
        if len(pts) >= 5 and len({p[0] for p in pts}) >= 4:
            rho, _ = spearmanr(*zip(*pts))
            if np.isfinite(rho):
                out.append(rho)
    return np.array(out)


def main():
    path = ROOT / "results" / "steering_logits.jsonl"
    rows = [json.loads(l) for l in open(path)]
    alphas = sorted({r["alpha"] for r in rows})
    prefixes = {(r["conv_id"], r["state"]) for r in rows}
    print(f"{len(rows)} forward passes, {len(prefixes)} prefixes "
          f"({len(prefixes)//2} conversations), alphas {alphas}")
    print("No text is generated: every value is read at the single answer "
          "position.\n")

    # Unsteered sanity check: does the readout track true state at alpha=0?
    print("=" * 78)
    print("SANITY (alpha=0, unsteered): does the length-free readout track state?")
    print("=" * 78)
    for key, probe, label in (CHANNELS[0], CHANNELS[2]):
        for state in ("early", "post"):
            v = [r[key] for r in rows if r["alpha"] == 0 and r["state"] == state
                 and r["probe"] == probe and key in r]
            print(f"  {label:<24} {state:<6} {np.mean(v):>+8.3f}  (n={len(v)})")
    print("  early = rule unknown, post = rule known and confirmed.")
    print("  If these separate, the readout is a working confidence measure.")

    for dname in DIRECTIONS:
        print("\n" + "=" * 78)
        print(f"DIRECTION: {dname}   (alpha=0 shared between directions)")
        print("=" * 78)
        for state in ("early", "post"):
            print(f"  state={state}")
            print(f"  {'alpha':>6} {'Yes-No':>9} {'P(Yes)':>8} {'E[rating]':>11} "
                  f"{'digitmass':>10} {'logP(end)':>10} {'n':>5}")
            for a in alphas:
                sel = [r for r in rows if r["state"] == state and r["alpha"] == a
                       and (r["direction"] == dname or a == 0)]
                if not sel:
                    continue
                g = lambda k: [r[k] for r in sel if k in r]
                print(f"  {a:>+6} {np.mean(g('yes_minus_no')):>9.3f} "
                      f"{np.mean(g('p_yes')):>8.3f} {np.mean(g('exp_rating')):>11.3f} "
                      f"{np.mean(g('digit_mass')):>10.3f} "
                      f"{np.mean(g('closure')):>10.3f} {len(sel):>5}")
            print()

        print("  dose-response (mean per-prefix Spearman rho vs alpha):")
        for key, probe, label in CHANNELS:
            s = slopes(rows, dname, key, probe)
            if len(s) < 3:
                continue
            se = s.std(ddof=1) / np.sqrt(len(s))
            t = s.mean() / se if se > 0 else float("nan")
            print(f"    {label:<24} [{probe:<6}] rho={s.mean():>+6.3f}  "
                  f"t={t:>+7.1f}  n={len(s)}")

    print("\n" + "=" * 78)
    print("READING IT")
    print("=" * 78)
    print("  Length cannot explain anything here: nothing was generated.")
    print("  Compare each channel's rho against the SAME channel's rho under the")
    print("  random direction. A channel whose rho is matched by the random")
    print("  control is reporting off-distribution damage from a large-norm")
    print("  intervention, not the content of the value axis. Also check whether")
    print("  each column is MONOTONE in alpha -- a U-shape symmetric about the")
    print("  unsteered value is the damage signature, and Spearman will report it")
    print("  as a slope if one arm rises more than the other.")


if __name__ == "__main__":
    main()
