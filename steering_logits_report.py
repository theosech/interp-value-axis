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

RANDOM unit directions are steered at the same alphas as a control. Pushing the
residual stream by |alpha|=75 in ANY direction takes it off-distribution, and an
off-distribution state degrades every readout. A channel that moves under the
value axis AND under the random directions is measuring steering damage; only a
channel that separates the two is measuring the axis's content. Read the random
column first.

SEVERAL random directions, not one. Hidden states have a large mean component, so
a single fixed random vector acquires an effective sign from its chance projection
onto that mean -- +d and -d are not equivalent perturbations, and its LINEAR
coefficient is chance-signed. The random arm is therefore reported as a BAND
(mean +/- sd over seeds, and the extreme), and the axis has to sit outside it.
Extra seeds live in results/steering_logits_randband.jsonl and are merged if
present.

Predictions, all on the LINEAR term b (not Spearman -- see below):
  confidence-encoding  yes_minus_no / p_yes / exp_rating have a b outside the
                       random band
  closure controller   closure has a b outside the random band; the confidence
                       channels do not

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
AXES = ("shipped", "corrected")


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


def quad(rows, dname, key, probe):
    """Per-prefix least-squares fit of value ~ a + b*alpha + c*alpha^2.

    A Spearman on a U-shaped profile reports whichever arm rises further, so it
    cannot tell a genuine dose-response from symmetric off-distribution damage.
    Splitting the profile does: damage is a negative quadratic with b ~ 0, a
    real directional effect is a nonzero b. Alpha is scaled to [-1, 1] so b and
    c are in readout units across the steering range, not per unit alpha.
    """
    b_, c_ = [], []
    for k in {(r["conv_id"], r["state"]) for r in rows}:
        pts = [(r["alpha"] / 75.0, r[key]) for r in rows
               if (r["conv_id"], r["state"]) == k and r["probe"] == probe
               and (r["direction"] == dname or r["alpha"] == 0)
               and key in r and np.isfinite(r[key])]
        if len(pts) < 5:
            continue
        x, v = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        coef = np.polyfit(x, v, 2)          # [c, b, a]
        c_.append(coef[0]); b_.append(coef[1])
    return np.array(b_), np.array(c_)


def _t(a):
    if len(a) < 3:
        return float("nan")
    se = a.std(ddof=1) / np.sqrt(len(a))
    return a.mean() / se if se > 0 else float("nan")


def rand_dirs(rows):
    return sorted({r["direction"] for r in rows if r["direction"].startswith("random")})


def main():
    rows = [json.loads(l) for l in open(ROOT / "results" / "steering_logits.jsonl")]
    band = ROOT / "results" / "steering_logits_randband.jsonl"
    if band.exists():
        extra = [json.loads(l) for l in open(band)]
        seen = {(r["conv_id"], r["state"], r["probe"], r["direction"], r["alpha"])
                for r in rows}
        rows += [r for r in extra
                 if (r["conv_id"], r["state"], r["probe"], r["direction"],
                     r["alpha"]) not in seen]
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

    for dname in AXES + tuple(rand_dirs(rows)[:1]):
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
    print("LINEAR vs QUADRATIC (per-prefix fit, alpha scaled to [-1,1])")
    print("  b = directional dose-response;  c = symmetric curvature.")
    print("  Off-distribution damage looks like b~0 with c<0 on EVERY direction,")
    print("  including random. A real effect of the axis is a b that the random")
    print("  control does not have.")
    print("=" * 78)
    rds = rand_dirs(rows)
    print(f"  {'channel':<24} {'direction':<14} {'b':>9} {'t(b)':>8} "
          f"{'c':>9} {'t(c)':>8}")
    for key, probe, label in CHANNELS:
        for dname in AXES:
            b, c = quad(rows, dname, key, probe)
            if len(b) < 3:
                continue
            print(f"  {label:<24} {dname:<14} {b.mean():>+9.3f} {_t(b):>+8.1f} "
                  f"{c.mean():>+9.3f} {_t(c):>+8.1f}")
        # random arm as a band over seeds, not a point estimate
        bs = [quad(rows, d, key, probe)[0].mean() for d in rds]
        cs = [quad(rows, d, key, probe)[1].mean() for d in rds]
        bs, cs = np.array(bs), np.array(cs)
        sd = bs.std(ddof=1) if len(bs) > 1 else float("nan")
        print(f"  {'':<24} {'random x' + str(len(rds)):<14} {bs.mean():>+9.3f} "
              f"{'sd ' + (f'{sd:.3f}' if len(bs) > 1 else 'n/a'):>8} "
              f"{cs.mean():>+9.3f}")
        if len(bs) > 1:
            k = (quad(rows, "corrected", key, probe)[0].mean() - bs.mean()) / sd
            print(f"  {'':<24} {'':<14} random range [{bs.min():+.3f}, {bs.max():+.3f}]"
                  f"   corrected is {k:+.1f} sd outside the random mean")
        print()

    print("=" * 78)
    print("READING IT")
    print("=" * 78)
    print("  Length cannot explain anything here: nothing was generated.")
    print("  Compare each channel's LINEAR term b against the random band for the")
    print("  same channel. A b inside the band is off-distribution damage from a")
    print("  large-norm intervention, not the content of the value axis.")
    print("  The quadratic term c is the damage itself and is large under every")
    print("  direction -- which is why Spearman on these profiles misleads: a")
    print("  U-shape symmetric about the unsteered value reports as a slope")
    print("  whenever one arm rises further than the other.")


if __name__ == "__main__":
    main()
