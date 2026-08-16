"""Prefill-probe results: value vs completion vs predictability.

Input: results/prefill_probes.npz (modal_app.py::prefill_probes_main).

Predicted sign patterns (corrected axis):
                                 value    completion    predictability
  correct_nocomplete prefill     HIGH     LOW           HIGH
  complete_nosucc   prefill      LOW      HIGH          LOW
  done?    Yes - No              ~0       positive      ?
  correct? Yes - No              positive ~0            ?

All contrasts are paired by base conversation. Segments: thinking / body are
condition-specific text (lexical content differs -- state + content mixed);
tail is byte-identical across the two continuation arms (state only);
answer is "Yes."/"No." (content-matched up to the answer word itself).

Usage: python prefill_probes_report.py [--layer 21]
"""

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"


def paired(conv_a, v_a, conv_b, v_b):
    da, db = {}, {}
    for c, v in zip(conv_a, v_a):
        da.setdefault(c, []).append(v)
    for c, v in zip(conv_b, v_b):
        db.setdefault(c, []).append(v)
    common = sorted(set(da) & set(db))
    d = np.array([np.nanmean(db[c]) - np.nanmean(da[c]) for c in common])
    if len(d) < 3:
        return float("nan"), float("nan"), len(d)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() / se if se > 0 else float("nan"), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=21)
    args = ap.parse_args()
    L = args.layer

    d = np.load(RES / "prefill_probes.npz", allow_pickle=True)
    cos = d["cos"][:, L]
    fam, cond, phase, seg = d["family"], d["cond"], d["phase"], d["segment"]
    base = d["base_conv"]
    rnd = d["cos_random21"].mean(axis=1)

    print(f"\nlayer {L}, corrected axis; random-direction control mean "
          f"{np.nanmean(rnd):+.4f}\n")

    # ---- A: continuation prefills ------------------------------------------
    print("=" * 76)
    print("A. CONTINUATION PREFILLS (after 3 fails; paired by conversation)")
    print("=" * 76)
    print(f"  {'segment':<10} {'correct_nocomplete':>19} {'complete_nosucc':>17} "
          f"{'diff':>8} {'t':>7} {'n':>4}")
    for s in ("thinking", "body", "tail"):
        ma = (fam == "continuation") & (cond == "complete_nosucc") & (seg == s)
        mb = (fam == "continuation") & (cond == "correct_nocomplete") & (seg == s)
        mu, t, k = paired(base[ma].tolist(), cos[ma], base[mb].tolist(), cos[mb])
        print(f"  {s:<10} {np.nanmean(cos[mb]):>19.4f} {np.nanmean(cos[ma]):>17.4f} "
              f"{mu:>+8.4f} {t:>7.1f} {k:>4}")
    print("\n  (diff = correct_nocomplete - complete_nosucc;")
    print("   value/predictability predict +, completion predicts -)")

    # ---- B/C: probes ---------------------------------------------------------
    print("\n" + "=" * 76)
    print("B/C. YES-NO PROBES on the answer tokens (paired by conversation)")
    print("=" * 76)
    print(f"  {'probe':<22} {'phase':<6} {'Yes':>9} {'No':>9} "
          f"{'Yes-No':>9} {'t':>7} {'n':>4}")
    for q in ("done", "correct"):
        for ph in ("pre", "post"):
            ma = (fam == "probe") & (cond == f"{q}_No") & (phase == ph) & (seg == "answer")
            mb = (fam == "probe") & (cond == f"{q}_Yes") & (phase == ph) & (seg == "answer")
            mu, t, k = paired(base[ma].tolist(), cos[ma], base[mb].tolist(), cos[mb])
            print(f"  {q + '?':<22} {ph:<6} {np.nanmean(cos[mb]):>9.4f} "
                  f"{np.nanmean(cos[ma]):>9.4f} {mu:>+9.4f} {t:>7.1f} {k:>4}")

    # ---- layer sweep of the headline contrasts ------------------------------
    print("\nlayer sweep (paired diffs):")
    print(f"  {'layer':>5} {'A: tail diff':>13} {'done? Y-N post':>15} "
          f"{'correct? Y-N post':>18}")
    for l in range(1, d["cos"].shape[1], 2):
        c = d["cos"][:, l]
        ma = (fam == "continuation") & (cond == "complete_nosucc") & (seg == "tail")
        mb = (fam == "continuation") & (cond == "correct_nocomplete") & (seg == "tail")
        d1, _, _ = paired(base[ma].tolist(), c[ma], base[mb].tolist(), c[mb])
        outs = [f"{d1:+.4f}"]
        for q in ("done", "correct"):
            ma = (fam == "probe") & (cond == f"{q}_No") & (phase == "post") & (seg == "answer")
            mb = (fam == "probe") & (cond == f"{q}_Yes") & (phase == "post") & (seg == "answer")
            d2, _, _ = paired(base[ma].tolist(), c[ma], base[mb].tolist(), c[mb])
            outs.append(f"{d2:+.4f}")
        print(f"  {l:>5} {outs[0]:>13} {outs[1]:>15} {outs[2]:>18}")


if __name__ == "__main__":
    main()
