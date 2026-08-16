"""Pre/post-split projections per attempt type, held-out CORRECTED axis.

Input: results/attempt_split.npz (modal_app.py::attempt_split_main).

Every row is one attempt x one split channel:
  actual   : split at the criterion-satisfying tokens (rewarded attempts only)
  believed : split at the model's own stated target words (all attempts)

Held-out cosine = mean over the splits whose held-out set contains the row's
reward function (directions built from activation_means_corrected.pt).

Groups cross para_phase (pre / disc / post relative to discovery_paragraph)
with outcome; the disc-paragraph +1 is the discovery moment itself.

Usage: python attempt_split_report.py [--layer 21]
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"


def tstat(x):
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan"), float("nan"), len(x)
    se = x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() / se if se > 0 else float("nan"), len(x)


def conv_mean(conv_ids, vals):
    acc = {}
    for c, v in zip(conv_ids, vals):
        acc.setdefault(c, []).append(v)
    return {c: float(np.nanmean(v)) for c, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=21)
    args = ap.parse_args()
    L = args.layer

    d = np.load(RES / "attempt_split.npz", allow_pickle=True)
    test_fns = json.loads((RES / "attempt_split_test_fns.json").read_text())
    n = len(d["conv_id"])

    # Held-out mask and per-row held-out before/after at layer L.
    held = np.zeros((n, len(test_fns)), dtype=bool)
    for si, fns in enumerate(test_fns):
        held[:, si] = np.isin(d["reward_fn"], fns)
    cnt = held.sum(axis=1).astype(float)
    bef_ho = (d["before_split"][:, :, L] * held).sum(axis=1) / np.maximum(cnt, 1)
    aft_ho = (d["after_split"][:, :, L] * held).sum(axis=1) / np.maximum(cnt, 1)
    bef_ho[cnt == 0] = np.nan
    aft_ho[cnt == 0] = np.nan

    bef_sh, aft_sh = d["before_shipped"][:, L], d["after_shipped"][:, L]

    ch = d["channel"]
    phase = d["para_phase"]
    cellv = d["cell"]
    disc_moment = d["is_discovery_moment"] == "True"
    conv = d["conv_id"]

    def group_label(i):
        p, c = phase[i], cellv[i]
        if disc_moment[i]:
            return "disc:DISCOVERY +1"
        if c == "no_feedback":
            return f"{p}:fail (no feedback)"
        pretty = {"pre_fail": "fail -1", "pre_lucky": "lucky +1",
                  "post_earned": "earned +1", "post_slip": "slip -1"}[c]
        return f"{p}:{pretty}"

    labels = np.array([group_label(i) for i in range(n)])
    order = ["pre:fail -1", "pre:fail (no feedback)", "pre:lucky +1",
             "disc:fail -1", "disc:fail (no feedback)", "disc:DISCOVERY +1",
             "post:earned +1", "post:slip -1", "post:fail (no feedback)"]

    print(f"\n{n} rows, {len(set(conv.tolist()))} conversations, layer {L}, "
          f"held-out corrected axis")
    for channel in ("actual", "believed"):
        print("\n" + "=" * 84)
        print(f"CHANNEL: {channel} split "
              f"({'criterion tokens, rewarded attempts only' if channel == 'actual' else 'model-stated target words, all attempts'})")
        print("=" * 84)
        print(f"  {'group':<26} {'n':>5} {'before':>9} {'after':>9} "
              f"{'diff':>9} {'t':>7} {'| shipped diff':>14}")
        for g in order:
            m = (ch == channel) & (labels == g)
            if not m.sum():
                continue
            diff = aft_ho[m] - bef_ho[m]
            mu, t, k = tstat(diff)
            sh = np.nanmean(aft_sh[m] - bef_sh[m])
            print(f"  {g:<26} {m.sum():5d} {np.nanmean(bef_ho[m]):9.4f} "
                  f"{np.nanmean(aft_ho[m]):9.4f} {mu:9.4f} {t:7.1f} {sh:14.4f}")

    # ---- paired contrasts (by conversation, held-out, after-before diff) ----
    print("\n" + "=" * 84)
    print("PAIRED CONTRASTS (per-conversation means of after-before, held-out)")
    print("=" * 84)
    contrasts = [
        ("actual",   "disc:DISCOVERY +1", "pre:lucky +1",
         "discovery +1 vs pre-disc lucky +1 (same actual-split instrument)"),
        ("actual",   "post:earned +1", "disc:DISCOVERY +1",
         "post earned +1 vs discovery +1"),
        ("actual",   "post:earned +1", "pre:lucky +1",
         "post earned +1 vs pre-disc lucky +1"),
        ("believed", "pre:lucky +1", "pre:fail -1",
         "pre-disc lucky +1 vs pre-disc fail (believed split)"),
        ("believed", "disc:DISCOVERY +1", "disc:fail -1",
         "discovery +1 vs disc-paragraph fail (believed split)"),
    ]
    for channel, ga, gb, desc in contrasts:
        ma = (ch == channel) & (labels == ga)
        mb = (ch == channel) & (labels == gb)
        if not (ma.sum() and mb.sum()):
            continue
        da = conv_mean(conv[ma].tolist(), (aft_ho - bef_ho)[ma])
        db = conv_mean(conv[mb].tolist(), (aft_ho - bef_ho)[mb])
        common = sorted(set(da) & set(db))
        diff = np.array([da[c] - db[c] for c in common])
        mu, t, k = tstat(diff)
        print(f"  {desc}\n    d={mu:+.4f}  t={t:+6.1f}  n={k} convs")

    # ---- BEFORE-level across phases: the state-value readout ---------------
    print("\n" + "=" * 84)
    print("BEFORE-SPAN LEVEL by phase (believed channel; pre-criterion text only)")
    print("  A stationary V(s) predicts post > disc > pre in the BEFORE level;")
    print("  a within-attempt jump story predicts the levels are flat.")
    print("=" * 84)
    for g in order:
        m = (ch == "believed") & (labels == g)
        if not m.sum():
            continue
        mu, t, k = tstat(bef_ho[m])
        print(f"  {g:<26} n={m.sum():5d}  before={np.nanmean(bef_ho[m]):+.4f} "
              f"(sd {np.nanstd(bef_ho[m]):.4f})")


if __name__ == "__main__":
    main()
