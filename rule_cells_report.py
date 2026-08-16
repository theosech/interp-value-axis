"""FC/FW/RC/RW x before/actual/after table of attempt-mean value-axis cosines.

Cells: R/F = attempt rewarded / failed; C/W = the model's stated rule belief was
correct / wrong (rule_labels.jsonl, joined to attempts via the id alignment with
believed_words.jsonl, which carries attempt_in_paragraph). Position is relative
to THE discovery attempt (the +1 ending paragraph dp): strictly earlier =
before, that attempt = actual, later = after. leak=True attempts are excluded.

The measure is asst_cos: the mean per-token cosine over the attempt's assistant
tokens (thinking + paragraph) -- not the cosine of the mean activation, which
was never stored.

Also reports the held-out version where available: fb_split exists (feedback
tokens) but asst tokens have no split projections in attempts*.npz, so the main
table is the full-axis cosine; interpret cross-function levels accordingly.

Usage:
  python rule_cells_report.py                       # corrected axis (default)
  python rule_cells_report.py --npz results/attempts.npz   # shipped axis
  (attempts.npz may live in archive/results/ -- shipped-axis numbers are
   retained in the write-up tables either way)
"""

import argparse
import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/attempts_corrected.npz")
    ap.add_argument("--layer", type=int, default=21)
    args = ap.parse_args()
    L = args.layer

    bw = {r["id"]: r for r in
          (json.loads(l) for l in open(RES / "believed_words.jsonl") if l.strip())}
    label_by_key = {}
    for line in open(RES / "rule_labels.jsonl"):
        if not line.strip():
            continue
        r = json.loads(line)
        b = bw.get(r["id"])
        if b is None or b["conv_id"] != r["conv_id"] or b["paragraph"] != r["paragraph"]:
            continue
        label_by_key[(r["conv_id"], r["paragraph"], b["attempt_in_paragraph"])] = (
            r["label"], r["leak"])

    a = np.load(ROOT / args.npz, allow_pickle=True)
    cos = a["asst_cos"][:, L]
    conv, para, att = a["conv_id"], a["paragraph"], a["attempt_in_paragraph"]
    dp, rew = a["discovery_paragraph"], a["reward"]

    last_in_dp = {}
    for i in range(len(conv)):
        if para[i] == dp[i]:
            last_in_dp[conv[i]] = max(last_in_dp.get(conv[i], 0), int(att[i]))

    cells = collections.defaultdict(list)
    n_unlabeled = n_leak = 0
    for i in range(len(conv)):
        lab = label_by_key.get((conv[i], int(para[i]), int(att[i])))
        if lab is None:
            n_unlabeled += 1
            continue
        label, leak = lab
        if leak:
            n_leak += 1
            continue
        cname = ("R" if rew[i] else "F") + ("C" if label == "correct" else "W")
        if para[i] < dp[i] or (para[i] == dp[i] and att[i] < last_in_dp[conv[i]]):
            pos = "before"
        elif para[i] == dp[i] and att[i] == last_in_dp[conv[i]]:
            pos = "actual"
        else:
            pos = "after"
        cells[(cname, pos)].append(cos[i])

    print(f"\n{args.npz} @L{L} -- mean asst-token cosine "
          f"(unlabeled: {n_unlabeled}, leak-excluded: {n_leak})")
    print(f"{'cell':>5} | {'before':>20} | {'actual':>20} | {'after':>20}")
    for c in ("FW", "FC", "RW", "RC"):
        line = f"{c:>5} |"
        for pos in ("before", "actual", "after"):
            v = cells.get((c, pos))
            line += (f"  {np.mean(v):+.4f} (n={len(v):4d}) |" if v
                     else f"  {'--':>17} |")
        print(line)

    # matched-position contrasts
    print("\ncontrasts at matched position (mean difference, unpaired):")
    for pos in ("before", "actual"):
        rc, fw = cells.get(("RC", pos)), cells.get(("FW", pos))
        rw = cells.get(("RW", pos))
        if rc and fw:
            print(f"  {pos:>7}: RC - FW = {np.mean(rc)-np.mean(fw):+.4f}")
        if rw and fw:
            print(f"  {pos:>7}: RW - FW = {np.mean(rw)-np.mean(fw):+.4f}  (RW n={len(rw)})")


if __name__ == "__main__":
    main()
