"""Gates report for the corrected-span axis rebuild.

Inputs (from modal_app.py::rebuild_axis_main):
  results/value_axis_corrected.npy         corrected axis (37, 4096)
  results/activation_means_corrected.pt    corrected per-function means
  results/projections_corrected.npz        per-token cosines, corrected labels
  results/corrected_split_test_fns.json    held-out membership per split

Baselines it compares against (from the original replication, contaminated
labels on both sides): held-out token AUROC 0.850 +/- 0.017 @L21, peak 0.852
@L20; per-function range 0.622 (contains_colon) .. 0.994 (contains_parentheses).

Usage: python corrected_axis_report.py [--layer 21]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
SHIPPED_AXIS = ROOT / "value-axis" / "data" / "value_axis.npy"


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=21)
    args = ap.parse_args()
    L = args.layer

    d = np.load(RES / "projections_corrected.npz", allow_pickle=True)
    test_fns = json.loads((RES / "corrected_split_test_fns.json").read_text())
    corr = np.load(RES / "value_axis_corrected.npy")
    ship = np.load(SHIPPED_AXIS)
    n_layers = corr.shape[0]

    y = (d["label"] == "after").astype(int)
    fn = d["reward_fn"]
    n = len(y)
    print(f"\n{n} corrected-label tokens "
          f"({int(y.sum())} after / {int((1-y).sum())} before), "
          f"{len(set(d['conv_id'].tolist()))} conversations, "
          f"{len(set(fn.tolist()))} reward functions")

    # ---- gate 1: axis similarity --------------------------------------------
    cu, su = unit(corr), unit(ship)
    ax_cos = np.einsum("ld,ld->l", cu, su)
    print("\n" + "=" * 74)
    print("GATE 1: cos(corrected axis, shipped axis) per layer")
    print("=" * 74)
    for l0 in range(0, n_layers, 8):
        seg = range(l0, min(l0 + 8, n_layers))
        print("  layer " + " ".join(f"{l:>7d}" for l in seg))
        print("  cos   " + " ".join(f"{ax_cos[l]:>7.3f}" for l in seg))
    print(f"\n  layer {L}: {ax_cos[L]:.4f}   "
          f"(min {ax_cos.min():.3f} @L{ax_cos.argmin()}, "
          f"max {ax_cos.max():.3f} @L{ax_cos.argmax()})")

    # ---- gate 2 + 3: held-out token AUROC per layer --------------------------
    held = np.zeros((n, len(test_fns)), dtype=bool)
    for si, fns in enumerate(test_fns):
        held[:, si] = np.isin(fn, fns)

    def split_aurocs(layer):
        """AUROC per split: split si's direction scored on its held-out tokens."""
        out = []
        for si in range(len(test_fns)):
            m = held[:, si]
            if m.sum() and 0 < y[m].sum() < m.sum():
                out.append(roc_auc_score(y[m], d["cos_split"][m, si, layer]))
        return np.array(out)

    print("\n" + "=" * 74)
    print("GATE 2/3: token AUROC per layer (before vs after, corrected labels)")
    print("  ho_corr : corrected split directions, held-out functions only")
    print("  in_corr : corrected full axis, all tokens (in-sample)")
    print("  shipped : SHIPPED full axis on the corrected labels")
    print("=" * 74)
    print(f"  {'layer':>5} {'ho_corr':>9} {'+/-':>6} {'in_corr':>9} {'shipped':>9}")
    ho_by_layer = []
    for l in range(n_layers):
        ho = split_aurocs(l)
        ho_by_layer.append(ho.mean())
        in_c = roc_auc_score(y, d["cos"][:, l])
        in_s = roc_auc_score(y, d["cos_shipped"][:, l])
        mark = "  <-- L%d" % L if l == L else ""
        print(f"  {l:5d} {ho.mean():9.3f} {ho.std():6.3f} {in_c:9.3f} {in_s:9.3f}{mark}")
    ho_by_layer = np.array(ho_by_layer)
    print(f"\n  held-out peak: {ho_by_layer.max():.3f} @L{ho_by_layer.argmax()}  "
          f"| baseline (contaminated): 0.850 +/- 0.017 @L21, peak 0.852 @L20")

    # ---- gate 4: controls -----------------------------------------------------
    print("\n" + "=" * 74)
    print(f"GATE 4: controls at layer {L}")
    print("=" * 74)
    r_auc = [roc_auc_score(y, d["cos_random"][:, ri])
             for ri in range(d["cos_random"].shape[1])]
    print(f"  random directions : {np.mean(r_auc):.3f} +/- {np.std(r_auc):.3f}  "
          f"(n={len(r_auc)}; expect ~0.5)")
    rng = np.random.default_rng(0)
    s_auc = [roc_auc_score(rng.permutation(y), d["cos"][:, L]) for _ in range(20)]
    print(f"  shuffled labels   : {np.mean(s_auc):.3f} +/- {np.std(s_auc):.3f}  "
          f"(corrected axis; expect ~0.5)")

    # ---- gate 5: per-function AUROC at L ------------------------------------
    print("\n" + "=" * 74)
    print(f"GATE 5: per-function held-out AUROC at layer {L} (corrected)")
    print("=" * 74)
    rows = []
    for f in sorted(set(fn.tolist())):
        mf = fn == f
        splits = [si for si, fns in enumerate(test_fns) if f in fns]
        aucs = [roc_auc_score(y[mf], d["cos_split"][mf, si, L]) for si in splits
                if 0 < y[mf].sum() < mf.sum()]
        if aucs:
            rows.append((f, np.mean(aucs), len(aucs), int(mf.sum())))
    rows.sort(key=lambda r: -r[1])
    above = sum(1 for _, a, _, _ in rows if a > 0.5)
    print(f"  {above}/{len(rows)} functions above chance "
          f"(baseline: 48/48, median 0.867, range 0.622-0.994)")
    print(f"  median: {np.median([a for _, a, _, _ in rows]):.3f}\n")
    for f, a, ns, nt in rows:
        print(f"    {f:<24} {a:.3f}  ({ns} splits, {nt} tokens)")


if __name__ == "__main__":
    main()
