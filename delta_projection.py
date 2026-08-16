"""Project the pre/post-split mean-activation difference vector onto the value axis.

For each reward function f and layer l, the "difference vector" is

    delta_f[l] = after_mean_f[l] - before_mean_f[l]

i.e. the mean residual-stream activation over post-split (after the
criterion-satisfying token) paragraph tokens minus the mean over pre-split
tokens. The value axis u[l] is the unit-normalized mean of those deltas over
reward functions (compute_vector.compute_value_axis).

Reported per layer:
  proj      delta_f . u_hat            scalar projection, activation-norm units
  frac      proj / ||delta_f||         = cos(delta_f, u), how axis-aligned the
                                        difference is (1.0 = entirely on-axis)

IN-SAMPLE is circular: u is built from these same deltas, so its mean projection
is positive by construction. HELD-OUT rebuilds u from the 35 training functions
of each of the 10 splits and projects only the 13 held-out functions' deltas --
same seeding as compute_vector.evaluate_heldout_auroc.

Usage: python delta_projection.py [--layer 21] [--csv results/delta_projection.csv]
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
for p in (ROOT / "value-axis", ROOT / "value-axis" / "construction"):
    sys.path.insert(0, str(p))

from compute_vector import (N_SPLITS, N_TRAIN, compute_value_axis,  # noqa: E402
                            geometry, load_activation_means,
                            mean_contrastive_direction, valid_functions)


def deltas(data, fns):
    """(n_fns, n_layers, hidden) stack of after_mean - before_mean, float64."""
    n_layers, hidden = geometry(data)
    out = np.zeros((len(fns), n_layers, hidden), dtype=np.float64)
    for i, fn in enumerate(fns):
        d = data[fn]
        for l in range(n_layers):
            out[i, l] = (d["after_mean"][l].numpy().astype(np.float64)
                         - d["before_mean"][l].numpy().astype(np.float64))
    return out


def unit(v):
    """Row-normalize the last axis."""
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--csv", default="results/delta_projection.csv")
    args = ap.parse_args()

    data = load_activation_means()
    fns = valid_functions(data)
    n_layers, _ = geometry(data)
    D = deltas(data, fns)                          # (n_fns, n_layers, hidden)
    norms = np.linalg.norm(D, axis=2)              # (n_fns, n_layers)

    # ---- in-sample: axis built from all functions -------------------------
    axis = unit(compute_value_axis(data))          # (n_layers, hidden)
    proj_in = np.einsum("flh,lh->fl", D, axis)     # (n_fns, n_layers)
    frac_in = proj_in / norms.clip(1e-10)

    # ---- held-out: axis built from the split's 35 train functions ---------
    idx = {fn: i for i, fn in enumerate(fns)}
    proj_ho = [[] for _ in range(n_layers)]
    frac_ho = [[] for _ in range(n_layers)]
    for si in range(N_SPLITS):
        rng = random.Random(si * 42)
        shuffled = fns.copy()
        rng.shuffle(shuffled)
        u = unit(mean_contrastive_direction(data, set(shuffled[:N_TRAIN])))
        for fn in shuffled[N_TRAIN:]:
            i = idx[fn]
            p = D[i] @ u.T                          # (n_layers, n_layers)
            p = np.diagonal(p)                      # delta_f[l] . u[l]
            for l in range(n_layers):
                proj_ho[l].append(p[l])
                frac_ho[l].append(p[l] / max(norms[i, l], 1e-10))
    proj_ho = np.array(proj_ho)                     # (n_layers, n_scores)
    frac_ho = np.array(frac_ho)

    # ---- report -----------------------------------------------------------
    print(f"\n{len(fns)} reward functions, {n_layers} layers "
          f"(index 0 = embedding output; layer L = output of block L-1)\n")
    hdr = (f"{'layer':>5} | {'||delta||':>9} | {'proj_in':>9} {'frac_in':>8} | "
           f"{'proj_ho':>9} {'+/-':>7} {'frac_ho':>8} {'>0':>6}")
    print(hdr)
    print("-" * len(hdr))
    for l in range(n_layers):
        print(f"{l:5d} | {norms[:, l].mean():9.3f} | "
              f"{proj_in[:, l].mean():9.4f} {frac_in[:, l].mean():8.3f} | "
              f"{proj_ho[l].mean():9.4f} {proj_ho[l].std():7.4f} "
              f"{frac_ho[l].mean():8.3f} {np.mean(proj_ho[l] > 0):6.1%}")

    L = args.layer
    print(f"\n=== layer {L} ===")
    print(f"mean ||delta||          {norms[:, L].mean():.3f}  "
          f"(sd {norms[:, L].std():.3f})")
    print(f"in-sample  proj         {proj_in[:, L].mean():.4f}  "
          f"(sd {proj_in[:, L].std():.4f}), frac {frac_in[:, L].mean():.3f}")
    print(f"held-out   proj         {proj_ho[L].mean():.4f}  "
          f"(sd {proj_ho[L].std():.4f}), frac {frac_ho[L].mean():.3f}, "
          f"{np.mean(proj_ho[L] > 0):.1%} positive")
    order = np.argsort(-proj_in[:, L])
    print(f"\nper-function projection at layer {L} (in-sample axis):")
    for i in order:
        print(f"  {fns[i]:<24} {proj_in[i, L]:8.4f}  "
              f"frac {frac_in[i, L]:6.3f}  ||delta|| {norms[i, L]:8.3f}")

    if args.csv:
        out = ROOT / args.csv
        out.parent.mkdir(exist_ok=True)
        with open(out, "w") as f:
            f.write("reward_fn,layer,delta_norm,proj_in,frac_in\n")
            for i, fn in enumerate(fns):
                for l in range(n_layers):
                    f.write(f"{fn},{l},{norms[i, l]:.6f},"
                            f"{proj_in[i, l]:.6f},{frac_in[i, l]:.6f}\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
