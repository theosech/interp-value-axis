"""Mean-level held-out validation of the corrected axis (the paper's own check).

For each of 10 splits, build the direction from 35 reward functions' corrected
(after_mean - before_mean); for each of the 13 held-out functions ask whether its
after_mean projects higher than its before_mean on that direction. This is
upstream evaluate_heldout_auroc run on activation_means_corrected.pt.

The 2-point AUROC saturates by construction (1.0 iff after > before), so the
margin table is the informative part: the held-out projection of the mean delta,
its size relative to ||delta||, and the fraction positive - per layer, corrected
vs shipped means side by side.

CPU-only. Usage: python corrected_mean_validation.py [--layer 21]
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
for p in (ROOT / "value-axis", ROOT / "value-axis" / "construction"):
    sys.path.insert(0, str(p))

from compute_vector import (N_SPLITS, N_TRAIN, load_activation_means,  # noqa: E402
                            mean_contrastive_direction, valid_functions)


def heldout_margins(data):
    """Per layer: held-out projections of (after-before) means onto split dirs.

    Returns (n_layers, n_scores) proj and matching frac = proj/||delta||.
    """
    fns = valid_functions(data)
    n_layers = len(next(iter(data.values()))["before_mean"])
    proj = [[] for _ in range(n_layers)]
    frac = [[] for _ in range(n_layers)]
    for si in range(N_SPLITS):
        rng = random.Random(si * 42)
        sh = fns.copy()
        rng.shuffle(sh)
        u = mean_contrastive_direction(data, set(sh[:N_TRAIN]))
        u = u / np.linalg.norm(u, axis=1, keepdims=True).clip(1e-10)
        for fn in sh[N_TRAIN:]:
            d = data[fn]
            for l in range(n_layers):
                delta = (d["after_mean"][l].numpy().astype(np.float64)
                         - d["before_mean"][l].numpy().astype(np.float64))
                p = float(delta @ u[l])
                proj[l].append(p)
                frac[l].append(p / max(np.linalg.norm(delta), 1e-10))
    return np.array(proj), np.array(frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=21)
    args = ap.parse_args()
    L = args.layer

    corrected = torch.load(ROOT / "results" / "activation_means_corrected.pt",
                           map_location="cpu", weights_only=False)
    shipped = load_activation_means()

    pc, fc = heldout_margins(corrected)
    ps, fs = heldout_margins(shipped)
    n_layers = pc.shape[0]

    print(f"\nMean-level held-out validation: does a held-out function's after_mean")
    print(f"project higher than its before_mean on a direction built from 35 others?")
    print(f"{pc.shape[1]} (split x held-out fn) scores per layer.\n")
    hdr = (f"{'layer':>5} | {'corr %pos':>9} {'proj':>8} {'frac':>7} | "
           f"{'ship %pos':>9} {'proj':>8} {'frac':>7}")
    print(hdr)
    print("-" * len(hdr))
    for l in range(n_layers):
        print(f"{l:5d} | {np.mean(pc[l] > 0):9.1%} {pc[l].mean():8.3f} "
              f"{fc[l].mean():7.3f} | {np.mean(ps[l] > 0):9.1%} "
              f"{ps[l].mean():8.3f} {fs[l].mean():7.3f}")

    print(f"\n=== layer {L} ===")
    print(f"  corrected: {np.mean(pc[L] > 0):.1%} positive "
          f"(AUROC-equivalent {np.mean(pc[L] > 0):.3f}), "
          f"proj {pc[L].mean():.3f} +/- {pc[L].std():.3f}, frac {fc[L].mean():.3f}")
    print(f"  shipped  : {np.mean(ps[L] > 0):.1%} positive, "
          f"proj {ps[L].mean():.3f} +/- {ps[L].std():.3f}, frac {fs[L].mean():.3f}")
    neg_c = [(l, np.mean(pc[l] > 0)) for l in range(n_layers) if np.mean(pc[l] > 0) < 1.0]
    print(f"\n  layers where corrected is not 100% positive: "
          f"{neg_c if neg_c else 'none'}")


if __name__ == "__main__":
    main()
