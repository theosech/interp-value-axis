"""Confidence-vs-length dissociation under value-axis steering.

Input: results/steering_probe.jsonl (modal_app.py::steering_probe_main).

Per (direction, state, alpha): mean parsed confidence rating (0-10), mean
response length in tokens, % hitting the token cap, % rating-parse failures,
and a degeneration flag (low unique-trigram fraction). Dose-response slopes
per prefix (Spearman across alphas), tested against zero across prefixes.

Predictions:
  confidence-encoding      rating rises with alpha
  episode-closure/wrap-up  length falls with alpha, rating flat
"""

import collections
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent


def trigram_uniq(text):
    toks = text.split()
    if len(toks) < 10:
        return 1.0
    tri = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
    return len(set(tri)) / len(tri)


def main():
    rows = [json.loads(l) for l in open(ROOT / "results/steering_probe.jsonl")]
    for r in rows:
        r["degen"] = trigram_uniq(r["text"]) < 0.5

    alphas = sorted(set(r["alpha"] for r in rows))
    print(f"{len(rows)} generations, "
          f"{len(set((r['conv_id'], r['state']) for r in rows))} prefixes\n")

    for dname in ("shipped", "corrected"):
        print("=" * 78)
        print(f"DIRECTION: {dname}  (alpha=0 shared between directions)")
        print("=" * 78)
        for state in ("early", "post"):
            print(f"  state={state}")
            print(f"  {'alpha':>6} {'rating':>7} {'len(tok)':>9} {'@cap':>6} "
                  f"{'no-parse':>9} {'degen':>6} {'n':>4}")
            for a in alphas:
                sel = [r for r in rows if r["state"] == state and r["alpha"] == a
                       and (r["direction"] == dname or a == 0)]
                if not sel:
                    continue
                ok = [r for r in sel if r["rating"] >= 0 and not r["degen"]]
                rat = np.mean([r["rating"] for r in ok]) if ok else float("nan")
                ln = np.mean([r["n_tokens"] for r in sel])
                cap = np.mean([not r["ended"] for r in sel])
                nop = np.mean([r["rating"] < 0 for r in sel])
                dg = np.mean([r["degen"] for r in sel])
                print(f"  {a:>+6} {rat:>7.2f} {ln:>9.0f} {cap:>6.0%} "
                      f"{nop:>9.0%} {dg:>6.0%} {len(sel):>4}")
            print()

        # per-prefix dose-response slopes (Spearman across alphas)
        for metric, get, cond in [
            ("rating", lambda r: r["rating"],
             lambda r: r["rating"] >= 0 and not r["degen"]),
            ("length", lambda r: r["n_tokens"], lambda r: True),
        ]:
            slopes = []
            for key in set((r["conv_id"], r["state"]) for r in rows):
                pts = [(r["alpha"], get(r)) for r in rows
                       if (r["conv_id"], r["state"]) == key
                       and (r["direction"] == dname or r["alpha"] == 0)
                       and cond(r)]
                if len(pts) >= 5 and len(set(p[0] for p in pts)) >= 4:
                    rho, _ = spearmanr(*zip(*pts))
                    if not np.isnan(rho):
                        slopes.append(rho)
            s = np.array(slopes)
            se = s.std(ddof=1) / np.sqrt(len(s))
            print(f"  {metric} ~ alpha: mean per-prefix Spearman rho = "
                  f"{s.mean():+.3f}  t={s.mean()/se:+.1f}  n={len(s)} prefixes")
        print()


if __name__ == "__main__":
    main()
