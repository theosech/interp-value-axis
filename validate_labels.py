"""Print a stratified sample of judge labels for manual validation.

Usage: python validate_labels.py [--n 20] [--seed 0]
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results" / "rule_labels.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    counts = Counter(r["label"] for r in rows)
    print(f"{len(rows)} labelled think blocks")
    for k in ("correct", "wrong", "unclear"):
        print(f"   {k:8s} {counts[k]:5d}  ({counts[k]/len(rows):5.1%})")
    print(f"   scaffolding-leak flagged: {sum(r['leak'] for r in rows)}")
    print()

    # Stratify so all three labels are represented even though `wrong` dominates.
    rng = random.Random(args.seed)
    per = max(1, args.n // 3)
    sample = []
    for lab in ("correct", "wrong", "unclear"):
        pool = [r for r in rows if r["label"] == lab]
        sample += rng.sample(pool, min(per, len(pool)))
    rest = [r for r in rows if r not in sample]
    sample += rng.sample(rest, max(0, args.n - len(sample)))
    rng.shuffle(sample)

    for i, r in enumerate(sample[:args.n], 1):
        flag = "  [LEAK]" if r["leak"] else ""
        print(f"--- {i:2d}. label={r['label'].upper()}{flag}   "
              f"({r['reward_fn']}, paragraph {r['paragraph']}, "
              f"discovery_paragraph {r['discovery_paragraph']})")
        print(f"    CRITERION: {r['criterion']}")
        print(f"    THINK    : {r['think']}")
        print()


if __name__ == "__main__":
    main()
