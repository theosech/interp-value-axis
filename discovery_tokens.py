"""Per-token value-axis alignment across the discovery attempt and the one after.

Input: results/discovery_tokens.npz (modal_app.py::discovery_tokens_main).

Two regions per conversation:
  discovery  every attempt in paragraph == discovery_paragraph. The last one is
             the +1 that ends the paragraph and is the "discovery moment".
  post1      paragraph == discovery_paragraph + 1, one attempt, always post_earned.

HELD-OUT: post1's assistant span is the span the value axis was built from, so
cos against the full axis is in-sample there. cos_split21 holds 10 directions
each built from 35 reward functions; for every token we average only the splits
whose held-out set contains that token's reward function. Every number below
labelled "ho" uses that, and it is the one to trust for post1.

Usage: python discovery_tokens.py [--layer 21]
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
LAYER = 21


def load(layer):
    d = np.load(ROOT / "results/discovery_tokens.npz", allow_pickle=True)
    test_fns = json.loads(
        (ROOT / "results/discovery_tokens_split_test_fns.json").read_text())

    cols = {k: d[k] for k in d.files if d[k].ndim == 1}
    cols["cos"] = d["cos"][:, layer]
    cols["hnorm"] = d["hnorm"][:, layer]
    cols["proj"] = cols["cos"] * cols["hnorm"]

    # Held-out cosine: mean over the splits that held this reward function out.
    sp = d["cos_split21"]                                   # (n, 10)
    held = np.zeros(sp.shape, dtype=bool)
    for si, fns in enumerate(test_fns):
        held[:, si] = np.isin(cols["reward_fn"], fns)
    cnt = held.sum(axis=1)
    cols["cos_ho"] = np.where(cnt > 0, (sp * held).sum(axis=1) / np.maximum(cnt, 1), np.nan)
    cols["n_held"] = cnt
    cols["cos_rand"] = d["cos_random21"].mean(axis=1)
    cols["cos_rand_absmax"] = np.abs(d["cos_random21"]).max(axis=1)
    return cols, d


def conv_mean(cols, mask, field):
    """Mean of `field` over `mask`, per conversation -> (n_convs,) array."""
    cid = cols["conv_id"][mask]
    val = cols[field][mask]
    out = {}
    for c, v in zip(cid.tolist(), val.tolist()):
        out.setdefault(c, []).append(v)
    keys = sorted(out)
    return keys, np.array([np.nanmean(out[k]) for k in keys])


def paired(cols, mask_a, mask_b, field):
    """Paired-by-conversation difference b - a."""
    ka, va = conv_mean(cols, mask_a, field)
    kb, vb = conv_mean(cols, mask_b, field)
    common = sorted(set(ka) & set(kb))
    ia = {k: i for i, k in enumerate(ka)}
    ib = {k: i for i, k in enumerate(kb)}
    a = np.array([va[ia[k]] for k in common])
    b = np.array([vb[ib[k]] for k in common])
    return a, b, common


def tstat(diff):
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    if n < 2:
        return float("nan"), float("nan"), n
    se = diff.std(ddof=1) / np.sqrt(n)
    return diff.mean(), (diff.mean() / se if se > 0 else float("nan")), n


def line(name, vals):
    v = vals[~np.isnan(vals)]
    return (f"  {name:<34} n={len(v):5d}  mean={v.mean():+.4f}  "
            f"sd={v.std():.4f}  median={np.median(v):+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()
    cols, raw = load(args.layer)
    L = args.layer

    n = len(cols["conv_id"])
    asst = cols["role"] == "assistant"
    fb = cols["role"] == "feedback"
    disc = cols["region"] == "discovery"
    post1 = cols["region"] == "post1"
    lastatt = cols["is_last_attempt"] == "True"
    think = cols["in_thinking"] == "True"
    para = ~think                      # the modified-paragraph body

    print(f"\n{n} tokens, {len(set(cols['conv_id'].tolist()))} conversations, layer {L}")
    print(f"held-out coverage: {np.mean(cols['n_held'] > 0):.1%} of tokens "
          f"({np.mean(cols['n_held']):.2f} splits each on average)\n")

    # ---- 1. the four attempt classes ------------------------------------
    print("=" * 78)
    print("1. ASSISTANT-TOKEN ALIGNMENT BY ATTEMPT CLASS  (held-out axis)")
    print("=" * 78)
    groups = [
        ("discovery: failing attempts", disc & asst & ~lastatt),
        ("discovery: THE discovery +1", disc & asst & lastatt),
        ("post1: the earned +1", post1 & asst),
    ]
    for name, m in groups:
        print(line(name, cols["cos_ho"][m]))
    print()
    for name, m in groups:
        print(line(name + " [in-sample]", cols["cos"][m]))
    print()
    for name, m in groups:
        print(line(name + " [random ctrl]", cols["cos_rand"][m]))

    print("\n  paired by conversation (held-out cosine, assistant tokens):")
    for lbl, ma, mb in [
        ("discovery+1  -  discovery fails", disc & asst & ~lastatt, disc & asst & lastatt),
        ("post1        -  discovery+1", disc & asst & lastatt, post1 & asst),
        ("post1        -  discovery fails", disc & asst & ~lastatt, post1 & asst),
    ]:
        a, b, _ = paired(cols, ma, mb, "cos_ho")
        m_, t_, n_ = tstat(b - a)
        print(f"    {lbl:<34} d={m_:+.4f}  t={t_:+6.1f}  n={n_}")

    # ---- 2. thinking block vs paragraph ---------------------------------
    print("\n" + "=" * 78)
    print("2. <thinking> BLOCK vs MODIFIED PARAGRAPH  (held-out axis)")
    print("=" * 78)
    for rname, rm in [("discovery fails", disc & ~lastatt), ("discovery +1", disc & lastatt),
                      ("post1", post1)]:
        for tname, tm in [("thinking", think), ("paragraph", para)]:
            print(line(f"{rname} / {tname}", cols["cos_ho"][rm & asst & tm]))
        a, b, _ = paired(cols, rm & asst & think, rm & asst & para, "cos_ho")
        m_, t_, n_ = tstat(b - a)
        print(f"    paragraph - thinking: d={m_:+.4f}  t={t_:+6.1f}  n={n_}\n")

    # ---- 3. within-attempt trajectory -----------------------------------
    print("=" * 78)
    print(f"3. WITHIN-ATTEMPT TRAJECTORY, assistant tokens binned into {args.bins}")
    print("=" * 78)
    frac = cols["pos_in_span"] / np.maximum(cols["n_span"] - 1, 1)
    b = np.clip((frac * args.bins).astype(int), 0, args.bins - 1)
    hdr = f"  {'bin':>4} " + "".join(f"{g:>14}" for g, _ in groups)
    print(hdr)
    for bi in range(args.bins):
        row = f"  {bi/args.bins:.1f}-{(bi+1)/args.bins:.1f}"[:6].ljust(6)
        for _, m in groups:
            v = cols["cos_ho"][m & (b == bi)]
            v = v[~np.isnan(v)]
            row += f"{v.mean():>14.4f}" if len(v) else f"{'-':>14}"
        print(row)

    # ---- 4. the feedback token ------------------------------------------
    print("\n" + "=" * 78)
    print("4. FEEDBACK TOKENS  (the '+1' / '-1' the model receives)")
    print("=" * 78)
    for name, m in [("discovery: -1 after a fail", disc & fb & ~lastatt),
                    ("discovery: +1 discovery moment", disc & fb & lastatt),
                    ("post1: +1 earned", post1 & fb)]:
        print(line(name, cols["cos_ho"][m]))
    a, b_, _ = paired(cols, disc & fb & lastatt, post1 & fb, "cos_ho")
    m_, t_, n_ = tstat(b_ - a)
    print(f"\n  paired  post1(+1 earned) - discovery(+1 moment): "
          f"d={m_:+.4f}  t={t_:+6.1f}  n={n_}")

    # ---- 5. before/after within post1 -----------------------------------
    print("\n" + "=" * 78)
    print("5. post1 CONSTRUCTION LABELS (the axis's own training split)")
    print("=" * 78)
    for name, m in [("post1 'before' tokens", post1 & (cols["axis_label"] == "before")),
                    ("post1 'after' tokens", post1 & (cols["axis_label"] == "after")),
                    ("post1 'excluded' tokens", post1 & asst & (cols["axis_label"] == "excluded"))]:
        print(line(name, cols["cos_ho"][m]))
    a, b_, _ = paired(cols, post1 & (cols["axis_label"] == "before"),
                      post1 & (cols["axis_label"] == "after"), "cos_ho")
    m_, t_, n_ = tstat(b_ - a)
    print(f"\n  paired  after - before (held-out): d={m_:+.4f}  t={t_:+6.1f}  n={n_}")
    a, b_, _ = paired(cols, post1 & (cols["axis_label"] == "before"),
                      post1 & (cols["axis_label"] == "after"), "cos")
    m_, t_, n_ = tstat(b_ - a)
    print(f"  paired  after - before (in-sample): d={m_:+.4f}  t={t_:+6.1f}  n={n_}")

    # ---- 6. layer profile of the headline contrast ----------------------
    print("\n" + "=" * 78)
    print("6. LAYER PROFILE: post1(+1 earned) - discovery(+1 moment), assistant tokens")
    print("=" * 78)
    print(f"  {'layer':>5} {'disc fails':>11} {'disc +1':>10} {'post1':>10} "
          f"{'post1-disc+1':>13} {'t':>7}")
    for l in range(raw["cos"].shape[1]):
        c = {"cos": raw["cos"][:, l], "conv_id": cols["conv_id"]}
        vals = []
        for _, m in groups:
            vals.append(np.nanmean(raw["cos"][m, l]))
        a, b_, _ = paired(c, disc & asst & lastatt, post1 & asst, "cos")
        m_, t_, _ = tstat(b_ - a)
        print(f"  {l:5d} {vals[0]:11.4f} {vals[1]:10.4f} {vals[2]:10.4f} "
              f"{m_:13.4f} {t_:7.1f}")


if __name__ == "__main__":
    main()
