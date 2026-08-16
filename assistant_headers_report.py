"""Assistant-header cosines on the held-out corrected axis, by cell.

Input: results/assistant_headers.npz (modal_app.py::assistant_headers_main).

The header span ("<|im_end|>\\n<|im_start|>assistant\\n<thinking>", 7 tokens) is
byte-identical everywhere, so cell differences are attributable to state.
Held-out cosine = mean over splits whose test set contains the row's reward fn.

Cells cross para_phase with the preceding message:
  prev=minus1     retry headers (only exist after a -1)
  prev=paragraph  paragraph-initial headers (after a "Paragraph N" message)

The requested contrast (post vs discovery) spans the two prev types; the
left-context-matched contrasts are within a prev type.

Usage: python assistant_headers_report.py [--layer 21]
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


def conv_mean(cids, vals):
    acc = {}
    for c, v in zip(cids, vals):
        acc.setdefault(c, []).append(v)
    return {c: float(np.nanmean(v)) for c, v in acc.items()}


def paired(cids_a, va, cids_b, vb):
    da, db = conv_mean(cids_a, va), conv_mean(cids_b, vb)
    common = sorted(set(da) & set(db))
    return np.array([db[c] - da[c] for c in common])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=21)
    args = ap.parse_args()
    L = args.layer

    d = np.load(RES / "assistant_headers.npz", allow_pickle=True)
    test_fns = json.loads((RES / "assistant_headers_split_test_fns.json").read_text())
    n = len(d["conv_id"])

    held = np.zeros((n, len(test_fns)), dtype=bool)
    for si, fns in enumerate(test_fns):
        held[:, si] = np.isin(d["reward_fn"], fns)
    cnt = held.sum(axis=1).astype(float)
    ho = (d["hdr_split"][:, :, L] * held).sum(axis=1) / np.maximum(cnt, 1)
    ho[cnt == 0] = np.nan

    prev = d["prev"]
    phase = d["para_phase"]
    dm = d["is_discovery_moment"] == "True"
    conv = d["conv_id"]

    cells = [
        ("after -1 : pre-disc retries", (prev == "minus1") & (phase == "pre")),
        ("after -1 : disc-para retries (not last)", (prev == "minus1") & (phase == "disc") & ~dm),
        ("after -1 : DISCOVERY attempt", (prev == "minus1") & dm),
        ("after Par: pre first-attempts", (prev == "paragraph") & (phase == "pre")),
        ("after Par: disc first-attempts", (prev == "paragraph") & (phase == "disc") & ~dm),
        ("after Par: disc first=DISCOVERY", (prev == "paragraph") & dm),
        ("after Par: post-disc attempts", (prev == "paragraph") & (phase == "post")),
    ]

    print(f"\n{n} headers, {len(set(conv.tolist()))} conversations, layer {L}, "
          f"held-out corrected axis (7-token content-matched span)\n")
    print(f"  {'cell':<40} {'n':>5} {'mean':>9} {'sd':>8}")
    for name, m in cells:
        if m.sum():
            print(f"  {name:<40} {m.sum():5d} {np.nanmean(ho[m]):9.4f} "
                  f"{np.nanstd(ho[m]):8.4f}")

    print("\n  paired-by-conversation contrasts (held-out):")
    contrasts = [
        ("DISCOVERY hdr - pre-disc retry hdrs   [matched after -1]",
         (prev == "minus1") & (phase == "pre"), (prev == "minus1") & dm),
        ("disc-para retry - pre-disc retry hdrs [matched after -1]",
         (prev == "minus1") & (phase == "pre"),
         (prev == "minus1") & (phase == "disc") & ~dm),
        ("post hdrs - pre first-attempt hdrs    [matched after Par]",
         (prev == "paragraph") & (phase == "pre"),
         (prev == "paragraph") & (phase == "post")),
        ("post hdrs - DISCOVERY hdr             [requested; prev MISMATCHED]",
         (prev == "minus1") & dm, (prev == "paragraph") & (phase == "post")),
    ]
    for name, ma, mb in contrasts:
        diff = paired(conv[ma].tolist(), ho[ma], conv[mb].tolist(), ho[mb])
        mu, t, k = tstat(diff)
        print(f"    {name}\n      d={mu:+.4f}  t={t:+6.1f}  n={k} convs")

    # layer profile of the two matched contrasts
    print("\n  layer profile (held-out d, paired by conv):")
    print(f"  {'layer':>5} {'disc-pre (after -1)':>21} {'post-pre (after Par)':>21}")
    for l in range(0, d["hdr_split"].shape[2]):
        hol = (d["hdr_split"][:, :, l] * held).sum(axis=1) / np.maximum(cnt, 1)
        hol[cnt == 0] = np.nan
        out = []
        for ma, mb in [((prev == "minus1") & (phase == "pre"), (prev == "minus1") & dm),
                       ((prev == "paragraph") & (phase == "pre"),
                        (prev == "paragraph") & (phase == "post"))]:
            diff = paired(conv[ma].tolist(), hol[ma], conv[mb].tolist(), hol[mb])
            mu, t, _ = tstat(diff)
            out.append(f"{mu:+.4f} (t={t:+5.1f})")
        print(f"  {l:5d} {out[0]:>21} {out[1]:>21}")


if __name__ == "__main__":
    main()
