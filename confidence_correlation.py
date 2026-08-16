"""Does codex-judged per-attempt confidence correlate with value-axis readings?

Confidence: the 1-5 rating codex assigned to each attempt's reasoning
(believed_words.jsonl). Two correlations, corrected axis:

1. ATTEMPT TOKENS: Eq. 2 mean-of-cos over the attempt's own assistant tokens
   (attempts_corrected.npz asst_cos) vs the confidence judged from that attempt.
2. NEXT HEADER: the content-matched header span of the FOLLOWING attempt
   (assistant_headers.npz, held-out splits available) vs the PREVIOUS attempt's
   confidence. Two variants:
     - retry headers (prev = -1 feedback): header of attempt k+1, confidence of k
     - paragraph-initial headers: header of paragraph p attempt 1, confidence of
       the last attempt of paragraph p-1 (split by whether it earned +1)

Spearman rho throughout (confidence is ordinal), overall and within cells, so
cell structure (phase x outcome) doesn't masquerade as a confidence signal.

Usage: python confidence_correlation.py [--layer 21]
"""

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"


def sp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 10:
        return float("nan"), float("nan"), int(m.sum())
    r, p = spearmanr(x[m], y[m])
    return r, p, int(m.sum())


def line(name, x, y):
    r, p, n = sp(x, y)
    star = " ***" if p < 1e-3 else " *" if p < 0.05 else ""
    print(f"  {name:<44} rho={r:+.3f}  p={p:.1e}  n={n}{star}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=21)
    args = ap.parse_args()
    L = args.layer

    conf, prev_reward = {}, {}
    for l in open(RES / "believed_words.jsonl"):
        if not l.strip():
            continue
        r = json.loads(l)
        conf[(r["conv_id"], r["paragraph"], r["attempt_in_paragraph"])] = r["confidence"]
    print("confidence distribution:",
          dict(sorted(collections.Counter(conf.values()).items())))

    # ---- 1. attempt tokens (Eq. 2 mean-of-cos, corrected axis) --------------
    a = np.load(RES / "attempts_corrected.npz", allow_pickle=True)
    cos = a["asst_cos"][:, L]
    keys = list(zip(a["conv_id"].tolist(), a["paragraph"].tolist(),
                    a["attempt_in_paragraph"].tolist()))
    c1 = np.array([conf.get(k, np.nan) for k in keys], float)
    cells = a["cell"]

    print(f"\n1. ATTEMPT TOKENS vs own confidence (corrected axis @L{L}, Eq. 2)")
    line("all attempts", c1, cos)
    for cl in ("pre_fail", "pre_lucky", "post_earned"):
        m = cells == cl
        line(f"  within {cl}", c1[m], cos[m])

    # ---- 2. next header vs previous attempt's confidence --------------------
    h = np.load(RES / "assistant_headers.npz", allow_pickle=True)
    test_fns = json.loads((RES / "assistant_headers_split_test_fns.json").read_text())
    n = len(h["conv_id"])
    held = np.zeros((n, len(test_fns)), dtype=bool)
    for si, fns in enumerate(test_fns):
        held[:, si] = np.isin(h["reward_fn"], fns)
    cnt = held.sum(axis=1).astype(float)
    ho = (h["hdr_split"][:, :, L] * held).sum(axis=1) / np.maximum(cnt, 1)
    ho[cnt == 0] = np.nan
    full = h["hdr_cos"][:, L]

    conv, para, att = h["conv_id"], h["paragraph"], h["attempt_in_paragraph"]
    prev = h["prev"]
    phase = h["para_phase"]

    # retry headers: previous attempt is (conv, para, att-1), which got -1
    m_retry = prev == "minus1"
    c_prev = np.array([conf.get((conv[i], int(para[i]), int(att[i]) - 1), np.nan)
                       for i in range(n)], float)

    print(f"\n2a. RETRY HEADER (after -1) vs PREVIOUS attempt's confidence @L{L}")
    line("all retry headers      [held-out]", c_prev[m_retry], ho[m_retry])
    line("all retry headers      [full axis]", c_prev[m_retry], full[m_retry])
    for ph in ("pre", "disc"):
        m = m_retry & (phase == ph)
        line(f"  within phase={ph}    [held-out]", c_prev[m], ho[m])

    # paragraph-initial headers: previous attempt = last attempt of para-1
    last_att = {}
    for k in conf:
        cid, p_, at_ = k
        last_att[(cid, p_)] = max(last_att.get((cid, p_), 0), at_)
    m_first = (prev == "paragraph") & (para > 1)
    c_prev_para = np.full(n, np.nan)
    for i in range(n):
        if not m_first[i]:
            continue
        k = (conv[i], int(para[i]) - 1)
        if k in last_att:
            c_prev_para[i] = conf.get((conv[i], int(para[i]) - 1, last_att[k]), np.nan)

    # was the previous paragraph's last attempt rewarded? (+1 ends a paragraph;
    # "Moving on" means it failed out)
    rewarded_prev = np.zeros(n, dtype=bool)
    bw_reward = {}
    for l in open(RES / "believed_words.jsonl"):
        if l.strip():
            r = json.loads(l)
            bw_reward[(r["conv_id"], r["paragraph"], r["attempt_in_paragraph"])] = r["reward"]
    for i in range(n):
        if m_first[i]:
            k = (conv[i], int(para[i]) - 1)
            if k in last_att:
                rewarded_prev[i] = bool(bw_reward.get(
                    (conv[i], int(para[i]) - 1, last_att[k]), False))

    print(f"\n2b. PARAGRAPH-INITIAL HEADER vs LAST attempt of previous paragraph @L{L}")
    line("prev paragraph ended +1   [held-out]", c_prev_para[m_first & rewarded_prev],
         ho[m_first & rewarded_prev])
    line("prev paragraph failed out [held-out]", c_prev_para[m_first & ~rewarded_prev],
         ho[m_first & ~rewarded_prev])

    # ---- layer scan for the two headline correlations -----------------------
    print("\nlayer scan (Spearman rho):")
    print(f"  {'layer':>5} {'attempt~own conf':>17} {'retry hdr~prev conf (ho)':>25}")
    for l in range(0, a["asst_cos"].shape[1], 2):
        r1, _, _ = sp(c1, a["asst_cos"][:, l])
        hol = (h["hdr_split"][:, :, l] * held).sum(axis=1) / np.maximum(cnt, 1)
        hol[cnt == 0] = np.nan
        r2, _, _ = sp(c_prev[m_retry], hol[m_retry])
        print(f"  {l:5d} {r1:>17.3f} {r2:>25.3f}")


if __name__ == "__main__":
    main()
