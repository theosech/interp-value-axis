"""Retry-depth climb vs positional drift (evidence for the 4.9x claim).

Paired consecutive header deltas within (conv, paragraph); per-token slope of
within-paragraph retry steps vs across-paragraph first-attempt-header drift;
residualization against the cross-paragraph position trend; layer profile.
Header positions reconstructed via attempts_corrected.npz fb_token_pos anchors.

Usage: python depth_check.py
"""

import json
import numpy as np

RES = "results"
h = np.load(f"{RES}/assistant_headers.npz", allow_pickle=True)
tf = json.load(open(f"{RES}/assistant_headers_split_test_fns.json"))
n = len(h["conv_id"]); L = 21
held = np.zeros((n, len(tf)), bool)
for si, fns in enumerate(tf):
    held[:, si] = np.isin(h["reward_fn"], fns)
cnt = held.sum(1).astype(float)

def ho_at(l):
    x = (h["hdr_split"][:, :, l] * held).sum(1) / np.maximum(cnt, 1)
    x[cnt == 0] = np.nan
    return x

ho = ho_at(L)
conv, para, att, prev = h["conv_id"], h["paragraph"], h["attempt_in_paragraph"], h["prev"]

a = np.load(f"{RES}/attempts_corrected.npz", allow_pickle=True)
pos_by_key = {}
for i in range(len(a["conv_id"])):
    k = (a["conv_id"][i], int(a["paragraph"][i]), int(a["attempt_in_paragraph"][i]))
    pos_by_key[k] = int(a["fb_token_pos"][i]) - int(a["n_asst_tokens"][i]) - 5
pos = np.array([pos_by_key.get((conv[i], int(para[i]), int(att[i])), np.nan)
                for i in range(n)], float)

m_retry = (prev == "minus1") & ~np.isnan(pos)
m_first = (prev == "paragraph") & ~np.isnan(pos)
print(f"headers with reconstructed position: retry {m_retry.sum()}, first {m_first.sum()}")

print(f"\n1. LEVEL BY RETRY DEPTH (held-out corrected @L{L})")
for a2 in (2, 3, 4, 5):
    m = m_retry & (att == a2)
    print(f"  att={a2}: mean={np.nanmean(ho[m]):+.4f}  sd={np.nanstd(ho[m]):.4f}  n={m.sum()}")

idx = {}
for i in range(n):
    if prev[i] == "minus1":
        idx[(conv[i], int(para[i]), int(att[i]))] = i

print("\n   paired consecutive deltas (same conv, same paragraph):")
allpairs = []
for s0, s1 in ((2, 3), (3, 4), (4, 5)):
    d, dp_ = [], []
    for (c, p, a2), i in idx.items():
        if a2 != s0:
            continue
        j = idx.get((c, p, s1))
        if j is not None and not np.isnan(ho[i]) and not np.isnan(ho[j]) \
           and not np.isnan(pos[i]) and not np.isnan(pos[j]):
            d.append(ho[j] - ho[i]); dp_.append(pos[j] - pos[i])
    d = np.array(d); allpairs += list(zip(d, dp_))
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"   att {s0}->{s1}: mean d={d.mean():+.4f}  t={d.mean()/se:+.1f}  n={len(d)}  "
          f"({np.mean(dp_):.0f} tokens apart)")

d, dp_ = zip(*allpairs)
slope_within = np.sum(d) / np.sum(dp_)
first_idx = {}
for i in range(n):
    if prev[i] == "paragraph" and not np.isnan(pos[i]):
        first_idx[(conv[i], int(para[i]))] = i

def across(minpara):
    dd, pp = [], []
    for (c, p), i in first_idx.items():
        if p < minpara:
            continue
        j = first_idx.get((c, p + 1))
        if j is not None and not np.isnan(ho[i]) and not np.isnan(ho[j]):
            dd.append(ho[j] - ho[i]); pp.append(pos[j] - pos[i])
    return np.array(dd), np.array(pp)

dd, pp = across(1)
dd2, pp2 = across(2)
print(f"\n2. PER-TOKEN SLOPES (delta cosine per 1000 tokens)")
print(f"   within-paragraph retry steps : {slope_within*1000:+.4f}  (n={len(allpairs)})")
print(f"   across paragraphs (att-1)    : {np.sum(dd)/np.sum(pp)*1000:+.4f}  (n={len(dd)})")
print(f"   across paragraphs, excl p1   : {np.sum(dd2)/np.sum(pp2)*1000:+.4f}  (n={len(dd2)})")
print(f"   ratio within / across(excl1) : "
      f"{slope_within/(np.sum(dd2)/np.sum(pp2)):.1f}x")

mf = m_first & (para > 1)
A = np.vstack([pos[mf], np.ones(mf.sum())]).T
coef, *_ = np.linalg.lstsq(A, ho[mf], rcond=None)
resid = ho - (coef[0] * pos + coef[1])
print(f"\n3. RETRY RESIDUALS after removing cross-paragraph position trend "
      f"({coef[0]*1000:+.4f}/1k tok)")
for a2 in (2, 3, 4, 5):
    m = m_retry & (att == a2)
    v = resid[m]; v = v[~np.isnan(v)]
    print(f"   att={a2}: mean resid={v.mean():+.4f}  sem={v.std(ddof=1)/np.sqrt(len(v)):.4f}  n={len(v)}")

print(f"\n4. LAYER PROFILE: paired (att3 - att2) within paragraph")
for l in (5, 10, 14, 18, 21, 24, 27, 30, 33, 36):
    hol = ho_at(l); d = []
    for (c, p, a2), i in idx.items():
        if a2 != 2:
            continue
        j = idx.get((c, p, 3))
        if j is not None and not np.isnan(hol[i]) and not np.isnan(hol[j]):
            d.append(hol[j] - hol[i])
    d = np.array(d); se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"   L{l:>2}: d={d.mean():+.4f}  t={d.mean()/se:+.1f}")
