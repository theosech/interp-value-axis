"""Regenerate every figure in WRITEUP.md from results/.

Each figure replaces a table that was hard to read as numbers. Nothing here
computes a new result -- the reports remain the source of truth, and every value
plotted is reproducible from the script named in the figure's docstring.

Usage: python make_figures.py            # writes figures/w*.png
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RES, FIGS = ROOT / "results", ROOT / "figures"
FIGS.mkdir(exist_ok=True)
L = 21

INDIGO, AMBER, GREY = "#4A63C4", "#B0742A", "#8A8F99"
RANGE = 0.165          # full before/after dynamic range at L21, the scale anchor

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 170, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .18, "grid.linewidth": .7,
    "axes.labelsize": 9, "axes.titlesize": 9.5, "legend.fontsize": 8.5,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  figures/{name}")


def held_out(d, test_fns, layer=L):
    n = len(d["reward_fn"])
    held = np.zeros((n, len(test_fns)), dtype=bool)
    for si, fns in enumerate(test_fns):
        held[:, si] = np.isin(d["reward_fn"], fns)
    cnt = np.maximum(held.sum(1).astype(float), 1)
    bef = (d["before_split"][:, :, layer] * held).sum(1) / cnt
    aft = (d["after_split"][:, :, layer] * held).sum(1) / cnt
    return bef, aft, held.sum(1) > 0


# ---------------------------------------------------------------- fig 1: ramp
def fig_ramp():
    """ramp_cut_invariance.py -- contrast by cut position, criterion vs placebo."""
    d = np.load(RES / "attempt_split.npz", allow_pickle=True)
    tf = json.loads((RES / "attempt_split_test_fns.json").read_text())
    bef, aft, ok = held_out(d, tf)
    diff = aft - bef
    nb, ns, na = d["n_before"], d["n_split_tokens"], d["n_after"]
    frac = nb / np.maximum(nb + ns + na, 1)
    bins = [(.15, .25), (.25, .35), (.35, .45), (.45, .55),
            (.55, .65), (.65, .75), (.75, .85)]
    ch, rew = d["channel"], d["reward"] == "True"

    series = [("Criterion split, rewarded attempts", ok & (ch == "actual") & rew, INDIGO),
              ("Placebo: believed word, failed attempts", ok & (ch == "believed") & ~rew, AMBER)]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    w = 0.38
    x = np.arange(len(bins))
    for k, (lab, base, c) in enumerate(series):
        ys, ns_ = [], []
        for lo, hi in bins:
            m = base & (frac >= lo) & (frac < hi)
            ys.append(diff[m].mean()); ns_.append(m.sum())
        ax.bar(x + (k - .5) * w, ys, w, color=c, label=f"{lab}  (n={sum(ns_)})", zorder=3)
        for xi, y in zip(x, ys):
            ax.text(xi + (k - .5) * w, y + .004, f"{y:.3f}".lstrip("0"),
                    ha="center", fontsize=7, color="#3C414B")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lo:.2f}–{hi:.2f}" for lo, hi in bins])
    ax.set_xlabel("fraction of the rewrite lying before the cut")
    ax.set_ylabel("after − before   (cosine with the axis, layer 21)")
    ax.set_ylim(0, .27)
    # second scale: the same numbers as a share of the axis's full swing, so the
    # reader does not have to hold 0.165 in their head
    r = ax.secondary_yaxis("right", functions=(lambda v: v / RANGE * 100,
                                               lambda v: v * RANGE / 100))
    r.set_ylabel("as % of the axis's full before/after swing (0.165)")
    ax.set_title("The contrast does not depend on where you cut — and survives "
                 "where no reward was given", loc="left", pad=10)
    ax.legend(loc="lower left", framealpha=.95)
    save(fig, "w1_ramp.png")


# ------------------------------------------------------------- fig 2: prefill
def fig_prefill():
    """prefill_probes_report.py -- matched-tail arms, paired by conversation."""
    d = np.load(RES / "prefill_probes.npz", allow_pickle=True)
    cos, fam, cond, seg = d["cos"][:, L], d["family"], d["cond"], d["segment"]
    base = d["base_conv"]
    segs = [("thinking", "thinking\n(condition-specific text)"),
            ("body", "body\n(condition-specific text)"),
            ("tail", "tail\nBYTE-IDENTICAL")]

    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    for i, (s, lab) in enumerate(segs):
        va = cos[(fam == "continuation") & (cond == "correct_nocomplete") & (seg == s)].mean()
        vb = cos[(fam == "continuation") & (cond == "complete_nosucc") & (seg == s)].mean()
        ax.plot([i, i], [va, vb], color=GREY, lw=1.2, zorder=2)
        ax.scatter([i], [va], s=64, color=INDIGO, zorder=3,
                   label="“I found it — ten paragraphs to go”  (HIGH value)" if not i else None)
        ax.scatter([i], [vb], s=64, color=AMBER, zorder=3,
                   label="“I'll stop here and call it done”  (HIGH completion)" if not i else None)
        ax.annotate(f"{vb - va:+.4f}", xy=(i + .08, (va + vb) / 2), fontsize=8,
                    color="#3C414B", va="center")
    ax.set_xticks(range(len(segs)))
    ax.set_xticklabels([l for _, l in segs])
    ax.set_xlim(-.5, 2.6)
    ax.set_ylabel("cosine with the axis, layer 21")
    ax.set_title("Giving up projects ABOVE succeeding — including on tokens that "
                 "are byte-identical\nacross the two arms (n=65 conversations, "
                 "paired; random-direction control +0.0004)",
                 loc="left", pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=1,
              frameon=False, handletextpad=.5)
    save(fig, "w2_prefill.png")


# ------------------------------------------------------------ fig 3: steering
def quad_b(rows, dname, key, probe):
    """Linear term of a per-prefix fit value ~ a + b*alpha + c*alpha^2."""
    out = []
    for k in {(r["conv_id"], r["state"]) for r in rows}:
        pts = [(r["alpha"] / 75.0, r[key]) for r in rows
               if (r["conv_id"], r["state"]) == k and r["probe"] == probe
               and (r["direction"] == dname or r["alpha"] == 0)
               and key in r and np.isfinite(r[key])]
        if len(pts) >= 5:
            out.append(np.polyfit([p[0] for p in pts], [p[1] for p in pts], 2)[1])
    return np.array(out)


def fig_steering():
    """steering_probe_report.py + steering_logits_report.py."""
    rows = [json.loads(l) for l in open(RES / "steering_logits.jsonl")]
    band = RES / "steering_logits_randband.jsonl"
    if band.exists():
        rows += [json.loads(l) for l in open(band)]
    rds = sorted({r["direction"] for r in rows if r["direction"].startswith("random")})

    chans = [("closure", "binary", "log P(end-of-turn)", "log-probability"),
             ("yes_minus_no", "binary", "logit(Yes) \u2212 logit(No)", "logits"),
             ("exp_rating", "digit", "E[rating]", "rating points, 0\u20139")]

    fig = plt.figure(figsize=(9.8, 4.0))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.15], hspace=.85, wspace=.32)
    a1 = fig.add_subplot(gs[:, 0])

    # (a) generation readout: both rows are Spearman rho, so one axis is honest
    corr, ship = [-0.959, -0.160], [-0.783, +0.003]
    y = np.arange(2)
    a1.barh(y + .19, ship, .34, color=GREY, label="shipped axis", zorder=3)
    a1.barh(y - .19, corr, .34, color=INDIGO, label="corrected axis", zorder=3)
    for yy, v in zip(y - .19, corr):
        a1.text(v + .03, yy, f"{v:+.3f}", ha="left", va="center", fontsize=8)
    a1.axvline(0, color="#C3C8D1", lw=1)
    a1.set_yticks(y)
    a1.set_yticklabels(["response\nlength", "verbalized\nconfidence"])
    a1.set_xlim(-1.1, .55)
    a1.set_xlabel("Spearman \u03c1 with steering strength \u03b1,\nmean over prefixes")
    a1.set_title("(a) generation readout\nsteering changes how much it writes,\n"
                 "not what it claims", loc="left", pad=8)
    a1.legend(loc="upper left", frameon=False)

    # (b) length-free readout. Each channel has its OWN units -- log-probability,
    # logits, rating points -- so they get their own axes rather than one shared
    # scale that would make the comparison look like something it is not.
    for i, (key, probe, lab, unit) in enumerate(chans):
        ax = fig.add_subplot(gs[i, 1])
        bc = quad_b(rows, "corrected", key, probe).mean()
        bs = np.array([quad_b(rows, dd, key, probe).mean() for dd in rds])
        ax.barh([1], [bc], .5, color=INDIGO, zorder=3)
        if len(bs) > 1:
            ax.barh([0], [bs.mean()], .5, color=GREY, zorder=3,
                    xerr=[[bs.mean() - bs.min()], [bs.max() - bs.mean()]],
                    ecolor="#5E626C", capsize=3)
        else:
            ax.barh([0], [bs.mean()], .5, color=GREY, zorder=3)
        ax.text(bc, 1, f"  {bc:+.2f}", ha="left" if bc > 0 else "right",
                va="center", fontsize=8)
        ax.axvline(0, color="#C3C8D1", lw=1)
        ax.set_yticks([1, 0])
        ax.set_yticklabels(["value axis",
                            f"random\u00d7{len(bs)}" if len(bs) > 1 else "random\u00d71"],
                           fontsize=8)
        lo, hi = min(0, bs.min(), bc), max(0, bs.max(), bc)
        pad = (hi - lo) * .35 + 1e-9
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_title(f"{lab}   ({unit})", loc="left", fontsize=8.5, pad=4)
        if len(bs) > 1:                       # how far outside the random spread
            k = (bc - bs.mean()) / bs.std(ddof=1)
            ax.text(.99, .04, f"{k:+.1f} sd vs random", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7.5, color="#5E626C")
        ax.tick_params(labelsize=7.5)
        if i == len(chans) - 1:
            ax.set_xlabel("linear dose-response b over \u03b1 \u2208 [\u221275, +75]",
                          fontsize=8.5)

    fig.text(.545, 1.015, "(b) length-free readout \u2014 one forward pass, nothing "
             "generated", fontsize=9.5, ha="left")
    fig.suptitle("Steering changes how much the model writes; against a "
                 "random-direction band, the\nlength-free channels are far less "
                 "clear-cut", x=.005, y=1.16, ha="left", fontsize=10.5)
    fig.savefig(FIGS / "w3_steering.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  figures/w3_steering.png")


# --------------------------------------------------------------- fig 4: retry
def fig_retry():
    """extend_retries_report.py -- 20 scripted failures, two arms."""
    d = np.load(RES / "extended_retries.npz", allow_pickle=True)
    cos, depth, arm, prev = d["hdr_cos"][:, L], d["depth"], d["arm"], d["prev"]
    retry = prev == "minus1"
    deps = sorted(set(depth[retry].tolist()))

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for a, c, lab in [("diverse", INDIGO, "diverse — every attempt a new rewrite"),
                      ("duplicate", AMBER, "duplicate — attempts 6-20 repeat verbatim\n"
                                           "(zero new information about the criterion)")]:
        ys = [cos[retry & (depth == k) & (arm == a)].mean() for k in deps]
        ax.plot(deps, ys, marker="o", ms=3.5, lw=1.4, color=c, label=lab, zorder=3)
    ax.axvline(5.5, color=GREY, ls=":", lw=1)
    ax.text(5.7, -0.056, "arms diverge here", fontsize=7.5, color=GREY)
    ax.annotate("a value account needs the climb to REVERSE\nout here — it never does",
                xy=(18.5, -0.028), xytext=(9.6, -0.0165), fontsize=7.5, color="#3C414B",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=.9))
    ax.set_xlabel("consecutive failed attempts on the same paragraph")
    ax.set_ylabel("cosine at the assistant header, layer 21")
    ax.set_xticks([2, 5, 8, 11, 14, 17, 20])
    ax.set_title("The retry climb decelerates but never turns over, and the "
                 "zero-information arm sits higher\n(35 paragraphs per arm; the "
                 "header span is byte-identical throughout)", loc="left", pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.19), ncol=2,
              frameon=False, handletextpad=.5)
    save(fig, "w4_retry.png")


# ----------------------------------------------------------------- fig 5: bug
def fig_bug():
    """corrected_axis_report.py -- token provenance and the AUROC it cost."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.3),
                                 gridspec_kw={"width_ratios": [1, 1.3]})

    # (a) where the axis's training tokens actually came from (span audit)
    parts = [("assistant turns\n(as intended)", 17467, INDIGO),
             ("user turns", 29460, AMBER),
             ("chat markers", 936, GREY)]
    left = 0
    for lab, v, c in parts:
        a1.barh(0, v, .55, left=left, color=c, zorder=3)
        if v / 47863 > .05:
            a1.text(left + v / 2, 0, f"{v:,}\n{v / 47863:.0%}", ha="center",
                    va="center", fontsize=8.5, color="white", fontweight="bold")
            a1.text(left + v / 2, .42, lab, ha="center", va="bottom", fontsize=7.5,
                    color="#3C414B")
        else:                                    # too thin to label inside
            a1.text(left + v / 2, -.42, f"{lab}\n{v:,} ({v / 47863:.0%})",
                    ha="center", va="top", fontsize=7.5, color="#3C414B")
        left += v
    a1.set_xlim(0, 47863); a1.set_ylim(-.5, 1.0)
    a1.set_yticks([]); a1.set_xticks([])
    a1.grid(False)
    for sp in a1.spines.values():
        sp.set_visible(False)
    a1.set_title("(a) the 47,863 tokens the released axis was built from",
                 loc="left", pad=8)

    # (b) what fixing it did to the paper's own metric
    d = np.load(RES / "projections_corrected.npz", allow_pickle=True)
    from sklearn.metrics import roc_auc_score
    y = (d["label"].astype(str) == "after").astype(int)
    tf = json.loads((RES / "corrected_split_test_fns.json").read_text())
    rf = d["reward_fn"].astype(str)
    nl = d["cos_split"].shape[2]
    ho = np.full((len(tf), nl), np.nan)
    for s, fns in enumerate(tf):
        m = np.isin(rf, list(fns))
        if m.sum() and len(np.unique(y[m])) == 2:
            for l in range(nl):
                ho[s, l] = roc_auc_score(y[m], d["cos_split"][m, s, l])
    corr_auc = np.nanmean(ho, 0)
    ship_auc = np.array([roc_auc_score(y, d["cos_shipped"][:, l])
                         for l in range(d["cos_shipped"].shape[1])])
    a2.plot(range(nl), corr_auc, lw=1.6, color=INDIGO,
            label="corrected labels + corrected axis")
    a2.plot(range(len(ship_auc)), ship_auc, lw=1.4, ls="--", color=GREY,
            label="corrected labels + released axis")
    a2.axhline(.5, color="#C3C8D1", ls=":", lw=1)
    a2.axvline(L, color=AMBER, lw=1, ls="--")
    a2.text(L - .8, .955, "the paper's layer, 21", fontsize=7.5, color=AMBER,
            ha="right")
    a2.scatter([int(np.nanargmax(corr_auc))], [np.nanmax(corr_auc)], s=34,
               color=INDIGO, zorder=4)
    a2.annotate(f"peak {np.nanmax(corr_auc):.3f} @L{int(np.nanargmax(corr_auc))}",
                xy=(int(np.nanargmax(corr_auc)), np.nanmax(corr_auc)),
                xytext=(27.5, .935), fontsize=7.5, color="#3C414B",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=.9))
    a2.set_xlabel("hidden-state layer"); a2.set_ylabel("held-out token AUROC")
    a2.set_ylim(.45, 1.0)
    a2.set_title("(b) fixing it raised the paper's own metric, 0.850 → 0.880 @L21",
                 loc="left", pad=8)
    a2.legend(loc="lower center", bbox_to_anchor=(.5, -.42), frameon=False)
    save(fig, "w5_bug.png")


if __name__ == "__main__":
    print("writing figures:")
    fig_ramp(); fig_prefill(); fig_steering(); fig_retry(); fig_bug()
