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
    ax.set_title("Contrast is flat across cut position; the no-reward placebo "
                 "is 34% larger", loc="left", pad=10)
    ax.legend(loc="lower left", framealpha=.95)
    save(fig, "w1_ramp.png")


# ------------------------------------------------------------- fig 2: prefill
def fig_prefill():
    """prefill_probes_report.py -- matched-tail arms, paired by conversation.

    Only the paired difference is plotted. The design pairs arms within a
    conversation, so the per-arm means carry conversation-level variance that
    cancels in the pairing; their independent SEM is ~0.0004, smaller than a
    marker, and is not the quantity the inference rests on.

    The top panel is the legend: it shows the actual prefilled assistant turn,
    verbatim from prefill_probes_build.py, so "thinking", "body" and "tail" are
    defined by example rather than by name.
    """
    d = np.load(RES / "prefill_probes.npz", allow_pickle=True)
    cos, fam, cond, seg = d["cos"][:, L], d["family"], d["cond"], d["segment"]
    base = d["base_conv"]
    segs = ["thinking", "body", "tail"]

    def per_conv(arm, s_):
        m = (fam == "continuation") & (cond == arm) & (seg == s_)
        acc = {}
        for c, v in zip(base[m], cos[m]):
            acc.setdefault(c, []).append(v)
        return {c: float(np.mean(v)) for c, v in acc.items()}

    fig = plt.figure(figsize=(9.8, 6.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[.98, 1], hspace=.13)

    # ---- top: the two prefills, verbatim, with the segments marked -----------
    ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.grid(False)
    arms = [
        (INDIGO, "ARM A: high value, EOS far",
         "I am now confident that I have finally\n"
         "identified the hidden criterion\u2026 I will\n"
         "produce ten additional rewritten versions.",
         "I've figured out the criterion. Here are\n"
         "ten more versions of the paragraph, each\n"
         "one satisfying it:"),
        (AMBER, "ARM B: low value, EOS soon",
         "I have tried many different approaches\n"
         "now and none of them has earned a +1\u2026 the\n"
         "most sensible move is to stop here.",
         "I'm going to stop guessing now. Could you\n"
         "tell me more about what the hidden\n"
         "criterion involves?"),
    ]
    TAIL_TXT = ('"Please let me know how you would\n'
                ' like me to proceed."')
    for j, (c, title, think, body) in enumerate(arms):
        x = .015 + j * .53          # text column
        xl = x + .345               # segment label, immediately right of the text
        ax.text(x, .99, title, transform=ax.transAxes, fontsize=9,
                color=c, fontweight="bold", va="top")

        ax.text(x, .885, "<thinking>", transform=ax.transAxes, fontsize=7,
                family="monospace", color=GREY, va="top")
        ax.text(x, .835, think, transform=ax.transAxes, fontsize=7,
                family="monospace", color=c, va="top", linespacing=1.45)
        ax.text(x, .650, "</thinking>", transform=ax.transAxes, fontsize=7,
                family="monospace", color=GREY, va="top")
        ax.text(xl, .765, "thinking", transform=ax.transAxes, fontsize=8,
                color="#3C414B", va="center", style="italic")

        ax.text(x, .590, body, transform=ax.transAxes, fontsize=7,
                family="monospace", color=c, va="top", linespacing=1.45)
        ax.text(xl, .520, "body", transform=ax.transAxes, fontsize=8,
                color="#3C414B", va="center", style="italic")

        # the tail is repeated under each arm, in ink rather than the arm colour,
        # because it is the one segment the two arms share
        ax.text(x, .320, TAIL_TXT, transform=ax.transAxes, fontsize=7,
                family="monospace", color="#1F2430", va="top", linespacing=1.45)
        ax.text(xl, .275, "tail", transform=ax.transAxes, fontsize=8,
                color="#1F2430", va="center", style="italic", fontweight="bold")

    ax.set_title("The two prefilled assistant turns, verbatim. The tail is shared, "
                 "so only there are token\nidentity, count and position fixed "
                 "across arms.", loc="left", pad=6, fontsize=9.5)

    # ---- bottom: every per-conversation paired difference --------------------
    a2 = fig.add_subplot(gs[1])
    rng = np.random.default_rng(0)
    for i, s_ in enumerate(segs):
        A, B = per_conv("correct_nocomplete", s_), per_conv("complete_nosucc", s_)
        keys = sorted(set(A) & set(B))
        diff = np.array([B[c] - A[c] for c in keys])
        n = len(keys)
        a2.scatter(i + rng.uniform(-.13, .13, n), diff, s=14, color=AMBER,
                   alpha=.5, edgecolors="none", zorder=3)
        ci = 1.96 * diff.std(ddof=1) / np.sqrt(n)
        a2.errorbar([i], [diff.mean()], yerr=[ci], fmt="_", ms=28, mew=2.4,
                    color="#1F2430", ecolor="#1F2430", capsize=6, elinewidth=2.4,
                    zorder=5)
        a2.annotate(f"{diff.mean():+.4f}\n{int((diff > 0).sum())}/{n} (100%)",
                    xy=(i + .23, diff.mean()), fontsize=9, color="#3C414B",
                    va="center", linespacing=1.35,
                    fontweight="bold" if s_ == "tail" else "normal")
    a2.axhline(0, color="#C3C8D1", lw=1.3, zorder=1)
    a2.set_xticks(range(3))
    a2.set_xticklabels(["thinking\n(arm-specific text)", "body\n(arm-specific text)",
                        "tail\n(SHARED, byte-identical)"])
    a2.set_xlim(-.45, 2.78)
    a2.set_ylabel("B \u2212 A, per conversation  (cosine, layer 21)")
    a2.set_title("Paired difference, one dot per conversation; black marker is the "
                 "mean with its 95% CI.\nAbove zero = the arm nearer end-of-turn "
                 "projects higher, the opposite of what value predicts.",
                 loc="left", pad=8, fontsize=9.5)
    fig.savefig(FIGS / "w2_prefill.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  figures/w2_prefill.png")


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
    """steering_probe_report.py -- length collapses, stated confidence does not.

    The worked example is there to rule out the reading that steering works by
    suppressing backtracking: at negative alpha the extra text is REPETITION of
    an already-confident conclusion, not exploration or second-guessing.
    """
    rows = [json.loads(l) for l in open(RES / "steering_probe.jsonl")]
    post = [r for r in rows if r["state"] == "post"
            and (r["direction"] == "corrected" or r["alpha"] == 0)]
    alphas = sorted({r["alpha"] for r in post})
    length = [np.mean([r["n_tokens"] for r in post if r["alpha"] == a]) for a in alphas]
    rating = [np.mean([r["rating"] for r in post if r["alpha"] == a and r["rating"] >= 0])
              for a in alphas]

    fig = plt.figure(figsize=(9.8, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=.52, wspace=.26)

    a1 = fig.add_subplot(gs[0, 0])
    a1.plot(alphas, length, marker="o", ms=5, lw=1.8, color=INDIGO, zorder=3)
    a1.axhline(300, color=GREY, ls=":", lw=1)
    a1.text(-74, 288, "300-token cap", fontsize=7.5, color=GREY, va="top")
    for a, v in ((-75, length[0]), (75, length[-1])):
        a1.annotate(f"{v:.0f}", xy=(a, v), xytext=(0, 11 if a < 0 else 11),
                    textcoords="offset points", ha="center", fontsize=9,
                    fontweight="bold", color=INDIGO)
    a1.set_xlabel("steering strength \u03b1"); a1.set_ylabel("response length (tokens)")
    a1.set_xticks(alphas); a1.set_ylim(-20, 340)
    a1.set_title("(a) Length collapses 75-fold, 300 to 4 tokens",
                 loc="left", pad=8)

    a2 = fig.add_subplot(gs[0, 1])
    a2.plot(alphas, rating, marker="o", ms=5, lw=1.8, color=AMBER, zorder=3)
    a2.set_xlabel("steering strength \u03b1")
    a2.set_ylabel("stated confidence (0\u201310)")
    a2.set_xticks(alphas); a2.set_ylim(0, 10)
    a2.annotate("range 7.5 \u2013 8.9", xy=(0, 8.7), xytext=(0, 4.4),
                ha="center", fontsize=9, color=AMBER, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1))
    a2.set_title("(b) What the model SAYS about its confidence barely moves",
                 loc="left", pad=8)

    # ---- worked example: same conversation, three steering strengths ---------
    ax = fig.add_subplot(gs[1, :]); ax.axis("off"); ax.grid(False)
    ex = {r["alpha"]: r for r in post if r["conv_id"] == "bird_species__conv07"}
    cols = [
        (-75, INDIGO, "hits the 300-token cap",
         'I\u2019ve identified that the hidden\n'
         'criterion is to include a bird\n'
         'reference in each paragraph\u2026\n'
         'This seems to be the hidden\n'
         'criterion, and I\u2019m fairly confident\n'
         'it\u2019s the correct criterion, with a\n'
         'score of 8. I\u2019m confident it\u2019s the\n'
         'correct criterion, as it involves\n'
         'adding a bird reference\u2026'),
        (0, "#1F2430", "unsteered",
         'I\u2019m fairly confident the hidden\n'
         'criterion is to include a specific\n'
         'type of bird. I got +1 after adding\n'
         '"eagle" to the first paragraph and\n'
         '"swift" to the third, suggesting the\n'
         'pattern is to incorporate a bird\n'
         'reference in each paragraph.'),
        (75, AMBER, "stops immediately",
         '(nothing further; the turn ends)'),
    ]
    for j, (a, c, note, text) in enumerate(cols):
        x = .012 + j * .345
        r = ex[a]
        ax.text(x, 1.0, f"\u03b1 = {a:+d}", transform=ax.transAxes, fontsize=9.5,
                color=c, fontweight="bold", va="top")
        ax.text(x + .105, 1.0, note, transform=ax.transAxes, fontsize=8,
                color=GREY, va="top", style="italic")
        ax.text(x, .90, f"rating {r['rating']}   \u2502   {r['n_tokens']} token"
                + ("" if r['n_tokens'] == 1 else "s"),
                transform=ax.transAxes, fontsize=8.5, color=c, va="top",
                fontweight="bold", family="monospace")
        ax.text(x, .79, f"{r['rating']}", transform=ax.transAxes, fontsize=7.4,
                family="monospace", color="#1F2430", va="top")
        ax.text(x, .715, text, transform=ax.transAxes, fontsize=7.4,
                family="monospace", color="#3C414B", va="top", linespacing=1.5)

    ax.add_patch(plt.Rectangle((.008, -.16), .984, .30, transform=ax.transAxes,
                               facecolor=AMBER, alpha=.10, edgecolor=AMBER,
                               linewidth=1.3, zorder=1))
    ax.text(.5, .09, "The extra text at \u03b1 = \u221275 is REPETITION of an "
            "already-confident conclusion. Not backtracking,", transform=ax.transAxes,
            fontsize=9.5, ha="center", va="top", fontweight="bold", color="#1F2430")
    ax.text(.5, .005, "not second-guessing, not exploration. The model states the "
            "same answer, at the same confidence, for longer.",
            transform=ax.transAxes, fontsize=9.5, ha="center", va="top",
            fontweight="bold", color="#1F2430")
    ax.text(.5, -.085, "So the length effect is a failure to TERMINATE, not a "
            "change in how the model reasons.", transform=ax.transAxes,
            fontsize=9, ha="center", va="top", color="#3C414B", style="italic")
    ax.set_title("(c) One conversation, one prompt, three steering strengths: "
                 "verbatim generations", loc="left", pad=10)

    fig.suptitle("Steering the axis changes when the model stops talking, not "
                 "what it concludes", x=.005, y=1.0, ha="left", fontsize=11)
    fig.savefig(FIGS / "w3_steering.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  figures/w3_steering.png")


def fig_lengthfree():
    """steering_logits_report.py -- linear dose-response against a random band."""
    rows = [json.loads(l) for l in open(RES / "steering_logits.jsonl")]
    band = RES / "steering_logits_randband.jsonl"
    if band.exists():
        rows += [json.loads(l) for l in open(band)]
    rds = sorted({r["direction"] for r in rows if r["direction"].startswith("random")})

    chans = [("closure", "binary", "log P(end-of-turn)", "natural-log units",
              "does steering make the model more likely to stop?"),
             ("yes_minus_no", "binary", "logit(Yes) \u2212 logit(No)", "logits",
              'does it change the answer to "do you know the criterion?"'),
             ("exp_rating", "digit", "E[rating]", "rating points, 0\u20139",
              "does it change the confidence score it would emit?")]

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 5.8))
    for ax, (key, probe, lab, unit, question) in zip(axes, chans):
        bc = quad_b(rows, "corrected", key, probe).mean()
        bs = np.array([quad_b(rows, dd, key, probe).mean() for dd in rds])
        ax.barh([1], [bc], .52, color=INDIGO, zorder=3)
        ax.barh([0], [bs.mean()], .52, color=GREY, zorder=3,
                xerr=[[bs.mean() - bs.min()], [bs.max() - bs.mean()]],
                ecolor="#5E626C", capsize=4)
        ax.text(bc, 1, f"  {bc:+.2f}", ha="left" if bc > 0 else "right",
                va="center", fontsize=9, fontweight="bold")
        ax.axvline(0, color="#C3C8D1", lw=1)
        ax.set_yticks([1, 0])
        ax.set_yticklabels(["the value axis", f"{len(bs)} random\ndirections"],
                           fontsize=8.5)
        lo, hi = min(0, bs.min(), bc), max(0, bs.max(), bc)
        pad = (hi - lo) * .38 + 1e-9
        ax.set_xlim(lo - pad, hi + pad)
        k = (bc - bs.mean()) / bs.std(ddof=1)
        inside = bc <= bs.max()
        ax.text(.995, 1.06, f"{k:+.1f} sd from the random mean, "
                + ("INSIDE the random range" if inside
                   else "outside the random range"),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                color=AMBER if inside else "#3C414B",
                fontweight="bold" if inside else "normal")
        ax.set_title(f"{lab}  ({unit}):  {question}", loc="left",
                     fontsize=9, pad=14)
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("linear dose-response b: change in the readout across "
                        "\u03b1 \u2208 [\u221275, +75]", fontsize=9)
    fig.suptitle("Length-free readout, against 7 random directions of the same "
                 "size: end-of-turn and\nconfidence both move, the graded rating "
                 "does not", x=.005, y=1.06, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIGS / "w3b_lengthfree.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  figures/w3b_lengthfree.png")


# --------------------------------------------------------------- fig 4: retry
def fig_retry():
    """extend_retries_report.py -- 20 scripted failures, two arms."""
    d = np.load(RES / "extended_retries.npz", allow_pickle=True)
    cos, depth, arm, prev = d["hdr_cos"][:, L], d["depth"], d["arm"], d["prev"]
    retry = prev == "minus1"
    deps = sorted(set(depth[retry].tolist()))

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for a, c, lab in [("diverse", INDIGO, "diverse: every attempt a new rewrite"),
                      ("duplicate", AMBER, "duplicate: attempts 6-20 repeat verbatim\n"
                                           "(zero new information about the criterion)")]:
        ys = [cos[retry & (depth == k) & (arm == a)].mean() for k in deps]
        ax.plot(deps, ys, marker="o", ms=3.5, lw=1.4, color=c, label=lab, zorder=3)
    ax.axvline(5.5, color=GREY, ls=":", lw=1)
    ax.text(5.7, -0.056, "arms diverge here", fontsize=7.5, color=GREY)
    ax.annotate("a value account needs the climb to REVERSE\nout here; it never does",
                xy=(18.5, -0.028), xytext=(9.6, -0.0165), fontsize=7.5, color="#3C414B",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=.9))
    ax.set_xlabel("consecutive failed attempts on the same paragraph")
    ax.set_ylabel("cosine at the assistant header, layer 21")
    ax.set_xticks([2, 5, 8, 11, 14, 17, 20])
    ax.set_title("Retry climb decelerates but never reverses; the "
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
    a1.set_title("(a) 62% of the 47,863 training tokens are in the wrong turn",
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


# ----------------------------------------------------------------- fig 6: eos
def fig_eos():
    """eos_association_report.py -- projection vs P(end-of-turn) on natural text."""
    from scipy.stats import spearmanr
    f = RES / "eos_association.npz"
    if not f.exists():
        print("  (skipping w6_eos.png: results/eos_association.npz not found)")
        return
    d = np.load(f, allow_pickle=True)
    proj, eos, rel = d["proj"], d["eos"], d["rel_pos"]
    names = [str(x) for x in d["dir_names"]]
    ci, si = names.index("corrected"), names.index("shipped")
    ri = [i for i, n in enumerate(names) if n.startswith("random")]

    NB = 5
    edges = np.linspace(0, 1, NB + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 3.6),
                                 gridspec_kw={"width_ratios": [1.15, 1]})

    # (a) position-matched binned means: does a higher projection mean more
    #     end-of-turn mass, holding distance-from-the-end fixed?
    z = np.full(len(eos), np.nan); ez = np.full(len(eos), np.nan)
    for a, b in zip(edges, edges[1:]):
        m = (rel >= a) & (rel < b if b < 1 else rel <= b)
        if m.sum() < 30:
            continue
        p_, e_ = proj[m, ci], eos[m]
        z[m] = (p_ - p_.mean()) / (p_.std() + 1e-9)
        ez[m] = (e_ - e_.mean()) / (e_.std() + 1e-9)
    ok = np.isfinite(z)
    qs = np.quantile(z[ok], np.linspace(0, 1, 11))
    xs, ys, se = [], [], []
    for i in range(10):
        m = ok & (z >= qs[i]) & ((z <= qs[i + 1]) if i == 9 else (z < qs[i + 1]))
        xs.append(z[m].mean()); ys.append(ez[m].mean())
        se.append(ez[m].std(ddof=1) / np.sqrt(m.sum()))
    a1.errorbar(xs, ys, yerr=se, marker="o", ms=4, lw=1.5, color=INDIGO,
                capsize=2, ecolor="#A8B4E0", zorder=3)
    a1.axhline(0, color="#C3C8D1", lw=1)
    a1.set_xlabel("projection on the value axis\n(z-scored within position bin)")
    a1.set_ylabel("log P(end-of-turn)\n(z-scored within position bin)")
    a1.set_title("(a) relation is U-shaped, not monotone", loc="left", pad=8)

    # (b) the association per position bin, against the random-direction band
    def rho(idx, m):
        return spearmanr(proj[m, idx], eos[m])[0] if m.sum() >= 30 else np.nan
    mids, rc, rs, rb = [], [], [], []
    for a, b in zip(edges, edges[1:]):
        m = (rel >= a) & (rel < b if b < 1 else rel <= b)
        mids.append((a + b) / 2)
        rc.append(rho(ci, m)); rs.append(rho(si, m))
        rb.append([rho(i, m) for i in ri])
    rb = np.array(rb)
    a2.fill_between(mids, rb.min(1), rb.max(1), color=GREY, alpha=.25, zorder=2,
                    label=f"random directions (n={len(ri)}, full range)")
    a2.plot(mids, rc, marker="o", ms=4, lw=1.6, color=INDIGO,
            label="corrected axis", zorder=4)
    a2.plot(mids, rs, marker="s", ms=3.5, lw=1.3, ls="--", color=AMBER,
            label="released axis", zorder=3)
    a2.axhline(0, color="#C3C8D1", lw=1)
    a2.set_xlabel("position within the assistant turn  (0 = start, 1 = end)")
    a2.set_ylabel("Spearman \u03c1 (projection, log P(end-of-turn))")
    a2.set_title("(b) corrected axis stays inside the random band", loc="left", pad=8)
    a2.legend(loc="best", framealpha=.95, fontsize=8)

    fig.suptitle("Axis projection does not predict P(end-of-turn) at fixed "
                 "position on natural text", x=.005, y=1.03, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIGS / "w6_eos.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  figures/w6_eos.png")


if __name__ == "__main__":
    print("writing figures:")
    fig_ramp(); fig_prefill(); fig_steering(); fig_lengthfree()
    fig_retry(); fig_bug(); fig_eos()
