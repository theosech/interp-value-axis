# What is the value axis a value function *of*?

A replication of **The Value Axis**, Jiang, Kauvar & Lindsey,
[arXiv 2606.17056](https://arxiv.org/abs/2606.17056),
code [nickjiang2378/value-axis](https://github.com/nickjiang2378/value-axis), on
Qwen3-8B, plus evidence that on the paper's own construction corpus the direction
dominantly tracks **proximity to episode closure** rather than how well the model
is doing.

**Start with [WRITEUP.md](WRITEUP.md).** That is the argument, with every number
and its provenance. This file is how to run it.

Two results in one line each:

- **A span-localization bug** in the released construction code puts 62% of the
  axis's training tokens in the wrong conversational turn. Fixing it makes the
  paper's own effect *stronger*: held-out AUROC 0.850 → 0.880.
- **The construction contrast is confounded with position in the response**, and
  on byte-identical text a "give up" prefill projects *above* a "found it, more
  to go" prefill, the sign a completion signal predicts and a value signal does
  not.

---

## Setup

```bash
git clone https://github.com/theosech/interp-value-axis.git
cd interp-value-axis
uv sync
```

The upstream repo is used, not reimplemented, and is not vendored here. Clone it
alongside and pin it, or line numbers in the write-up will not match:

```bash
git clone https://github.com/nickjiang2378/value-axis.git
cd value-axis && git checkout 44ad182 && cd ..
```

GPU steps run on [Modal](https://modal.com) against `Qwen/Qwen3-8B` on an A10G:

```bash
uv run modal setup
```

A Jupyter kernel `value-axis-adapt (.venv)` is registered for the notebooks.

---

## Layout

```
WRITEUP.md              The argument. Read this first.
modal_app.py            Every GPU step: projections, axis rebuild, steering.
corrected_spans.py      The bug fix, structural span location.
turns.py                Structural conversation parser everything relies on.
results/MANIFEST.md     What every artifact is and how to regenerate it.
replication.ipynb       Phase 1 replication, and the alignment audit (cell D3).
lucky_vs_earned.ipynb   Attempt-level exploratory tables and EDA.
value-axis/             Upstream, cloned (gitignored, see Setup).
```

Most of `results/` is gitignored: it is 1.6 GB of activation projections.
`results/MANIFEST.md` documents every file and the command that regenerates it.
Small, expensive-to-regenerate artifacts (the corrected axis, the LLM-judged
labels, the steering outputs) are committed.

---

## Reproducing

### CPU only, seconds

```bash
uv run python corrected_axis_report.py        # replication gates, corrected vs shipped
uv run python corrected_mean_validation.py    # the paper's mean-level metric
uv run python ramp_cut_invariance.py          # the ramp / placebo result
uv run python attempt_split_report.py         # within-attempt cells, non-habituation
uv run python assistant_headers_report.py     # retry-depth climb
uv run python extend_retries_report.py        # extended-retry falsifications
uv run python prefill_probes_report.py        # matched-token prefills
uv run python prefill_rephrase_report.py      # phrasing robustness
uv run python steering_probe_report.py        # causal dissociation
uv run python steering_logits_report.py       # length-free readout
uv run python confidence_correlation.py       # position confounds
uv run python check_corrected_labels.py       # label validation (minutes)
```

These read `results/`. If you do not have the artifacts, regenerate them first.

### GPU, on Modal

Order matters: the axis rebuild produces the input everything else projects onto.

```bash
uv run modal run modal_app.py::rebuild_axis_main        # corrected axis
uv run modal run modal_app.py::attempt_split_main       # within-attempt splits
uv run modal run modal_app.py::assistant_headers_main   # header spans (1.28 GB out)
uv run modal run modal_app.py::attempts_main --corrected
uv run modal run modal_app.py::steering_probe_main      # generation steering
uv run modal run modal_app.py::steering_logits_main     # length-free steering
uv run modal run modal_app.py::logit_lens_main --corrected   # the one to cite
uv run modal run modal_app.py::logit_lens_main               # shipped axis, for comparison
```

Two need a local build step first:

```bash
uv run python prefill_probes_build.py   && uv run modal run modal_app.py::prefill_probes_main
uv run python prefill_rephrase_build.py && uv run modal run modal_app.py::prefill_rephrase_main
uv run python extend_retries_build.py   && uv run modal run modal_app.py::extended_retries_main
```

The LLM-judged label files are committed, so `judge_*.py` / `merge_labels.py`
only need rerunning if you want to regenerate them (they cost API calls).

---

## Three things to know before you touch this

**1. Layer indexing.** `value_axis.npy` is `(37, 4096)`, `num_hidden_layers + 1`,
because index 0 is the embedding output. Index 21, the paper's layer, is
therefore the output of transformer block 20. Use `hidden_states[21]` and stay
consistent.

**2. The repo's CPU-only AUROC is not Figure 2a.**
`construction/compute_vector.py`'s `evaluate_heldout_auroc` projects *one*
before-mean and *one* after-mean per held-out reward function and calls
`roc_auc_score` on two points, which is 1.0 exactly when `after > before`. It
saturates at ≥ 0.98 on 34 of 37 layers and its argmax is layer 2, not 21. The
paper's stated task is classifying paragraph *tokens*, which needs forward
passes. That is why `modal_app.py` exists, and why the token-level AUROC is the
replication target here.

**3. Some artifacts predate the bug fix.** Anything in `results/` marked
superseded in the manifest, `projections.npz`, `attempts.npz`,
`discovery_tokens.npz`, `delta_projection.csv`, carries
cosines against the **shipped** axis. Re-project against
`results/value_axis_corrected.npy` before using any number from them. The one
thing `projections.npz` is still exactly right for is the alignment audit in
`replication.ipynb` cell D3, because it carries per-token strings.

---

## Units

Every projection is a cosine against the unit axis at a given layer, the paper's
Eq. 2 metric. Anchors for judging any of them:

| Quantity | Value |
|---|---|
| Full before/after dynamic range at L21 | 0.165 |
| Random-direction null (cell differences) | ≤ 0.001 |
| Within-cell sd at headers | ≈ 0.015 |
| The paper's own headline behavioral effects | 0.02 – 0.04 |

"Held-out" always means **function-held-out**, using the paper's exact split
seeding (`Random(si*42)`, 35/13, ten splits), not held out over conversations.
