# results/ manifest

Most of this directory is gitignored — 1.6 GB of activation projections. Every file
is listed here with the command that regenerates it, so the repo is reproducible
without shipping the artifacts.

Anything marked **in git** is small and expensive to regenerate (LLM-judged labels,
the corrected axis itself), so it is committed.

All GPU steps run on Modal against `Qwen/Qwen3-8B`, A10G, and need
`modal run` plus a Hugging Face cache volume — see the README.

## The corrected axis and its inputs

| File | Size | Produced by | Notes |
|---|---|---|---|
| `value_axis_corrected.npy` | 1.2 MB | `modal run modal_app.py::rebuild_axis_main` | **in git.** `(37, 4096)`, the deliverable of the bug fix. Index 0 is the embedding output, so index 21 is the output of transformer block 20. |
| `activation_means_corrected.pt` | 58 MB | same | Per-reward-function before/after mean activations under corrected labels. The direct analogue of upstream `activation_means.pt`. |
| `corrected_split_test_fns.json` | 2.5 KB | same | **in git.** The 10 function-held-out splits, seeded exactly as upstream (`Random(si*42)`, 35/13). Byte-identical to `split_test_fns.json`. |
| `projections_corrected.npz` | 81 MB | same | Per-token projections of the corrected labels onto the corrected axis. Input to `corrected_axis_report.py`. |

## Observational battery

| File | Size | Produced by | Notes |
|---|---|---|---|
| `attempt_split.npz` | 11 MB | `modal run modal_app.py::attempt_split_main` | One row per (attempt × split channel), 4807 rows. Carries `n_before`/`n_split_tokens`/`n_after`, which is what makes the cut-invariance test possible. Input to `attempt_split_report.py` and `ramp_cut_invariance.py`. |
| `assistant_headers.npz` | 1.28 GB | `modal run modal_app.py::assistant_headers_main` | 4516 assistant headers, 7-token content-matched span. Includes raw per-header fp16 means, which is why it is large. Input to `assistant_headers_report.py`. |
| `attempts_corrected.npz` | 7.4 MB | `modal run modal_app.py::attempts_main --corrected` | Per-attempt mean cosines, corrected axis. Input to `rule_cells_report.py --npz results/attempts_corrected.npz`. |
| `extended_retries.{json,npz}` | 1.5 MB / 437 KB | `python extend_retries_build.py` then `modal run modal_app.py::extended_retries_main` | Scripted 20-failure corpus, two arms (diverse / duplicate), 35 paragraphs. |

## Prefill experiments (the matched-token evidence)

| File | Size | Produced by | Notes |
|---|---|---|---|
| `prefill_probes.{json,npz}` | 7.9 MB / 4.3 MB | `python prefill_probes_build.py` then `modal run modal_app.py::prefill_probes_main` | Continuation prefills + Yes/No probes. Report: `prefill_probes_report.py`. |
| `prefill_rephrase.{json,npz}` | 5.1 MB / 14 MB | `python prefill_rephrase_build.py` then `modal run modal_app.py::prefill_rephrase_main` | 3×3 phrasings × 2 tails robustness replication. Report: `prefill_rephrase_report.py`. |

## Steering (the causal evidence)

| File | Size | Produced by | Notes |
|---|---|---|---|
| `steering_probe.jsonl` | 270 KB | `modal run modal_app.py::steering_probe_main` | **in git.** 390 generations. Confidence-rating vs response-length dissociation. Report: `steering_probe_report.py`. |
| `steering_logits.jsonl` | small | `modal run modal_app.py::steering_logits_main` | **in git.** Length-free readout: one forward pass, logits at the answer position. Report: `steering_logits_report.py`. |

## LLM-judged labels (committed — regenerating these costs API calls)

| File | Size | Produced by | Notes |
|---|---|---|---|
| `believed_words.jsonl` | 1.2 MB | `python judge_believed_words.py` | **in git.** Per attempt, the word the model itself said it was targeting. Drives the placebo split. |
| `rule_labels.jsonl` | 2.0 MB | `python judge_rules.py` → `python merge_labels.py` | **in git.** Whether the model's stated rule belief was correct. Validated by `validate_labels.py`. |
| `reward_words.jsonl` | 227 KB | `python judge_reward_words.py` | **in git.** Semantic-criterion reward words per attempt. |

## Superseded — kept for provenance, do not cite

These were computed against the **shipped (buggy) axis**, before the label fix. Any
number lifted from them is a pre-correction number. Re-project against
`value_axis_corrected.npy` before use.

| File | Size | Why superseded |
|---|---|---|
| `projections.npz` | 75 MB | Shipped axis, upstream labels. Still used by `replication.ipynb` cell D3 for the alignment audit, which is the one thing it is *right* for. |
| `attempts.npz` | 7.4 MB | Shipped axis. Superseded by `attempts_corrected.npz`. |
| `discovery_tokens.npz` | 118 MB | Shipped axis. Superseded by `attempt_split.npz`. |
| `divider.npz`, `divider_believed.npz` | 5.3 / 9.3 MB | Superseded by `attempt_split.npz`, which covers both channels and adds held-out directions. |
| `delta_projection.csv` | 83 KB | Shipped-means delta analysis; superseded by `corrected_mean_validation.py`. |
| `attempt_table.csv`, `attempt_sequences.csv` | 1.8 MB / 33 KB | **in git.** Exploratory attempt-level tables behind `lucky_vs_earned.ipynb`. Shipped-axis projections. |
| `logit_lens.json` | 3.4 KB | **in git.** Shipped axis. The corrected-axis logit lens has never been run — a known gap. |

## Dead — outputs of a superseded 8-conversation smoke run

`post_discovery_means.pt`, `post_vs_discovery_axis.npy`. Replaced by the full
`assistant_headers` run. Safe to delete.

## Built but never analysed

`header_axis_disc_minus_pre.npy`, `header_axis_post_minus_pre.npy` (47 reward
functions each). Contrast directions built from the header spans. Nothing in the
write-up rests on them; they are a starting point for future header-vector work.
