# What is the value axis a value function *of*?

A replication of **The Value Axis** (Jiang, Kauvar & Lindsey, [arXiv 2606.17056](https://arxiv.org/abs/2606.17056))
on Qwen3-8B, and an argument that on the paper's own construction corpus the
direction dominantly tracks **proximity to episode closure** — how near the model
is to being finished — rather than how well it is doing.

The direction is real, linear, and causally potent. All three replicate. What is
at issue is the label.

---

## Summary

1. **The replication succeeds.** The axis rebuilds bit-identically from the
   released activation means (`max|diff| = 0.0`), and held-out token-level AUROC
   at layer 21 is 0.850 ± 0.017 against 0.516 ± 0.051 for random directions.

2. **There is a span-localization bug in the released construction code**, and
   fixing it makes the paper's own effect *stronger*: held-out AUROC 0.850 →
   0.880, per-function median 0.867 → 0.917, and the worst-behaved reward
   functions turn out to have been artifacts. This is a data-pipeline defect, not
   a problem with the thesis. It is reported separately below and everything
   downstream uses the corrected axis.

3. **The construction contrast is confounded with position in the response.**
   The before/after difference is flat in where you cut (+0.176 → +0.164 across
   cut fractions 0.15–0.85, Spearman −0.077), which is the algebraic signature of
   a monotone ramp rather than a criterion-locked update. The same contrast
   appears, 34% *larger*, in failed attempts split at a word that earned nothing.

4. **A matched-token test separates completion from value, and completion wins.**
   On byte-identical text, a "give up and call it done" prefill projects *above*
   a "found the criterion, ten paragraphs still to go" prefill. Value predicts the
   opposite ordering. This holds in 18 of 18 phrasing × tail cells, 100% of
   conversations.

5. **Causally, steering the axis mostly moves when the model stops.** Response
   length tracks steering strength at Spearman −0.96 (t = −107). In a length-free
   readout — one forward pass, no generation, with a random-direction control —
   the axis drives log P(end-of-turn) with a linear coefficient of **+12.9**
   against **−4.3** for a random direction of the same norm. It also moves a
   confidence probe by a real but smaller direction-specific amount (+4.0 vs
   +0.5), so the axis is **not purely** a closure signal and the paper's Fig 5a
   substantially survives. See §5, which is the section that most complicates the
   headline.

6. **This does not overturn the paper's behavioral results.** Backtracking,
   self-correction and the AIME correlations are real effects of this direction.
   Completion-proximity and value are strongly correlated in most task settings —
   which is why those results work. The ICRL corpus is unusual in letting the two
   come apart.

---

## Units, and how to read every number here

Every projection is a **cosine** between a token's residual-stream state and the
unit axis at a given layer — the paper's Eq. 2 metric. Layer 21 throughout, which
is the paper's layer. `value_axis.npy` has shape `(37, 4096)` =
`num_hidden_layers + 1`, so index 0 is the embedding output and index 21 is the
output of transformer block 20.

Anchors for judging any cosine below:

| Quantity | Value |
|---|---|
| Full before/after dynamic range at L21 | **0.165** |
| Random-direction null (cell differences) | **≤ 0.001** |
| Within-cell standard deviation at headers | **≈ 0.015** |
| The paper's own headline behavioral effects | **0.02 – 0.04** |

So an effect of 0.03 is not small — it is the size of the effects the paper
itself reports, and thirty times the random-direction floor.

"Held-out" always means **function-held-out**: the direction was built without
that criterion's data, using the paper's exact split seeding (`Random(si*42)`,
35/13, ten splits). It does not mean held out over conversations.

---

## Why I was looking

I came to this paper from a different direction. I work on a predictive-processing
model of valence, in which the felt signal is not a value level `V(s)` but an
expectation-relative rate of change in it. That model makes a sharp prediction
about any putative "value" or "welfare" direction in an LLM:

> A value function is stationary. `V(s)` does not habituate. Valence does.

Two 2026 papers — this one and Han, Chalmers & Izmailov's
[functional welfare axis](https://arxiv.org/pdf/2605.30232) — independently
found a linear direction encoding how well a model is doing, and both define it
as a level. If the direction is stationary, it is `V`, and "functional welfare"
overstates what was found. If it is expectation-relative, it is the architecture
my model predicts, and two labs published on a signal without noticing its
defining dynamical property.

Either answer is worth having, which is why I replicated first: a null result in
the follow-up experiment is only interpretable if the measurement stack is shown
to reproduce the original. The replication is what turned up everything below.

---

## 1. The replication

47,863 labeled tokens over 380 conversations and 48 reward functions, Qwen3-8B.

| Gate | Result |
|---|---|
| Axis rebuilds bit-identically from released means | ✔ `max\|diff\| = 0.0` |
| Held-out token AUROC @L21 exceeds early layers | ✔ 0.850 ± 0.017 vs 0.802 @L5 |
| Held-out peak in late-middle layers | ✔ peak layer 20 (0.852) |
| Random directions at chance | ✔ 0.516 ± 0.051 |
| Shuffled labels at chance | ✔ 0.500 ± 0.003 |
| Logit lens promotes completion-flavoured tokens | ✔ 想办法 / 进一步 / 加分 in top-30 |

Two honest gaps, both of which I think are reconcilable:

**We get 0.850, the paper reports 0.95+.** The aggregation unit explains most of
it: pooling tokens across criteria mixes per-criterion baseline offsets. Computed
per conversation, the same layer gives 0.928 ± 0.112. The paper does not state its
aggregation unit, so this is a plausible reconciliation rather than a confirmed
one.

**Layer choice matters less than Figure 2a implies.** Held-out AUROC is 0.802 at
layer 5 and 0.850 at layer 21 — a real but modest gap, and the curve is broad and
flat from roughly layer 4 to layer 30.

One methodological note, offered as a caution rather than a criticism.
`construction/compute_vector.py`'s `evaluate_heldout_auroc` projects *one*
before-mean and *one* after-mean per held-out reward function and calls
`roc_auc_score` on two points, which is 1.0 exactly when `after > before`. It
saturates at ≥ 0.98 on 34 of 37 layers and its argmax is layer 2. The paper's
stated task is to classify paragraph *tokens*, which needs forward passes — so
the token-level number is the one I treat as the replication target.

---

## 2. The bug

**`construction/extract_activations.py:47`**, upstream commit `44ad182`, inside
`find_modified_text_spans()`:

```python
idx = formatted_text.find(modified[:150], search_from)
```

Each attempt's rewritten paragraph is located by searching the whole rendered
conversation for its first 150 characters. But every paragraph appears **twice**:
the user posts the original, then the assistant emits an edited copy. Most
rewrites keep the opening sentences verbatim and edit later in the paragraph, so
`modified[:150] == original[:150]` and the search matches the **user's copy**,
which comes first.

Two failures follow:

- **Line 49**, `end = idx + len(modified)`, takes a length from one string and
  adds it to an index found in another. The rewrite is longer than the original,
  so the span overruns the user turn and swallows `<|im_end|>`,
  `<|im_start|>assistant` and the opening of the model's `<thinking>` block.
- **Lines 64–66** compute reward-word offsets with `re.finditer` inside
  `attempt["modified_text"]`, then add them to a span start that points at the
  *original* text. The split point lands wherever the two strings diverge. This
  affects all 35 semantic criteria.

### Measured consequence

| Quantity | Value |
|---|---|
| Conversations whose target span starts in a **user** turn | **234 / 380 (62%)** |
| Training tokens in user turns | **29,460 / 47,863 (61.6%)** |
| Training tokens in chat markers | 936 (2.0%) |
| Training tokens in assistant turns (as intended) | 17,467 (36.5%) |
| Conversations where the labeled reward word is inside the span | 163 / 380 (43%) |

A worked example: in `gemstone__conv03` the reward word `sapphire` sits at
character 4777 of the rendered conversation. The computed reward span is
4121–4129, which decodes to `"ve figur"` — inside *"I've figured out the
criterion"*. The `before` side is the user's copy of the paragraph; the `after`
side is the first 20 tokens of the model's reasoning. The word `sapphire` is not
in the span at all.

The defect is in released code and is inherited by the released artifacts:
re-running upstream's `classify_tokens` reproduces the shipped
`activation_means.pt` per-function counts 48/48, and `compute_vector.py` on those
means reproduces `value_axis.npy` bit-identically. Figure 2a's evaluation uses
the same labels on both sides, so it does not catch this.

### The correction, and why it is good news

`corrected_spans.py` keeps the paper's design exactly — successful attempt of
paragraph `discovery_paragraph + 1`, split on criterion-satisfying tokens,
≥ 3 tokens per side — and changes only the localization: the span is found
*structurally*, as the assistant turn's body after its `</thinking>` block, never
by text search. Reward words are then matched inside that same span, so offsets
index the text they were computed from.

| | Shipped | Corrected |
|---|---|---|
| Held-out token AUROC @L21 | 0.850 ± 0.017 | **0.880 ± 0.013** |
| Peak layer / value | 20 / 0.852 | 24 / **0.900** |
| Per-function median | 0.867 | **0.917** |
| Functions above chance | 48/48 | 48/48 |
| `contains_colon` (was the worst) | 0.622 | **0.811** |
| `contains_dash` | — | 0.989 |
| Worst function after correction | — | `chemical_element` 0.746 |

`cos(corrected, shipped)` at L21 is **0.707** — about 45°. Across layers it runs
from 0.574 (L14) to 0.902 (L36). These are clearly not the same direction, and
they are clearly not unrelated.

**The bug was attenuating a real effect, not manufacturing one.** Roughly 62% of
the training signal was a user-text-vs-assistant-text contrast riding on top of
the intended one. Remove it and the intended effect gets cleaner. Splitting the
original projections by alignment shows the same thing from the other side: on
the 146 conversations where the localizer happened to land correctly, the effect
is present and slightly stronger.

Everything in the rest of this document uses the corrected axis.

---

## 3. The construction contrast is a ramp, not a jump

This is the methodological heart of the argument.

The axis is built from a within-paragraph contrast: mean projection of tokens
*after* the criterion-satisfying token, minus mean projection of tokens *before*
it, inside one rewarded rewrite. That is read as a value update at the moment the
criterion is met.

But the same contrast is produced by *any* quantity that rises monotonically
through the response, with no reference to the criterion at all. The two are
distinguishable by where you cut:

- A **criterion-locked jump** should be large only when the cut is at the
  criterion token.
- A **linear ramp** gives `mean(after) − mean(before) = slope · T / 2` for a cut
  at *any* fraction `p` — independent of `p`. Flat in cut position, and present
  wherever you cut.

Binning rewarded attempts by the fraction of the rewrite preceding the split
(`ramp_cut_invariance.py`, held-out corrected axis, L21):

| Cut fraction | n | after − before |
|---|---|---|
| 0.15–0.25 | 118 | +0.1758 |
| 0.25–0.35 | 205 | +0.1669 |
| 0.35–0.45 | 197 | +0.1600 |
| 0.45–0.55 | 241 | +0.1598 |
| 0.55–0.65 | 178 | +0.1578 |
| 0.65–0.75 | 203 | +0.1643 |
| 0.75–0.85 | 216 | +0.1638 |
| **pooled** | **1358** | **+0.1633** |

Spearman(diff, cut fraction) = **−0.077** (p = 4.5e−3). Across a range where the
cut moves through 70% of the paragraph, the contrast varies by about 7% of itself.
That is the ramp signature.

### The placebo

Now split **failing** attempts at the word the model itself said it was
targeting. These attempts earn −1. No criterion is met anywhere in them, and the
cut word earned nothing:

| Channel | n | after − before |
|---|---|---|
| Criterion split, rewarded | 1358 | +0.1633 |
| **Believed-word split, failed** | **1140** | **+0.2190** |

The placebo is not smaller. It is **34% larger**, and equally flat in cut position
(Spearman −0.112). A contrast that survives at full size in attempts where no
reward was delivered is not a measurement of the reward event.

Two further observations in the same table:

- The jump **does not habituate**. Across the first, second, third and fourth
  post-discovery paragraphs — where the reward is fully predicted — it is
  +0.168 / +0.166 / +0.167 / +0.160, and the paired later-minus-first difference
  is −0.0005 (t = −0.2, n = 260). The *n*th entirely expected reward produces the
  same "update" as the first. (This is the direct test of my own habituation
  hypothesis for the within-response signal, and it comes out negative. It is
  exactly what a ramp predicts.)
- The **level** at the start of each attempt is flat across phases, where a
  stationary `V(s)` predicts it should climb: pre-discovery failures −0.2251,
  post-discovery earned successes −0.1877, with the discovery attempt in between.
  Most of the movement is within-response, not across-state.

---

## 4. The matched-token test: completion beats value

The ramp result says the construction contrast doesn't measure a value update.
It does not by itself say what the axis *does* measure. For that we need a
contrast where value and completion make opposite predictions on identical text.

Take a real conversation, truncated after three failed attempts, and prefill the
assistant turn one of two ways:

| Arm | Content | Value | Completion |
|---|---|---|---|
| `correct_nocomplete` | "I've worked out the criterion — but there are still ten paragraphs to go" | **HIGH** | LOW |
| `complete_nosucc` | "I'm going to stop here and call it done" (without ever succeeding) | LOW | **HIGH** |

Then append a **byte-identical tail** to both. Token identity, token count and
absolute position are held fixed by construction; only what came before differs.

- A value/welfare direction predicts `correct_nocomplete` > `complete_nosucc`.
- A completion-proximity direction predicts the reverse.

Measured on the tail tokens, corrected axis, L21, paired by conversation
(random-direction control on the same tokens: +0.0004):

| Segment | `correct_nocomplete` | `complete_nosucc` | difference | t | n |
|---|---|---|---|---|---|
| thinking | −0.0947 | −0.0690 | **−0.0257** | −79.7 | 65 |
| body | −0.1206 | −0.0877 | **−0.0329** | −98.4 | 65 |
| **tail (byte-identical)** | −0.0404 | −0.0316 | **−0.0087** | −32.4 | 65 |

**Giving up projects higher than succeeding**, on identical text. The sign is the
completion sign.

### Robustness

Because a single phrasing pair could be carrying something idiosyncratic, the
whole thing was rerun with 3 phrasings per arm × 2 different tails — 18 cells:

- **18 of 18 cells are completion-signed.**
- **100% of conversations** in every cell.
- Effects run −0.0087 to −0.0566, i.e. 5% to 34% of the full 0.165 dynamic range.
  The original pair was the *weakest* of the set.
- Pooled per-phrasing diagonals: −0.0286 (t = −16.0), −0.0403 (t = −26.3),
  −0.0381 (t = −40.8), n = 130 each.

### The logit lens says the same thing

Unembedding the **corrected** axis at L21 and reading the top-30 promoted tokens
gives, in order:

> ` afterwards`, `后续` (follow-up), `这才是` (*this* is what really…),
> ` thereafter`, `IfNeeded`, `其它问题` (other problems), `等等` (etc.),
> `另行` (separately, later), `不会再` (won't again), `后再` (after, then), …
> ` afterward`, `进一步` (go further), `结尾` (ending), `下次` (next time),
> `最后` (finally), `结局` (outcome)

This is aftermath-and-sequencing vocabulary: *afterwards, thereafter, follow-up,
next time, ending, finally, won't happen again*. The paper's cited encouragement
tokens survive the correction — `想办法` (figure out a way) is at rank 14 and
`进一步` is in the top 30 — but they are not what the cleaned direction is mostly
made of. Only 10 of 30 tokens overlap with the shipped axis's top-30, so this is
largely a view of the direction that the bug was obscuring.

I would not rest anything on a logit lens alone, and the magnitudes here are raw
unembedding logits of a unit vector, so only the ranking is meaningful. But it is
the most directly legible piece of evidence in the whole stack, and it points the
same way as everything else.

### One negative result, stated plainly

The same experiment included direct Yes/No probes ("are you done?", "is it
correct?"). I do **not** treat these as evidence. At L21 the token "No" projects
above "Yes" on both questions, the same way "−1" projects above "+1" — the
readout is dominated by token identity rather than by state. That is precisely
why the matched-tail design is the only trustworthy version of this test, and it
is worth flagging that the same lexical dominance would affect any probe-based
readout in this setting.

---

## 5. Causally, the axis controls stopping — and, it turns out, something else too

The correlational case above needs a causal counterpart. Using the paper's own
steering paradigm — unit direction at L21, added as `α·d` by a forward hook on
decoder block 20 at every position throughout generation, α ∈ [−75, +75],
temperature 0.7, top_p 0.9 — on 15 conversations × 2 truncation states, with a
probe asking the model to rate its confidence 0–10 and then explain:

| Readout | Shipped axis | Corrected axis |
|---|---|---|
| **Response length** vs α (per-prefix Spearman) | −0.783 (t = −19.8) | **−0.959 (t = −107.3)** |
| **Verbalized confidence rating** vs α | +0.003 (t = 0.0) | −0.160 (t = −1.7, n.s.) |

At α = −75 every generation runs into the 300-token cap. At α = +75 on the
corrected axis the model produces well-formed four-token answers — `"8"` and
stop — with zero parse failures. That is not degeneration; it is wrapping up.

Meanwhile the confidence rating is flat under steering, and this is not because
the probe is insensitive: unsteered, the same probe reads 5.7 in the early state
(rule unknown) and 8.7 in the post-discovery state (rule known and confirmed).

### The obvious objection, and the experiment that answers it

The rating above was read out of *generated text*, so a wrap-up push could
truncate the response before the rating settles. That is essentially the
objection the paper's own Figure 5a invites, and it deserved a real test rather
than an argument.

So: **one forward pass, no generation at all.** Same steering hook, same α grid,
same prefixes. Read the logits at the single position where the answer token
would go. Three channels — a forced Yes/No confidence probe, the expected value
of a 0–9 digit distribution, and log P(`<|im_end|>`) as the closure channel.

Crucially, this run includes a **random unit direction** steered at the same α.
Pushing the residual stream by ‖α‖ = 75 in *any* direction takes it
off-distribution, and an off-distribution state degrades every readout. Without
that control the experiment cannot distinguish content from damage.

It matters, because the damage is real and large. Fitting each prefix's profile
as `a + bα + cα²`: the **quadratic** term is big and positive on every channel
under every direction, random included (log P(end): c = +32.7 shipped, +28.5
corrected, +26.1 random). Any hard push in any direction raises the probability
of just stopping. A Spearman correlation on that U-shaped profile reports
whichever arm rises further, which is why the linear term is the statistic that
matters:

| Channel | Shipped | Corrected | **Random control** |
|---|---|---|---|
| log P(`<|im_end|>`) — closure | +2.81 (t = +4.6) | **+12.86 (t = +39.3)** | **−4.25 (t = −21.6)** |
| logit(Yes) − logit(No) — confidence | +1.62 (t = +3.3) | **+4.03 (t = +24.9)** | +0.48 (t = +2.8) |
| P(Yes \| Yes or No) — bounded | +0.140 | +0.171 | −0.019 |
| E[rating] over 0–9 | +0.788 | +0.970 | **+0.981 (t = +11.6)** |

Reading this honestly, three things:

1. **The closure channel is enormous and specific.** The corrected axis drives
   log P(end-of-turn) with a linear coefficient of +12.9, four and a half times
   the shipped axis, while the random control of identical norm moves it in the
   *opposite* direction. With no generation involved, length cannot explain any
   of it. This is the strongest single piece of evidence in the whole document.
2. **But confidence is not flat.** The Yes/No margin has a real, direction-specific
   linear component: +4.03 on the corrected axis against +0.48 for the random
   control. In the post-discovery state, P(Yes) goes from ≈0 at α = 0 to 0.51 at
   α = +50 and 0.72 at α = +75. **The paper's Figure 5a substantially survives
   this test**, and the flat rating in the generation experiment appears to have
   been an artifact of reading confidence out of text the model had already
   committed to.
3. **The graded rating shows nothing.** E[rating] moves *identically* under the
   random direction (+0.981) as under the corrected axis (+0.970). Whatever that
   channel is picking up, it is not the axis.

One more thing worth noting, because it speaks to whether this is a targeted
intervention at all: under the random direction the model's first-token
probability mass on the digit set **collapses** — 1.000 → 0.51 → 0.015 → 0.001 as
α goes +25 → +50 → +75. It stops answering the question. Under the corrected axis
that mass stays at 1.000 until α = ±75. The axis makes the model wrap up; a
random push of the same size just breaks it.

### What I now think this means

The axis is **not purely a closure controller**. It carries a genuine confidence
component, and I would have been wrong to claim otherwise — this experiment was
run to try to falsify the closure reading and it partly did.

What survives is the weaker but still substantial claim: the closure component is
much the larger of the two, it is what dominates behavior (generation length at
Spearman −0.96), and it is what the construction procedure preferentially samples,
because contrasting late-in-response against early-in-response text is a
completion contrast. The label "how well the model is doing" attributes to the
whole direction what appears to belong mostly to one component of it.


---

## 6. What else the retry structure rules out

Within a paragraph, the projection at the assistant header climbs with retry
depth. The header is a 7-token span that is byte-identical everywhere, so token
identity is controlled by construction. In the released corpus depth only reaches
about 5, which is too short to discriminate. So I spliced scripted 20-failure
sequences out of the shared paragraph pool, in two arms (35 paragraphs each):
`diverse` (attempts 6–20 are distinct rewrites) and `duplicate` (attempts 6–20
are verbatim repeats of the first five).

**The climb never reverses.** Bands of retry depth, paired by paragraph:

| Depth band | Mean | Step |
|---|---|---|
| 2–5 | −0.0546 | — |
| 6–10 | −0.0388 | +0.0158 (t = +32.0) |
| 11–15 | −0.0338 | +0.0050 (t = +14.6) |
| 16–20 | −0.0303 | +0.0035 (t = +11.2) |

It decelerates sharply — the step shrinks by a factor of about 4.5 — and stays
positive out to twenty consecutive failures. A value or solvability-inference
account needs it to turn *negative*: twenty failures is strong evidence the
paragraph is not going to be solved, and expected value should fall. It doesn't.

**Zero new information does not flatten the climb — it raises it.** After attempt
5 the duplicate arm delivers no new information about the criterion whatsoever.
Diverse minus duplicate:

| Depth band | diverse − duplicate | t |
|---|---|---|
| 6–10 | −0.0036 | −8.3 |
| 11–15 | −0.0070 | −10.1 |
| 16–20 | −0.0056 | −7.3 |

The sign is negative: the zero-information arm sits **higher**. No
information-accumulation account produces that.

I flag this as the one result that the pure closure reading does not
straightforwardly explain either. Repetition makes the context more predictable,
and a "settledness" variant of the hypothesis handles it more naturally than
"proximity to done" does. Which of those two it is remains open, and there is a
designed experiment for it in §8.

---

## 7. What this does not claim

I want to be precise about the scope, because it is narrower than the headline
might suggest.

- **The direction is real.** It is linear, it is decodable at 0.880 held-out
  AUROC, it survives random-direction and shuffled-label controls, and it
  generalizes to reward functions it was not built from. None of that is in
  question.
- **The behavioral results are real.** Steering this direction changes
  backtracking, self-correction and task behavior. I reproduce a large causal
  effect of the same direction. I am reinterpreting *what the direction is*, not
  whether steering it does something.
- **Completion and value are usually correlated**, which is why the paper's
  correlational results hold. On a math problem, being near the end and being on
  track are nearly the same thing. The ICRL corpus is unusual in letting them
  come apart — and the construction procedure, by contrasting late-in-response
  against early-in-response text, samples the completion component preferentially.
- **This is all on the construction corpus**, not on AIME or Arena. The most
  reasonable objection is that closure and value are simply confounded
  in-distribution and only dissociate here. Testing that requires running the
  matched-tail design in the paper's own settings, which I have not done.
- **The paper's Figure 5a substantially survives.** I ran the length-free version
  of that test specifically because it could falsify the closure reading, and it
  partly did: steering does move a confidence probe, by a real and
  direction-specific amount (§5). The claim that survives is about which
  component dominates, not that the confidence component is absent.

Other caveats worth having on the table: the steering experiment is 15
conversations, one seed per condition, one probe phrasing; the layer picture is
not uniform, with a small counter-signed band around L25–27 in the prefill
probes; the spliced retry sequences carry mild history-incoherence; and the
LLM-judged confidence labels are heavily skewed, so the belief-correctness cells
are hints rather than findings.

---

## 8. What I would do next

Ordered by discriminating power per unit of cost. The first two bear directly on
the paper's own figures.

1. **EOS-masked and prompt-only steering.** Separates "the axis encodes a closure
   state that then promotes end-of-turn" from "the axis promotes the EOS logit
   directly." Cheap, and it is the main unresolved mechanistic question.
2. ~~A length-free confidence readout under steering.~~ **Done — see §5.** It
   partly falsified the strong version of my own claim, which is the main reason
   §5 now reads the way it does. The natural follow-up is to repeat it on the
   paper's own Fig 3a phrasings, so the sign-flip control carries over directly.
3. **The matched-tail prefill design on AIME/Arena.** Takes the experiment that
   actually discriminates into the paper's own distribution. This is the single
   most decisive test of whether the reinterpretation generalizes.
4. **Build a closure axis explicitly** — end-of-response vs mid-response means on
   neutral text — measure its cosine against the value axis per layer, then
   project the value axis into closure + residual and re-run the paper's
   correlational results on the residual. If the Figure 3 effects live in the
   closure component, that settles it. Notably, the Figure 3b bands rise
   monotonically through rollouts, which is what a position/length component
   would do.
5. **A designed predictability-vs-closure manipulation** — matched depth,
   entropy-varied continuations — to resolve the duplicate-beats-diverse residue
   in §6.
6. **Scripted post-discovery failures.** The retry climb cannot be tested after
   discovery observationally: post-discovery paragraphs are 100% single-attempt
   (1042/1042). Rule known, execution scripted to fail, retry structure matched.
   Information-gain predicts no climb; hazard/value predicts a high flat start;
   counter/closure predicts the same climb as before.
7. **The matched-suffix history experiment** that sent me here in the first place.
   Identical final rounds, differing prefix valence:

   ```
   LATE    -1 -1 -1 -1 -1 | +1 +1 +1     <- measure only here
   EARLY   +1 +1 +1 +1 +1 | +1 +1 +1     <- byte-identical tokens
   ```

   A stationary `V` predicts identical projections. An expectation-relative
   signal predicts `LATE > EARLY`. The ramp finding in §3 upgrades this design
   from nice-to-have to necessary: byte-identical suffixes are the only control
   that removes the position confound.

---

## Reproducing this

See [README.md](README.md) for setup. Every number above is produced by a script
in this repository; [results/MANIFEST.md](results/MANIFEST.md) says which command
regenerates each artifact.

| Claim | Script |
|---|---|
| Replication gates, corrected-vs-shipped axis, AUROC tables | `corrected_axis_report.py` |
| The bug fix itself | `corrected_spans.py` |
| Paper's mean-level metric on corrected means | `corrected_mean_validation.py` |
| Ramp / cut-invariance / placebo (§3) | `ramp_cut_invariance.py` |
| Within-attempt cells, non-habituation, level-by-phase (§3) | `attempt_split_report.py` |
| Logit lens, corrected axis (§4) | `results/logit_lens_corrected.json` |
| Matched-token prefills (§4) | `prefill_probes_report.py` |
| Phrasing robustness (§4) | `prefill_rephrase_report.py` |
| Steering dissociation (§5) | `steering_probe_report.py` |
| Length-free readout (§5) | `steering_logits_report.py` |
| Retry depth, extended retries (§6) | `extend_retries_report.py`, `assistant_headers_report.py` |
| Position confounds that killed earlier findings | `confidence_correlation.py` |

Upstream code is pinned to commit `44ad182` of
[nickjiang2378/value-axis](https://github.com/nickjiang2378/value-axis).
