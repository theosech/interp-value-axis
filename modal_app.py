"""GPU step for the value-axis replication: token-level projections on Modal.

Why this exists
---------------
The repo's CPU-only `compute_vector.py` computes a *mean-level* AUROC: for each
held-out reward function it projects one before-mean and one after-mean and asks
which is higher. With two points that metric is 1.0 iff after > before, so it
saturates near 1.0 at almost every layer and does NOT reproduce the paper's
Figure 2a. The paper's stated task is to "classify paragraph tokens before and
after the criterion-satisfying token" -- i.e. TOKEN-level. That needs forward
passes, which is what this file does.

Outputs per labeled token: cosine similarity against the value axis at every
layer (paper Eq. 2 uses mean cosine over a span; we return per-token cosines so
the notebook can aggregate however it likes).

Usage
-----
    modal run modal_app.py                  # writes results/projections.npz
    modal run modal_app.py --limit 40       # quick smoke test
    modal run modal_app.py::logit_lens_main # writes results/logit_lens.json
"""

import modal

MODEL = "Qwen/Qwen3-8B"
MAX_LENGTH = 16384
GPU = "A10G"   # 24 GB; A100 requires a payment method on this Modal account

app = modal.App("value-axis-adapt")

# Persist the HF cache so Qwen3-8B (~16 GB) is downloaded once, not per run.
hf_cache = modal.Volume.from_name("value-axis-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.1",
        "transformers==4.56.2",
        "huggingface_hub==0.36.2",
        "numpy==2.2.6",
        "scikit-learn==1.8.0",
        "accelerate==1.12.0",
    )
    .env({"HF_HOME": "/cache/huggingface",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    # Lazy local mounts MUST be last in the chain, and nothing local may be
    # imported at module scope -- see the modal-local-source-build-import skill.
    .add_local_dir("value-axis", remote_path="/root/value-axis")
    .add_local_python_source("turns", "corrected_spans")
    .add_local_file("results/reward_words.jsonl", "/root/reward_words.jsonl")
    .add_local_file("results/believed_words.jsonl", "/root/believed_words.jsonl")
    .add_local_file("results/activation_means_corrected.pt",
                    "/root/activation_means_corrected.pt")
    .add_local_file("results/extended_retries.json", "/root/extended_retries.json")
    .add_local_file("results/prefill_probes.json", "/root/prefill_probes.json")
    .add_local_file("results/prefill_rephrase.json", "/root/prefill_rephrase.json")
)


def _setup_paths():
    """Put the cloned repo on sys.path. Call inside function bodies only."""
    import sys
    for p in ("/root/value-axis", "/root/value-axis/construction"):
        if p not in sys.path:
            sys.path.insert(0, p)


@app.function(image=image, gpu=GPU, volumes={"/cache": hf_cache}, timeout=60 * 60)
def compute_projections(limit=None, trace_ids=None, n_random=20, seed=0):
    """Forward every ICRL conversation; return per-layer cosines for labeled tokens.

    Returns a dict with:
      rows:   list of {conv_id, reward_fn, reward_fn_type, label, token,
                       token_pos, n_tokens} for each before/after token
      cos:    (n_rows, n_layers) float32 cosine of that token against the axis
      cos_random: (n_rows, n_random) cosine against random unit directions at
              layer 21 -- the specificity control. A real effect must not
              appear here.
      traces: {conv_id: {"cos21": [...], "tokens": [...], "rounds": [...]}}
              full per-token layer-21 traces for the conversations in trace_ids
    """
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _setup_paths()
    from common.paths import value_axis
    from shared import load_conversations, load_reward_functions, load_reward_labels
    from extract_activations import classify_tokens

    axis = np.load(value_axis())                      # (37, 4096)
    axis_t = torch.tensor(axis, dtype=torch.float32).cuda()
    axis_t = axis_t / axis_t.norm(dim=1, keepdim=True)
    n_layers_axis = axis_t.shape[0]

    # Random unit directions in the same space, evaluated at layer 21 only.
    g = torch.Generator(device="cpu").manual_seed(seed)
    rand_t = torch.randn(n_random, axis_t.shape[1], generator=g).cuda()
    rand_t = rand_t / rand_t.norm(dim=1, keepdim=True)

    # Held-out split directions, so the token-level AUROC can be computed the way
    # the paper reports it: build the axis from N_TRAIN reward functions and score
    # only tokens whose reward function was NOT used to build it. Same split
    # seeding as compute_vector.evaluate_heldout_auroc.
    import random as _random

    from compute_vector import (N_SPLITS, N_TRAIN, load_activation_means,
                                mean_contrastive_direction, valid_functions)
    act = load_activation_means()
    valid = valid_functions(act)
    split_dirs, split_test_fns = [], []
    for si in range(N_SPLITS):
        rng = _random.Random(si * 42)
        shuffled = valid.copy()
        rng.shuffle(shuffled)
        d = mean_contrastive_direction(act, set(shuffled[:N_TRAIN]))
        d = d / np.linalg.norm(d, axis=1, keepdims=True).clip(1e-10)
        split_dirs.append(torch.tensor(d, dtype=torch.float32).cuda())
        split_test_fns.append(shuffled[N_TRAIN:])
    split_t = torch.stack(split_dirs)          # (n_splits, n_layers, hidden)
    del act
    print(f"{N_SPLITS} held-out split directions "
          f"({N_TRAIN} train / {len(valid)-N_TRAIN} test functions)", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda", output_hidden_states=True)
    model.eval()
    n_layers = model.config.num_hidden_layers + 1
    assert n_layers == n_layers_axis, f"axis has {n_layers_axis} layers, model {n_layers}"

    rf_by_name = {rf["name"]: rf for rf in load_reward_functions()}
    reward_labels = load_reward_labels()
    conversations = load_conversations()
    if limit:
        conversations = conversations[:limit]
    trace_ids = set(trace_ids or [])
    print(f"{len(conversations)} conversations, {n_layers} layers", flush=True)

    rows, cos_all, cos_rand_all, cos_split_all, traces = [], [], [], [], {}

    for ci, conv in enumerate(conversations):
        reward_fn = conv["reward_fn"]
        conv_idx = int(conv["conversation_id"].split("__conv")[1])

        try:
            formatted = tokenizer.apply_chat_template(
                conv["full_messages"], tokenize=False,
                add_generation_prompt=False, enable_thinking=False)
        except TypeError:
            formatted = tokenizer.apply_chat_template(
                conv["full_messages"], tokenize=False, add_generation_prompt=False)

        enc = tokenizer(formatted, return_tensors="pt", return_offsets_mapping=True,
                        add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
        input_ids = enc["input_ids"].cuda()
        offsets = enc["offset_mapping"][0].tolist()

        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            # Loop over layers rather than stacking all 37 in float32: stacking
            # peaks at ~6 GB for a 10k-token conversation and was what forced an
            # A100. One layer at a time is ~160 MB, so this fits a 24 GB card.
            n_tok_i = out.hidden_states[0].shape[1]
            cos = np.empty((n_layers, n_tok_i), dtype=np.float32)
            cos_rand = np.empty((rand_t.shape[0], n_tok_i), dtype=np.float32)
            cos_split = np.empty((split_t.shape[0], n_layers, n_tok_i), dtype=np.float32)
            for l in range(n_layers):
                h = out.hidden_states[l][0].float()
                h = h / h.norm(dim=1, keepdim=True).clamp_min(1e-6)
                cos[l] = (h @ axis_t[l]).cpu().numpy()
                cos_split[:, l, :] = (split_t[:, l, :] @ h.T).cpu().numpy()
                if l == 21:
                    cos_rand[:] = (rand_t @ h.T).cpu().numpy()
                del h
            del out
            torch.cuda.empty_cache()

        token_strings = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
        classifications = classify_tokens(
            token_strings, offsets, formatted, conv, reward_fn,
            rf_by_name[reward_fn]["type"], reward_labels, conv_idx)

        n_tok = len(token_strings)
        for pos, label in classifications:
            if label in ("before", "after"):
                rows.append({
                    "conv_id": conv["conversation_id"], "reward_fn": reward_fn,
                    "reward_fn_type": rf_by_name[reward_fn]["type"], "label": label,
                    "token": token_strings[pos], "token_pos": pos, "n_tokens": n_tok,
                })
                cos_all.append(cos[:, pos])
                cos_rand_all.append(cos_rand[:, pos])
                cos_split_all.append(cos_split[:, :, pos])

        if conv["conversation_id"] in trace_ids:
            traces[conv["conversation_id"]] = {
                "cos21": cos[21].tolist(), "tokens": token_strings,
                "reward_fn": reward_fn, "discovery_paragraph": conv["discovery_paragraph"],
            }

        if (ci + 1) % 25 == 0:
            print(f"  [{ci+1}/{len(conversations)}] {len(rows)} labeled tokens", flush=True)

    return {"rows": rows,
            "cos": np.asarray(cos_all, dtype=np.float32),
            "cos_random": np.asarray(cos_rand_all, dtype=np.float32),
            "cos_split": np.asarray(cos_split_all, dtype=np.float32),
            "split_test_fns": split_test_fns,
            "traces": traces}


@app.function(image=image, gpu=GPU, volumes={"/cache": hf_cache}, timeout=60 * 60)
def attempt_projections(limit=None, n_random=20, seed=0, corrected=False):
    """Projections at every ATTEMPT in every conversation -- the lucky-vs-earned test.

    Two measurement points per attempt:
      fb_*   : the feedback turn body, whose text is byte-identical ("+1" / "-1")
               across all instances. Content-matched by construction, so a
               difference there is attributable to history alone.
      asst_* : mean over the attempt's own assistant tokens (not content-matched).

    Cells (see turns.cell):
      pre_fail    2446   failing guess, rule unknown
      pre_lucky    842   +1 while still guessing  -> UNPREDICTED reward
      post_earned 1026   +1 after announcing the rule -> PREDICTED reward

    A value-function reading of the axis predicts post_earned > pre_lucky.
    A prediction-error reading predicts pre_lucky > post_earned.

    corrected=True uses the corrected-span axis and split directions (built from
    /root/activation_means_corrected.pt) instead of the shipped ones.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _setup_paths()
    from common.paths import value_axis
    from shared import load_conversations
    from turns import attempt_records, cell

    import random as _random

    from compute_vector import (N_SPLITS, N_TRAIN, compute_value_axis,
                                load_activation_means,
                                mean_contrastive_direction, valid_functions)

    if corrected:
        act = torch.load("/root/activation_means_corrected.pt",
                         map_location="cpu", weights_only=False)
        axis = compute_value_axis(act)
    else:
        act = load_activation_means()
        axis = np.load(value_axis())
    axis_t = torch.tensor(axis, dtype=torch.float32).cuda()
    axis_t = axis_t / axis_t.norm(dim=1, keepdim=True)

    g = torch.Generator(device="cpu").manual_seed(seed)
    rand_t = torch.randn(n_random, axis_t.shape[1], generator=g).cuda()
    rand_t = rand_t / rand_t.norm(dim=1, keepdim=True)
    valid = valid_functions(act)
    split_dirs, split_test_fns = [], []
    for si in range(N_SPLITS):
        rng = _random.Random(si * 42)
        shuffled = valid.copy()
        rng.shuffle(shuffled)
        dvec = mean_contrastive_direction(act, set(shuffled[:N_TRAIN]))
        dvec = dvec / np.linalg.norm(dvec, axis=1, keepdims=True).clip(1e-10)
        split_dirs.append(torch.tensor(dvec, dtype=torch.float32).cuda())
        split_test_fns.append(shuffled[N_TRAIN:])
    split_t = torch.stack(split_dirs)
    del act

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda", output_hidden_states=True)
    model.eval()
    n_layers = model.config.num_hidden_layers + 1

    conversations = load_conversations()
    if limit:
        conversations = conversations[:limit]
    print(f"{len(conversations)} conversations", flush=True)

    rows, fb_cos, fb_split, fb_rand, asst_cos = [], [], [], [], []

    for ci, conv in enumerate(conversations):
        try:
            formatted = tokenizer.apply_chat_template(
                conv["full_messages"], tokenize=False,
                add_generation_prompt=False, enable_thinking=False)
        except TypeError:
            formatted = tokenizer.apply_chat_template(
                conv["full_messages"], tokenize=False, add_generation_prompt=False)

        enc = tokenizer(formatted, return_tensors="pt", return_offsets_mapping=True,
                        add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
        input_ids = enc["input_ids"].cuda()
        offsets = enc["offset_mapping"][0].tolist()
        n_tok = len(offsets)

        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            cos = np.empty((n_layers, n_tok), dtype=np.float32)
            cos_sp = np.empty((split_t.shape[0], n_layers, n_tok), dtype=np.float32)
            cos_rd = np.empty((rand_t.shape[0], n_tok), dtype=np.float32)
            for l in range(n_layers):
                h = out.hidden_states[l][0].float()
                h = h / h.norm(dim=1, keepdim=True).clamp_min(1e-6)
                cos[l] = (h @ axis_t[l]).cpu().numpy()
                cos_sp[:, l, :] = (split_t[:, l, :] @ h.T).cpu().numpy()
                if l == 21:
                    cos_rd[:] = (rand_t @ h.T).cpu().numpy()
                del h
            del out
            torch.cuda.empty_cache()

        def span_positions(a, b):
            return [i for i in range(n_tok)
                    if offsets[i][0] >= a and offsets[i][1] <= b and offsets[i][1] > offsets[i][0]]

        for rec in attempt_records(formatted, conv["discovery_paragraph"]):
            if rec["reward"] is None:
                continue
            fb_pos = span_positions(rec["fb_start"], rec["fb_end"])
            as_pos = span_positions(rec["assistant_start"], rec["assistant_end"])
            if not fb_pos or not as_pos:
                continue
            rows.append({
                "conv_id": conv["conversation_id"], "reward_fn": conv["reward_fn"],
                "paragraph": rec["paragraph"],
                "attempt_in_paragraph": rec["attempt_in_paragraph"],
                "phase": rec["phase"], "reward": bool(rec["reward"]), "cell": cell(rec),
                "discovery_paragraph": conv["discovery_paragraph"],
                "fb_token_pos": int(fb_pos[0]), "n_fb_tokens": len(fb_pos),
                "n_asst_tokens": len(as_pos), "n_tokens": n_tok,
            })
            fb_cos.append(cos[:, fb_pos].mean(axis=1))
            fb_split.append(cos_sp[:, :, fb_pos].mean(axis=2))
            fb_rand.append(cos_rd[:, fb_pos].mean(axis=1))
            asst_cos.append(cos[:, as_pos].mean(axis=1))

        if (ci + 1) % 25 == 0:
            print(f"  [{ci+1}/{len(conversations)}] {len(rows)} attempts", flush=True)

    return {"rows": rows,
            "fb_cos": np.asarray(fb_cos, dtype=np.float32),
            "fb_split": np.asarray(fb_split, dtype=np.float32),
            "fb_random": np.asarray(fb_rand, dtype=np.float32),
            "asst_cos": np.asarray(asst_cos, dtype=np.float32),
            "split_test_fns": split_test_fns}


@app.function(image=image, gpu=GPU, volumes={"/cache": hf_cache}, timeout=60 * 60)
def divider_projections(believed=False):
    """Within-attempt (mean AFTER - mean BEFORE) around the criterion-satisfying word.

    This is the quantity the axis was literally built from -- construction takes exactly
    this contrast, but only on ONE paragraph per conversation. Computing it per attempt
    gives a self-normalising readout: as a within-attempt difference it cancels the
    attempt's content, its position in the conversation, context length, and any
    token-identity baseline, which is what has confounded every span-mean comparison.

    Only defined for SUCCESSFUL attempts -- a failing attempt has no reward word to
    divide on. So this covers RC vs RW and cannot speak to FC vs FW.

    Reward positions: syntactic criteria via SYNTACTIC_TOKEN_CHECKERS, semantic via the
    codex-labelled spans in reward_words.jsonl.
    """
    import json as _json

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _setup_paths()
    from common.paths import value_axis
    from shared import (SYNTACTIC_TOKEN_CHECKERS, load_conversations,
                        load_reward_functions)
    from turns import attempt_records

    MIN_PER_SIDE = 3            # same threshold the construction pipeline uses
    THINK_END = ("</think>", "</thinking>")

    axis = np.load(value_axis())
    axis_t = torch.tensor(axis, dtype=torch.float32).cuda()
    axis_t = axis_t / axis_t.norm(dim=1, keepdim=True)

    import random as _random

    from compute_vector import (N_SPLITS, N_TRAIN, load_activation_means,
                                mean_contrastive_direction, valid_functions)
    act = load_activation_means()
    valid = valid_functions(act)
    split_dirs, split_test_fns = [], []
    for si in range(N_SPLITS):
        rng = _random.Random(si * 42)
        sh = valid.copy()
        rng.shuffle(sh)
        dv = mean_contrastive_direction(act, set(sh[:N_TRAIN]))
        dv = dv / np.linalg.norm(dv, axis=1, keepdims=True).clip(1e-10)
        split_dirs.append(torch.tensor(dv, dtype=torch.float32).cuda())
        split_test_fns.append(sh[N_TRAIN:])
    split_t = torch.stack(split_dirs)
    del act

    words_by_key = {}
    src = "/root/believed_words.jsonl" if believed else "/root/reward_words.jsonl"
    field = "believed_words" if believed else "words"
    for line in open(src):
        if line.strip():
            r = _json.loads(line)
            words_by_key[(r["conv_id"], r["paragraph"], r["attempt_in_paragraph"])] = r[field]
    print(f"{len(words_by_key)} labels from {src}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda", output_hidden_states=True)
    model.eval()
    n_layers = model.config.num_hidden_layers + 1
    rf = {r["name"]: r for r in load_reward_functions()}

    rows, before_a, after_a, before_s, after_s = [], [], [], [], []
    skipped = {"no_reward_pos": 0, "too_short": 0, "no_para": 0}

    for ci, conv in enumerate(load_conversations()):
        cid, fn = conv["conversation_id"], conv["reward_fn"]
        is_syn = rf[fn]["type"] == "syntactic"
        try:
            formatted = tokenizer.apply_chat_template(
                conv["full_messages"], tokenize=False,
                add_generation_prompt=False, enable_thinking=False)
        except TypeError:
            formatted = tokenizer.apply_chat_template(
                conv["full_messages"], tokenize=False, add_generation_prompt=False)

        recs = [r for r in attempt_records(formatted, conv["discovery_paragraph"])
                if believed or r["reward"] is True]
        if not recs:
            continue

        enc = tokenizer(formatted, return_tensors="pt", return_offsets_mapping=True,
                        add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
        input_ids = enc["input_ids"].cuda()
        offsets = enc["offset_mapping"][0].tolist()
        n_tok = len(offsets)

        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            cos = np.empty((n_layers, n_tok), dtype=np.float32)
            cos_sp = np.empty((split_t.shape[0], n_layers, n_tok), dtype=np.float32)
            for l in range(n_layers):
                h = out.hidden_states[l][0].float()
                h = h / h.norm(dim=1, keepdim=True).clamp_min(1e-6)
                cos[l] = (h @ axis_t[l]).cpu().numpy()
                cos_sp[:, l, :] = (split_t[:, l, :] @ h.T).cpu().numpy()
                del h
            del out
            torch.cuda.empty_cache()
        tok_str = [tokenizer.decode([t]) for t in input_ids[0].tolist()]

        for rec in recs:
            body = formatted[rec["assistant_start"]:rec["assistant_end"]]
            cut = max((body.find(t) + len(t) for t in THINK_END if t in body), default=-1)
            if cut < 0:
                skipped["no_para"] += 1
                continue
            p0, p1 = rec["assistant_start"] + cut, rec["assistant_end"]
            in_para = [i for i in range(n_tok)
                       if offsets[i][0] >= p0 and offsets[i][1] <= p1
                       and offsets[i][1] > offsets[i][0]]
            if not in_para:
                skipped["no_para"] += 1
                continue

            if is_syn and not believed:
                chk = SYNTACTIC_TOKEN_CHECKERS[fn]
                reward_pos = {i for i in in_para if chk(tok_str[i])}
            else:
                words = words_by_key.get((cid, rec["paragraph"], rec["attempt_in_paragraph"]), [])
                spans = []
                for w in words:
                    start = 0
                    while (j := formatted.find(w, p0 + start, p1)) != -1:
                        spans.append((j, j + len(w)))
                        start = j - p0 + 1
                reward_pos = {i for i in in_para
                              if any(offsets[i][0] < e and offsets[i][1] > s for s, e in spans)}
            if not reward_pos:
                skipped["no_reward_pos"] += 1
                continue

            lo, hi = min(reward_pos), max(reward_pos)
            bef = [i for i in in_para if i < lo]
            aft = [i for i in in_para if i > hi]
            if len(bef) < MIN_PER_SIDE or len(aft) < MIN_PER_SIDE:
                skipped["too_short"] += 1
                continue

            rows.append({"conv_id": cid, "reward_fn": fn,
                         "type": "syntactic" if is_syn else "semantic",
                         "paragraph": rec["paragraph"],
                         "attempt_in_paragraph": rec["attempt_in_paragraph"],
                         "reward": rec["reward"],
                         "n_before": len(bef), "n_after": len(aft),
                         "n_reward_tokens": len(reward_pos)})
            before_a.append(cos[:, bef].mean(axis=1))
            after_a.append(cos[:, aft].mean(axis=1))
            before_s.append(cos_sp[:, :, bef].mean(axis=2))
            after_s.append(cos_sp[:, :, aft].mean(axis=2))

        if (ci + 1) % 50 == 0:
            print(f"  [{ci+1}/380] {len(rows)} attempts, skipped={skipped}", flush=True)

    print(f"final skipped: {skipped}", flush=True)
    return {"rows": rows,
            "before": np.asarray(before_a, dtype=np.float32),
            "after": np.asarray(after_a, dtype=np.float32),
            "before_split": np.asarray(before_s, dtype=np.float32),
            "after_split": np.asarray(after_s, dtype=np.float32),
            "split_test_fns": split_test_fns}


@app.function(image=image, gpu=GPU, volumes={"/cache": hf_cache}, timeout=60 * 60)
def discovery_token_projections(limit=None, n_random=20, seed=0):
    """PER-TOKEN axis alignment across the discovery paragraph and the one after.

    attempt_projections means over a span and so hides the within-attempt
    trajectory; this keeps every token. Two regions per conversation:

      discovery : every attempt in paragraph == discovery_paragraph. The LAST one
                  is the +1 that ends the paragraph and, per
                  generate_conversations.py's is_discovery_moment flag, is the
                  response where the generator was told to announce the rule.
                  turns.cell files it under pre_lucky, which understates it.
      post1     : paragraph == discovery_paragraph + 1. Exactly one attempt per
                  conversation, always post_earned (RC). This is the SAME span
                  the value axis was built from, so cos against the full axis is
                  in-sample here -- use cos_split21 with the held-out mask.

    Per token we keep the assistant body and the +1/-1 feedback body, tagged with
    role, whether it sits inside the <thinking> block, and the before/after label
    extract_activations.classify_tokens assigns (non-"excluded" only inside the
    post1 rewarded span, by construction).

    cos/hnorm are full-depth; split and random controls are layer 21 only, which
    keeps the output near 100 MB instead of a gigabyte. Raw projection onto the
    axis is recoverable as cos * hnorm.
    """
    import re

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _setup_paths()
    from common.paths import value_axis
    from shared import load_conversations, load_reward_functions, load_reward_labels
    from extract_activations import classify_tokens
    from turns import attempt_records, cell

    axis = np.load(value_axis())
    axis_t = torch.tensor(axis, dtype=torch.float32).cuda()
    axis_t = axis_t / axis_t.norm(dim=1, keepdim=True)

    g = torch.Generator(device="cpu").manual_seed(seed)
    rand_t = torch.randn(n_random, axis_t.shape[1], generator=g).cuda()
    rand_t = rand_t / rand_t.norm(dim=1, keepdim=True)

    import random as _random

    from compute_vector import (N_SPLITS, N_TRAIN, load_activation_means,
                                mean_contrastive_direction, valid_functions)
    act = load_activation_means()
    valid = valid_functions(act)
    split_dirs, split_test_fns = [], []
    for si in range(N_SPLITS):
        rng = _random.Random(si * 42)
        sh = valid.copy()
        rng.shuffle(sh)
        dv = mean_contrastive_direction(act, set(sh[:N_TRAIN]))
        dv = dv / np.linalg.norm(dv, axis=1, keepdims=True).clip(1e-10)
        split_dirs.append(torch.tensor(dv, dtype=torch.float32).cuda())
        split_test_fns.append(sh[N_TRAIN:])
    split21_t = torch.stack([d[21] for d in split_dirs])       # (n_splits, hidden)
    del act

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda", output_hidden_states=True)
    model.eval()
    n_layers = model.config.num_hidden_layers + 1

    rf_by_name = {r["name"]: r for r in load_reward_functions()}
    reward_labels = load_reward_labels()
    conversations = load_conversations()
    if limit:
        conversations = conversations[:limit]
    print(f"{len(conversations)} conversations, {n_layers} layers", flush=True)

    THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL)

    rows, cos_all, hnorm_all, split_all, rand_all = [], [], [], [], []

    for ci, conv in enumerate(conversations):
        dp = conv["discovery_paragraph"]
        fn = conv["reward_fn"]
        conv_idx = int(conv["conversation_id"].split("__conv")[1])

        try:
            formatted = tokenizer.apply_chat_template(
                conv["full_messages"], tokenize=False,
                add_generation_prompt=False, enable_thinking=False)
        except TypeError:
            formatted = tokenizer.apply_chat_template(
                conv["full_messages"], tokenize=False, add_generation_prompt=False)

        enc = tokenizer(formatted, return_tensors="pt", return_offsets_mapping=True,
                        add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
        input_ids = enc["input_ids"].cuda()
        offsets = enc["offset_mapping"][0].tolist()
        n_tok = len(offsets)

        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            cos = np.empty((n_layers, n_tok), dtype=np.float32)
            hnorm = np.empty((n_layers, n_tok), dtype=np.float32)
            sp21 = np.empty((N_SPLITS, n_tok), dtype=np.float32)
            rd21 = np.empty((n_random, n_tok), dtype=np.float32)
            for l in range(n_layers):
                h = out.hidden_states[l][0].float()
                nrm = h.norm(dim=1).clamp_min(1e-6)
                hnorm[l] = nrm.cpu().numpy()
                hu = h / nrm.unsqueeze(1)
                cos[l] = (hu @ axis_t[l]).cpu().numpy()
                if l == 21:
                    sp21[:] = (split21_t @ hu.T).cpu().numpy()
                    rd21[:] = (rand_t @ hu.T).cpu().numpy()
                del h, hu
            del out
            torch.cuda.empty_cache()

        tok_str = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
        axis_label = dict(classify_tokens(
            tok_str, offsets, formatted, conv, fn,
            rf_by_name[fn]["type"], reward_labels, conv_idx))

        def span_positions(a, b):
            return [i for i in range(n_tok)
                    if offsets[i][0] >= a and offsets[i][1] <= b
                    and offsets[i][1] > offsets[i][0]]

        for rec in attempt_records(formatted, dp):
            if rec["paragraph"] == dp:
                region = "discovery"
            elif rec["paragraph"] == dp + 1:
                region = "post1"
            else:
                continue

            body = formatted[rec["assistant_start"]:rec["assistant_end"]]
            think = [(rec["assistant_start"] + m.start(), rec["assistant_start"] + m.end())
                     for m in THINK_RE.finditer(body)]

            spans = [("assistant", rec["assistant_start"], rec["assistant_end"])]
            if rec["fb_start"] is not None:
                spans.append(("feedback", rec["fb_start"], rec["fb_end"]))

            for role, a, b in spans:
                pos_list = span_positions(a, b)
                for j, pos in enumerate(pos_list):
                    o0, o1 = offsets[pos]
                    rows.append({
                        "conv_id": conv["conversation_id"], "reward_fn": fn,
                        "reward_fn_type": rf_by_name[fn]["type"],
                        "region": region, "role": role,
                        "paragraph": rec["paragraph"],
                        "attempt_in_paragraph": rec["attempt_in_paragraph"],
                        "discovery_paragraph": dp, "phase": rec["phase"],
                        "reward": "none" if rec["reward"] is None else str(bool(rec["reward"])),
                        "cell": cell(rec),
                        "is_last_attempt": "",       # filled in after the loop
                        "in_thinking": str(any(s <= o0 and o1 <= e for s, e in think)),
                        "axis_label": axis_label.get(pos, "excluded"),
                        "token": tok_str[pos], "token_pos": pos,
                        "pos_in_span": j, "n_span": len(pos_list),
                        "n_tokens": n_tok,
                    })
                    cos_all.append(cos[:, pos])
                    hnorm_all.append(hnorm[:, pos])
                    split_all.append(sp21[:, pos])
                    rand_all.append(rd21[:, pos])

        if (ci + 1) % 25 == 0:
            print(f"  [{ci+1}/{len(conversations)}] {len(rows)} tokens", flush=True)

    # Flag the final attempt of each (conv, paragraph); in the discovery paragraph
    # that attempt IS the discovery moment.
    last = {}
    for r in rows:
        k = (r["conv_id"], r["paragraph"])
        last[k] = max(last.get(k, 0), r["attempt_in_paragraph"])
    for r in rows:
        r["is_last_attempt"] = str(
            r["attempt_in_paragraph"] == last[(r["conv_id"], r["paragraph"])])

    return {"rows": rows,
            "cos": np.asarray(cos_all, dtype=np.float32),
            "hnorm": np.asarray(hnorm_all, dtype=np.float32),
            "cos_split21": np.asarray(split_all, dtype=np.float32),
            "cos_random21": np.asarray(rand_all, dtype=np.float32),
            "split_test_fns": split_test_fns}


@app.function(image=image, gpu=GPU, volumes={"/cache": hf_cache}, timeout=60 * 60,
              memory=49152)
def rebuild_axis(limit=None, n_random=20, seed=0):
    """Regenerate activation means with CORRECTED span localization; rebuild the axis.

    Upstream extract_activations.py locates the labeled span by text search
    (find(modified[:150])), which matches the user's copy of the paragraph in 62%
    of conversations and computes semantic reward-word offsets in the wrong
    coordinate frame. corrected_spans.corrected_classify_tokens fixes localization
    (structural span via turns.attempt_records, words matched in-span) while
    keeping the paper's design: dp+1 successful attempt, split on
    criterion-satisfying tokens, >=3 tokens per side.

    Single pass: forward every conversation once, cache labeled tokens' hidden
    states (fp16, all layers) in RAM (~15 GB), then in-job compute
      - corrected per-function before/after means (upstream schema),
      - corrected axis (compute_value_axis) + 10 split directions (same seeding),
      - per-token cosines vs corrected axis, shipped axis, split dirs (all
        layers), and n_random random directions (layer 21).

    The means dict is key-ordered like the shipped activation_means.pt so
    valid_functions() ordering — and hence the Random(si*42) split membership —
    matches the shipped splits exactly.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _setup_paths()
    from common.paths import value_axis
    from shared import (SYNTACTIC_TOKEN_CHECKERS, load_conversations,
                        load_reward_functions, load_reward_labels)
    from corrected_spans import corrected_classify_tokens

    shipped_axis = np.load(value_axis())                 # (37, 4096)

    import random as _random

    from compute_vector import (N_SPLITS, N_TRAIN, compute_value_axis,
                                load_activation_means, mean_contrastive_direction,
                                valid_functions)
    shipped_key_order = list(load_activation_means().keys())

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda", output_hidden_states=True)
    model.eval()
    n_layers = model.config.num_hidden_layers + 1
    hidden = model.config.hidden_size

    rf_by_name = {r["name"]: r for r in load_reward_functions()}
    reward_labels = load_reward_labels()
    conversations = load_conversations()
    if limit:
        conversations = conversations[:limit]
    print(f"{len(conversations)} conversations, {n_layers} layers", flush=True)

    rows, act_cache = [], []          # act_cache[i]: (n_i, n_layers, hidden) fp16
    sums = {}                          # fn -> {"before": (n_layers, hidden) f64, ...}
    counts = {}
    skipped_convs = []

    for ci, conv in enumerate(conversations):
        fn = conv["reward_fn"]
        conv_idx = int(conv["conversation_id"].split("__conv")[1])
        try:
            formatted = tokenizer.apply_chat_template(
                conv["full_messages"], tokenize=False,
                add_generation_prompt=False, enable_thinking=False)
        except TypeError:
            formatted = tokenizer.apply_chat_template(
                conv["full_messages"], tokenize=False, add_generation_prompt=False)

        enc = tokenizer(formatted, return_tensors="pt", return_offsets_mapping=True,
                        add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
        input_ids = enc["input_ids"].cuda()
        offsets = enc["offset_mapping"][0].tolist()
        token_strings = [tokenizer.decode([t]) for t in input_ids[0].tolist()]

        labels = [(p, l) for p, l in corrected_classify_tokens(
            token_strings, offsets, formatted, conv, fn,
            rf_by_name[fn]["type"], reward_labels, conv_idx,
            SYNTACTIC_TOKEN_CHECKERS) if l != "excluded"]
        if not labels:
            skipped_convs.append(conv["conversation_id"])
            continue
        pos_list = [p for p, _ in labels]

        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            # (n_layers, n_labeled, hidden) -> (n_labeled, n_layers, hidden)
            acts = torch.stack([out.hidden_states[l][0, pos_list].float()
                                for l in range(n_layers)])
            acts = acts.permute(1, 0, 2).cpu()
            del out
            torch.cuda.empty_cache()

        act_cache.append(acts.to(torch.float16).numpy())
        if fn not in sums:
            sums[fn] = {"before": np.zeros((n_layers, hidden), dtype=np.float64),
                        "after": np.zeros((n_layers, hidden), dtype=np.float64)}
            counts[fn] = {"before": 0, "after": 0}
        a64 = acts.numpy().astype(np.float64)
        for i, (p, l) in enumerate(labels):
            sums[fn][l] += a64[i]
            counts[fn][l] += 1
            rows.append({"conv_id": conv["conversation_id"], "reward_fn": fn,
                         "reward_fn_type": rf_by_name[fn]["type"], "label": l,
                         "token": token_strings[p], "token_pos": p,
                         "n_tokens": len(token_strings)})

        if (ci + 1) % 25 == 0:
            print(f"  [{ci+1}/{len(conversations)}] {len(rows)} labeled tokens", flush=True)

    print(f"skipped (no labels): {len(skipped_convs)} {skipped_convs}", flush=True)

    # ---- corrected means, shipped key order, upstream schema -----------------
    means = {}
    for fn in shipped_key_order:
        if fn not in sums:
            continue
        b, a = counts[fn]["before"], counts[fn]["after"]
        means[fn] = {
            "before_mean": torch.tensor(sums[fn]["before"] / max(b, 1), dtype=torch.float32),
            "after_mean": torch.tensor(sums[fn]["after"] / max(a, 1), dtype=torch.float32),
            "before_count": b, "after_count": a,
        }
    for fn in sums:                     # any fn not in the shipped file (none expected)
        if fn not in means:
            print(f"  WARNING: {fn} not in shipped key order; appending")
            b, a = counts[fn]["before"], counts[fn]["after"]
            means[fn] = {"before_mean": torch.tensor(sums[fn]["before"] / max(b, 1),
                                                     dtype=torch.float32),
                         "after_mean": torch.tensor(sums[fn]["after"] / max(a, 1),
                                                    dtype=torch.float32),
                         "before_count": b, "after_count": a}

    # ---- corrected axis + split directions (upstream code, same seeding) ----
    axis_corr = compute_value_axis(means)                # (n_layers, hidden) f64
    valid = valid_functions(means)
    split_dirs, split_test_fns = [], []
    for si in range(N_SPLITS):
        rng = _random.Random(si * 42)
        sh = valid.copy()
        rng.shuffle(sh)
        dv = mean_contrastive_direction(means, set(sh[:N_TRAIN]))
        dv = dv / np.linalg.norm(dv, axis=1, keepdims=True).clip(1e-10)
        split_dirs.append(dv.astype(np.float32))
        split_test_fns.append(sh[N_TRAIN:])
    split_u = np.stack(split_dirs)                       # (n_splits, n_layers, hidden)

    def unit(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-10)

    corr_u = unit(axis_corr).astype(np.float32)
    ship_u = unit(shipped_axis).astype(np.float32)
    g = torch.Generator(device="cpu").manual_seed(seed)
    rand_u = torch.randn(n_random, hidden, generator=g).numpy()
    rand_u = unit(rand_u).astype(np.float32)

    # ---- per-token cosines from the cache ------------------------------------
    cos_c, cos_s, cos_sp, cos_r = [], [], [], []
    for arr in act_cache:
        h = arr.astype(np.float32)                       # (n, n_layers, hidden)
        hu = unit(h)
        cos_c.append(np.einsum("ntd,td->nt", hu, corr_u))
        cos_s.append(np.einsum("ntd,td->nt", hu, ship_u))
        cos_sp.append(np.einsum("ntd,std->nst", hu, split_u))
        cos_r.append(hu[:, 21, :] @ rand_u.T)
    cos_c = np.concatenate(cos_c); cos_s = np.concatenate(cos_s)
    cos_sp = np.concatenate(cos_sp); cos_r = np.concatenate(cos_r)
    print(f"{cos_c.shape[0]} labeled tokens; "
          f"axis cos(corr, shipped) @L21 = "
          f"{float(np.dot(corr_u[21], ship_u[21])):.4f}", flush=True)

    return {"rows": rows,
            "cos": cos_c.astype(np.float32),
            "cos_shipped": cos_s.astype(np.float32),
            "cos_split": cos_sp.astype(np.float32),
            "cos_random": cos_r.astype(np.float32),
            "axis_corrected": axis_corr,
            "means": {fn: {"before_mean": d["before_mean"].numpy(),
                           "after_mean": d["after_mean"].numpy(),
                           "before_count": d["before_count"],
                           "after_count": d["after_count"]}
                      for fn, d in means.items()},
            "split_test_fns": split_test_fns,
            "skipped_convs": skipped_convs}


@app.function(image=image, gpu=GPU, volumes={"/cache": hf_cache}, timeout=60 * 60)
def attempt_split_projections(limit=None):
    """Pre/post-split span means for EVERY attempt, vs the CORRECTED axis.

    For each assistant attempt (post-thinking paragraph body), split the span at
    its divider tokens and mean the per-token cosines on each side. Two split
    channels per attempt:

      actual   : rewarded attempts only -- the criterion-satisfying tokens
                 (syntactic checkers / codex reward_words.jsonl). Ground truth.
      believed : all attempts with a non-diffuse believed_words.jsonl entry --
                 what the model SAID it was trying (works for failures, where no
                 actual criterion token exists).

    Projections stored per side: corrected axis (all layers), corrected held-out
    split directions (10, all layers; built from activation_means_corrected.pt
    with the upstream Random(si*42) seeding), and the shipped axis (all layers)
    for continuity with the pre-correction runs.

    Tags allow the pre-discovery / discovery / post-discovery x outcome analysis:
    paragraph, discovery_paragraph, reward, cell, is_discovery_moment.
    """
    import json as _json
    import re as _re

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _setup_paths()
    from common.paths import value_axis
    from shared import (SYNTACTIC_TOKEN_CHECKERS, load_conversations,
                        load_reward_functions)
    from turns import attempt_records, cell

    MIN_PER_SIDE = 3
    THINK_END = ("</think>", "</thinking>")

    # Shipped axis + corrected axis / corrected split directions.
    shipped = np.load(value_axis())
    import random as _random

    from compute_vector import (N_SPLITS, N_TRAIN, compute_value_axis,
                                mean_contrastive_direction, valid_functions)
    means = torch.load("/root/activation_means_corrected.pt",
                       map_location="cpu", weights_only=False)
    corr = compute_value_axis(means)
    valid = valid_functions(means)
    split_dirs, split_test_fns = [], []
    for si in range(N_SPLITS):
        rng = _random.Random(si * 42)
        sh = valid.copy()
        rng.shuffle(sh)
        dv = mean_contrastive_direction(means, set(sh[:N_TRAIN]))
        dv = dv / np.linalg.norm(dv, axis=1, keepdims=True).clip(1e-10)
        split_dirs.append(torch.tensor(dv, dtype=torch.float32).cuda())
        split_test_fns.append(sh[N_TRAIN:])
    split_t = torch.stack(split_dirs)                       # (10, 37, 4096)
    del means

    def unit_t(a):
        t = torch.tensor(a, dtype=torch.float32).cuda()
        return t / t.norm(dim=1, keepdim=True).clamp_min(1e-10)

    corr_t, ship_t = unit_t(corr), unit_t(shipped)

    actual_words, believed_words = {}, {}
    for line in open("/root/reward_words.jsonl"):
        if line.strip():
            r = _json.loads(line)
            actual_words[(r["conv_id"], r["paragraph"], r["attempt_in_paragraph"])] = r["words"]
    for line in open("/root/believed_words.jsonl"):
        if line.strip():
            r = _json.loads(line)
            if not r.get("diffuse") and r.get("believed_words"):
                believed_words[(r["conv_id"], r["paragraph"], r["attempt_in_paragraph"])] = (
                    r["believed_words"], r.get("confidence", 0))
    print(f"{len(actual_words)} actual-word labels, "
          f"{len(believed_words)} believed-word labels", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda", output_hidden_states=True)
    model.eval()
    n_layers = model.config.num_hidden_layers + 1
    rf = {r["name"]: r for r in load_reward_functions()}

    conversations = load_conversations()
    if limit:
        conversations = conversations[:limit]
    print(f"{len(conversations)} conversations", flush=True)

    rows = []
    before_c, after_c, before_s, after_s, before_sp, after_sp = [], [], [], [], [], []
    skipped = {"no_para": 0, "no_split_tokens": 0, "too_short": 0}

    for ci, conv in enumerate(conversations):
        cid, fn = conv["conversation_id"], conv["reward_fn"]
        dp = conv["discovery_paragraph"]
        is_syn = rf[fn]["type"] == "syntactic"
        try:
            formatted = tokenizer.apply_chat_template(
                conv["full_messages"], tokenize=False,
                add_generation_prompt=False, enable_thinking=False)
        except TypeError:
            formatted = tokenizer.apply_chat_template(
                conv["full_messages"], tokenize=False, add_generation_prompt=False)

        enc = tokenizer(formatted, return_tensors="pt", return_offsets_mapping=True,
                        add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
        input_ids = enc["input_ids"].cuda()
        offsets = enc["offset_mapping"][0].tolist()
        n_tok = len(offsets)

        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            cos_c = np.empty((n_layers, n_tok), dtype=np.float32)
            cos_s = np.empty((n_layers, n_tok), dtype=np.float32)
            cos_sp = np.empty((split_t.shape[0], n_layers, n_tok), dtype=np.float32)
            for l in range(n_layers):
                h = out.hidden_states[l][0].float()
                h = h / h.norm(dim=1, keepdim=True).clamp_min(1e-6)
                cos_c[l] = (h @ corr_t[l]).cpu().numpy()
                cos_s[l] = (h @ ship_t[l]).cpu().numpy()
                cos_sp[:, l, :] = (split_t[:, l, :] @ h.T).cpu().numpy()
                del h
            del out
            torch.cuda.empty_cache()
        tok_str = [tokenizer.decode([t]) for t in input_ids[0].tolist()]

        for rec in attempt_records(formatted, dp):
            body = formatted[rec["assistant_start"]:rec["assistant_end"]]
            cut = max((body.find(t) + len(t) for t in THINK_END if t in body), default=-1)
            if cut < 0:
                skipped["no_para"] += 1
                continue
            p0, p1 = rec["assistant_start"] + cut, rec["assistant_end"]
            in_para = [i for i in range(n_tok)
                       if offsets[i][0] >= p0 and offsets[i][1] <= p1
                       and offsets[i][1] > offsets[i][0]]
            if not in_para:
                skipped["no_para"] += 1
                continue
            key = (cid, rec["paragraph"], rec["attempt_in_paragraph"])

            channels = []
            if rec["reward"] is True:
                if is_syn:
                    chk = SYNTACTIC_TOKEN_CHECKERS[fn]
                    pos = {i for i in in_para if chk(tok_str[i])}
                else:
                    spans = []
                    for w in actual_words.get(key, []):
                        for m in _re.finditer(_re.escape(w), formatted[p0:p1], _re.IGNORECASE):
                            spans.append((p0 + m.start(), p0 + m.end()))
                    pos = {i for i in in_para
                           if any(offsets[i][0] < e and offsets[i][1] > s for s, e in spans)}
                channels.append(("actual", pos, 0))
            if key in believed_words:
                words, conf = believed_words[key]
                spans = []
                for w in words:
                    for m in _re.finditer(_re.escape(w), formatted[p0:p1], _re.IGNORECASE):
                        spans.append((p0 + m.start(), p0 + m.end()))
                pos = {i for i in in_para
                       if any(offsets[i][0] < e and offsets[i][1] > s for s, e in spans)}
                channels.append(("believed", pos, conf))

            for channel, pos, conf in channels:
                if not pos:
                    skipped["no_split_tokens"] += 1
                    continue
                lo, hi = min(pos), max(pos)
                bef = [i for i in in_para if i < lo]
                aft = [i for i in in_para if i > hi]
                if len(bef) < MIN_PER_SIDE or len(aft) < MIN_PER_SIDE:
                    skipped["too_short"] += 1
                    continue
                para_phase = ("pre" if rec["paragraph"] < dp
                              else "disc" if rec["paragraph"] == dp else "post")
                rows.append({
                    "conv_id": cid, "reward_fn": fn,
                    "reward_fn_type": "syntactic" if is_syn else "semantic",
                    "channel": channel, "paragraph": rec["paragraph"],
                    "attempt_in_paragraph": rec["attempt_in_paragraph"],
                    "discovery_paragraph": dp, "para_phase": para_phase,
                    "reward": "none" if rec["reward"] is None else str(bool(rec["reward"])),
                    "cell": cell(rec),
                    "is_discovery_moment": str(rec["paragraph"] == dp
                                               and rec["reward"] is True),
                    "believed_confidence": conf,
                    "n_before": len(bef), "n_after": len(aft),
                    "n_split_tokens": len(pos),
                })
                before_c.append(cos_c[:, bef].mean(axis=1))
                after_c.append(cos_c[:, aft].mean(axis=1))
                before_s.append(cos_s[:, bef].mean(axis=1))
                after_s.append(cos_s[:, aft].mean(axis=1))
                before_sp.append(cos_sp[:, :, bef].mean(axis=2))
                after_sp.append(cos_sp[:, :, aft].mean(axis=2))

        if (ci + 1) % 25 == 0:
            print(f"  [{ci+1}/{len(conversations)}] {len(rows)} rows, "
                  f"skipped={skipped}", flush=True)

    print(f"final skipped: {skipped}", flush=True)
    return {"rows": rows,
            "before": np.asarray(before_c, dtype=np.float32),
            "after": np.asarray(after_c, dtype=np.float32),
            "before_shipped": np.asarray(before_s, dtype=np.float32),
            "after_shipped": np.asarray(after_s, dtype=np.float32),
            "before_split": np.asarray(before_sp, dtype=np.float32),
            "after_split": np.asarray(after_sp, dtype=np.float32),
            "split_test_fns": split_test_fns}


@app.function(image=image, gpu=GPU, volumes={"/cache": hf_cache}, timeout=60 * 60,
              memory=32768)
def assistant_headers(limit=None):
    """Assistant-header measurements: cosines + RAW mean activations per header.

    The header span is the byte-identical text between the previous message's
    content and the attempt's first content token:
    "<|im_end|>\n<|im_start|>assistant\n" plus the opening "<thinking>"/"<think>"
    tag when the body starts with one (has_think_open tag records this). Being
    content-matched, differences across headers are attributable to state.

    Per header we store:
      hdr_cos    (37,)      cosine vs the corrected axis, all layers
      hdr_split  (10, 37)   cosine vs the corrected held-out split directions
      hdr_raw    (37, 4096) RAW mean activation over the span, fp16 -- so any
                            difference-of-means vector over header cells is a
                            CPU one-liner afterwards, no further GPU passes.

    Tags: para_phase (pre/disc/post), prev (minus1 / paragraph / system / ...),
    cell, is_discovery_moment, paragraph, attempt_in_paragraph.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _setup_paths()
    from shared import load_conversations
    from turns import attempt_records, cell, parse_turns

    THINK_OPEN = ("<thinking>", "<think>")

    import random as _random

    from compute_vector import (N_SPLITS, N_TRAIN, compute_value_axis,
                                mean_contrastive_direction, valid_functions)
    means_c = torch.load("/root/activation_means_corrected.pt",
                         map_location="cpu", weights_only=False)
    corr = compute_value_axis(means_c)
    valid = valid_functions(means_c)
    split_dirs, split_test_fns = [], []
    for si in range(N_SPLITS):
        rng = _random.Random(si * 42)
        sh = valid.copy()
        rng.shuffle(sh)
        dv = mean_contrastive_direction(means_c, set(sh[:N_TRAIN]))
        dv = dv / np.linalg.norm(dv, axis=1, keepdims=True).clip(1e-10)
        split_dirs.append(torch.tensor(dv, dtype=torch.float32).cuda())
        split_test_fns.append(sh[N_TRAIN:])
    split_t = torch.stack(split_dirs)
    corr_t = torch.tensor(corr, dtype=torch.float32).cuda()
    corr_t = corr_t / corr_t.norm(dim=1, keepdim=True).clamp_min(1e-10)
    del means_c

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda", output_hidden_states=True)
    model.eval()
    n_layers = model.config.num_hidden_layers + 1
    hidden = model.config.hidden_size

    conversations = load_conversations()
    if limit:
        conversations = conversations[:limit]
    print(f"{len(conversations)} conversations", flush=True)

    rows, hdr_cos, hdr_split, hdr_raw = [], [], [], []

    for ci, conv in enumerate(conversations):
        cid, fn = conv["conversation_id"], conv["reward_fn"]
        dp = conv["discovery_paragraph"]
        try:
            formatted = tokenizer.apply_chat_template(
                conv["full_messages"], tokenize=False,
                add_generation_prompt=False, enable_thinking=False)
        except TypeError:
            formatted = tokenizer.apply_chat_template(
                conv["full_messages"], tokenize=False, add_generation_prompt=False)

        enc = tokenizer(formatted, return_tensors="pt", return_offsets_mapping=True,
                        add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
        input_ids = enc["input_ids"].cuda()
        offsets = enc["offset_mapping"][0].tolist()
        n_tok = len(offsets)

        # ---- locate every header span BEFORE the forward pass ----------------
        turns = parse_turns(formatted)
        by_start = {t["body_start"]: i for i, t in enumerate(turns)}

        def span_positions(a, b):
            return [i for i in range(n_tok)
                    if offsets[i][0] >= a and offsets[i][1] <= b
                    and offsets[i][1] > offsets[i][0]]

        headers = []           # (row_dict, positions)
        for rec in attempt_records(formatted, dp):
            ti = by_start.get(rec["assistant_start"])
            if ti is None or ti == 0:
                continue
            prev = turns[ti - 1]
            txt = prev["text"].strip()
            prev_tag = ("minus1" if txt == "-1" else "plus1" if txt == "+1"
                        else "paragraph" if txt.startswith("Paragraph")
                        else "moving_on" if txt.startswith("Moving")
                        else "system" if prev["role"] == "system" else "other")
            body = formatted[rec["assistant_start"]:rec["assistant_end"]]
            think_open = next((t for t in THINK_OPEN if body.startswith(t)), None)
            span_end = rec["assistant_start"] + (len(think_open) if think_open else 0)
            pos = span_positions(prev["body_end"], span_end)
            if not pos:
                continue
            para_phase = ("pre" if rec["paragraph"] < dp
                          else "disc" if rec["paragraph"] == dp else "post")
            headers.append(({
                "conv_id": cid, "reward_fn": fn,
                "paragraph": rec["paragraph"],
                "attempt_in_paragraph": rec["attempt_in_paragraph"],
                "discovery_paragraph": dp, "para_phase": para_phase,
                "reward": ("none" if rec["reward"] is None
                           else str(bool(rec["reward"]))),
                "cell": cell(rec),
                "is_discovery_moment": str(rec["paragraph"] == dp
                                           and rec["reward"] is True),
                "prev": prev_tag,
                "has_think_open": str(think_open is not None),
                "n_header": len(pos),
            }, pos))

        if not headers:
            continue

        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            k = len(headers)
            cos_h = np.empty((k, n_layers), dtype=np.float32)
            sp_h = np.empty((k, split_t.shape[0], n_layers), dtype=np.float32)
            raw_h = np.empty((k, n_layers, hidden), dtype=np.float16)
            for l in range(n_layers):
                h = out.hidden_states[l][0].float()
                hu = h / h.norm(dim=1, keepdim=True).clamp_min(1e-6)
                cos_l = (hu @ corr_t[l]).cpu().numpy()
                sp_l = (split_t[:, l, :] @ hu.T).cpu().numpy()
                h_cpu = h.cpu().numpy()
                for j, (_, pos) in enumerate(headers):
                    cos_h[j, l] = cos_l[pos].mean()
                    sp_h[j, :, l] = sp_l[:, pos].mean(axis=1)
                    raw_h[j, l] = h_cpu[pos].mean(axis=0).astype(np.float16)
                del h, hu
            del out
            torch.cuda.empty_cache()

        for j, (row, _) in enumerate(headers):
            rows.append(row)
            hdr_cos.append(cos_h[j])
            hdr_split.append(sp_h[j])
            hdr_raw.append(raw_h[j])

        if (ci + 1) % 25 == 0:
            print(f"  [{ci+1}/{len(conversations)}] {len(rows)} headers", flush=True)

    return {"rows": rows,
            "hdr_cos": np.asarray(hdr_cos, dtype=np.float32),
            "hdr_split": np.asarray(hdr_split, dtype=np.float32),
            "hdr_raw": np.asarray(hdr_raw, dtype=np.float16),
            "split_test_fns": split_test_fns}


@app.function(image=image, gpu=GPU, volumes={"/cache": hf_cache}, timeout=60 * 60)
def extended_retry_headers(n_random=20, seed=0):
    """Header cosines for the extended-retry corpus (20 scripted fails, 2 arms).

    Measures the same 7-token content-matched header span as assistant_headers
    (chat markers + <thinking> open tag) at every attempt depth 1..20, against
    the corrected axis (all layers), the shipped axis (all layers), and
    n_random random directions at layer 21. No held-out machinery: synthetic
    conversations have no reward function.
    """
    import json as _json

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _setup_paths()
    from common.paths import value_axis
    from turns import parse_turns

    THINK_OPEN = ("<thinking>", "<think>")

    from compute_vector import compute_value_axis
    means_c = torch.load("/root/activation_means_corrected.pt",
                         map_location="cpu", weights_only=False)
    corr = compute_value_axis(means_c)
    del means_c
    shipped = np.load(value_axis())

    def unit_t(a):
        t = torch.tensor(a, dtype=torch.float32).cuda()
        return t / t.norm(dim=1, keepdim=True).clamp_min(1e-10)

    corr_t, ship_t = unit_t(corr), unit_t(shipped)
    g = torch.Generator(device="cpu").manual_seed(seed)
    rand_t = torch.randn(n_random, corr_t.shape[1], generator=g).cuda()
    rand_t = rand_t / rand_t.norm(dim=1, keepdim=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda", output_hidden_states=True)
    model.eval()
    n_layers = model.config.num_hidden_layers + 1

    conversations = _json.loads(open("/root/extended_retries.json").read())
    print(f"{len(conversations)} synthetic conversations", flush=True)

    rows, hdr_cos, hdr_ship, hdr_rand = [], [], [], []

    for ci, conv in enumerate(conversations):
        try:
            formatted = tokenizer.apply_chat_template(
                conv["full_messages"], tokenize=False,
                add_generation_prompt=False, enable_thinking=False)
        except TypeError:
            formatted = tokenizer.apply_chat_template(
                conv["full_messages"], tokenize=False, add_generation_prompt=False)
        enc = tokenizer(formatted, return_tensors="pt", return_offsets_mapping=True,
                        add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
        input_ids = enc["input_ids"].cuda()
        offsets = enc["offset_mapping"][0].tolist()
        n_tok = len(offsets)

        turns = parse_turns(formatted)
        headers, depth = [], 0
        for ti, t in enumerate(turns):
            if t["role"] != "assistant" or ti == 0:
                continue
            depth += 1
            prev = turns[ti - 1]
            body = formatted[t["body_start"]:t["body_end"]]
            think = next((tg for tg in THINK_OPEN if body.startswith(tg)), None)
            b = t["body_start"] + (len(think) if think else 0)
            pos = [i for i in range(n_tok)
                   if offsets[i][0] >= prev["body_end"] and offsets[i][1] <= b
                   and offsets[i][1] > offsets[i][0]]
            if pos:
                headers.append((depth, prev["text"].strip(), pos))

        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            k = len(headers)
            cc = np.empty((k, n_layers), dtype=np.float32)
            cs = np.empty((k, n_layers), dtype=np.float32)
            cr = np.empty((k, rand_t.shape[0]), dtype=np.float32)
            for l in range(n_layers):
                h = out.hidden_states[l][0].float()
                hu = h / h.norm(dim=1, keepdim=True).clamp_min(1e-6)
                a = (hu @ corr_t[l]).cpu().numpy()
                s = (hu @ ship_t[l]).cpu().numpy()
                if l == 21:
                    r = (rand_t @ hu.T).cpu().numpy()
                for j, (_, _, pos) in enumerate(headers):
                    cc[j, l] = a[pos].mean()
                    cs[j, l] = s[pos].mean()
                    if l == 21:
                        cr[j] = r[:, pos].mean(axis=1)
                del h, hu
            del out
            torch.cuda.empty_cache()

        for j, (depth, prev_txt, pos) in enumerate(headers):
            rows.append({"conv_id": conv["conversation_id"], "arm": conv["arm"],
                         "paragraph_id": conv["paragraph_id"], "depth": depth,
                         "prev": ("minus1" if prev_txt == "-1" else "paragraph"),
                         "n_header": len(pos)})
            hdr_cos.append(cc[j]); hdr_ship.append(cs[j]); hdr_rand.append(cr[j])

        if (ci + 1) % 10 == 0:
            print(f"  [{ci+1}/{len(conversations)}] {len(rows)} headers", flush=True)

    return {"rows": rows,
            "hdr_cos": np.asarray(hdr_cos, dtype=np.float32),
            "hdr_shipped": np.asarray(hdr_ship, dtype=np.float32),
            "hdr_random21": np.asarray(hdr_rand, dtype=np.float32)}


@app.function(image=image, gpu=GPU, volumes={"/cache": hf_cache}, timeout=2 * 60 * 60)
def prefill_probe_projections(limit=None, n_random=20, seed=0,
                              json_path="/root/prefill_probes.json"):
    """Per-token corrected-axis cosines on prefilled final assistant messages.

    Conversations from /root/prefill_probes.json (prefill_probes_build.py):
    real corpus contexts + a prefilled final assistant message whose segment
    char-ranges (thinking / body / tail, or answer) are provided. Measures every
    token of the final assistant body, tagged by segment, vs the corrected axis
    (all layers), the shipped axis (all layers), and n_random random directions
    at layer 21.
    """
    import json as _json

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _setup_paths()
    from common.paths import value_axis
    from turns import parse_turns

    from compute_vector import compute_value_axis
    means_c = torch.load("/root/activation_means_corrected.pt",
                         map_location="cpu", weights_only=False)
    corr = compute_value_axis(means_c)
    del means_c
    shipped = np.load(value_axis())

    def unit_t(a):
        t = torch.tensor(a, dtype=torch.float32).cuda()
        return t / t.norm(dim=1, keepdim=True).clamp_min(1e-10)

    corr_t, ship_t = unit_t(corr), unit_t(shipped)
    g = torch.Generator(device="cpu").manual_seed(seed)
    rand_t = torch.randn(n_random, corr_t.shape[1], generator=g).cuda()
    rand_t = rand_t / rand_t.norm(dim=1, keepdim=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda", output_hidden_states=True)
    model.eval()
    n_layers = model.config.num_hidden_layers + 1

    conversations = _json.loads(open(json_path).read())
    if limit:
        conversations = conversations[:limit]
    print(f"{len(conversations)} prefilled conversations", flush=True)

    rows, cos_all, ship_all, rand_all = [], [], [], []

    for ci, conv in enumerate(conversations):
        try:
            formatted = tokenizer.apply_chat_template(
                conv["full_messages"], tokenize=False,
                add_generation_prompt=False, enable_thinking=False)
        except TypeError:
            formatted = tokenizer.apply_chat_template(
                conv["full_messages"], tokenize=False, add_generation_prompt=False)
        enc = tokenizer(formatted, return_tensors="pt", return_offsets_mapping=True,
                        add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
        input_ids = enc["input_ids"].cuda()
        offsets = enc["offset_mapping"][0].tolist()
        n_tok = len(offsets)

        turns = parse_turns(formatted)
        asst = [t for t in turns if t["role"] == "assistant"][-1]
        content = conv["full_messages"][-1]["content"]
        # The template injects an empty <think>\n\n</think> block before the final
        # assistant message's content, so locate the content within the body.
        a0 = formatted.find(content, asst["body_start"])
        assert a0 != -1, "prefill not verbatim in render"

        def seg_of(pos):
            o0, o1 = offsets[pos]
            if not (a0 <= o0 and o1 <= a0 + len(content)) or o1 <= o0:
                return None
            for name, (s, e) in conv["segments"].items():
                if o0 >= a0 + s and o1 <= a0 + e:
                    return name
            return "other"

        keep = [(p, seg_of(p)) for p in range(n_tok)]
        keep = [(p, s) for p, s in keep if s]

        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            k = len(keep)
            cc = np.empty((k, n_layers), dtype=np.float32)
            cs = np.empty((k, n_layers), dtype=np.float32)
            cr = np.empty((k, rand_t.shape[0]), dtype=np.float32)
            pos_idx = [p for p, _ in keep]
            for l in range(n_layers):
                h = out.hidden_states[l][0].float()
                hu = h / h.norm(dim=1, keepdim=True).clamp_min(1e-6)
                cc[:, l] = (hu @ corr_t[l]).cpu().numpy()[pos_idx]
                cs[:, l] = (hu @ ship_t[l]).cpu().numpy()[pos_idx]
                if l == 21:
                    cr[:, :] = (rand_t @ hu.T).cpu().numpy()[:, pos_idx].T
                del h, hu
            del out
            torch.cuda.empty_cache()

        tok_str = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
        for j, (p, s) in enumerate(keep):
            rows.append({"conv_id": conv["conversation_id"],
                         "base_conv": conv["base_conv"], "family": conv["family"],
                         "cond": conv["cond"], "phase": conv["phase"],
                         "reward_fn": conv["reward_fn"], "segment": s,
                         "token": tok_str[p], "token_pos": p, "n_tokens": n_tok})
            cos_all.append(cc[j]); ship_all.append(cs[j]); rand_all.append(cr[j])

        if (ci + 1) % 50 == 0:
            print(f"  [{ci+1}/{len(conversations)}] {len(rows)} tokens", flush=True)

    return {"rows": rows,
            "cos": np.asarray(cos_all, dtype=np.float32),
            "cos_shipped": np.asarray(ship_all, dtype=np.float32),
            "cos_random21": np.asarray(rand_all, dtype=np.float32)}


@app.function(image=image, gpu=GPU, volumes={"/cache": hf_cache}, timeout=2 * 60 * 60)
def steering_probe(n_convs=15, max_new_tokens=300, temperature=0.7, top_p=0.9,
                   layer=21, seed=42):
    """Steer the value axis while the model answers a confidence probe.

    Paradigm matched to experiments/tasks/verbalized_confidence_steering.py:
    unit direction at `layer` added as alpha*d to the residual stream via a
    forward hook on decoder block layer-1, at every position, throughout
    generation; alphas -75..+75; temperature 0.7, top_p 0.9. (Their make_hook
    assumes the block returns a tensor; ours is tuple-safe, same semantics.)

    Contexts: real ICRL conversations truncated at two states --
      early : right after the first "-1" of paragraph 1 (rule unknown)
      post  : right after the "+1" of paragraph dp+1 (rule known & confirmed)
    -- then a user probe: rate confidence 0-10, then explain.

    Dissociation readout per generation: parsed rating (confidence channel) vs
    response length in tokens (wrap-up / episode-closure channel).
    Directions: shipped axis (the paper's alleged vector) AND corrected axis.
    """
    import re as _re

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _setup_paths()
    from common.paths import value_axis
    from shared import load_conversations
    from compute_vector import compute_value_axis

    PROBE = ("Before we continue: how confident are you right now that you know "
             "the hidden criterion? Reply with a single number from 0 to 10 on "
             "the first line, then briefly explain your reasoning.")
    ALPHAS = [-75, -50, -25, 0, 25, 50, 75]

    ship = np.load(value_axis())[layer]
    means_c = torch.load("/root/activation_means_corrected.pt",
                         map_location="cpu", weights_only=False)
    corr = compute_value_axis(means_c)[layer]
    del means_c
    dirs = {"shipped": torch.tensor(ship / np.linalg.norm(ship), dtype=torch.bfloat16),
            "corrected": torch.tensor(corr / np.linalg.norm(corr), dtype=torch.bfloat16)}

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    # ---- build prefixes ------------------------------------------------------
    conversations = load_conversations()
    prefixes = []          # (conv_id, state, messages)
    for conv in conversations:
        if len(prefixes) >= 2 * n_convs:
            break
        # dp <= 2 keeps the post-state prefix short (~2-3k tokens): fits the
        # A10G, and shrinks the position gap between the two states.
        if conv["discovery_paragraph"] > 2:
            continue
        msgs = conv["full_messages"]
        # early: first "-1" after the first assistant turn
        early = next((i for i, m in enumerate(msgs)
                      if m["role"] == "user" and m["content"].strip() == "-1"), None)
        if early is None:
            continue
        # post: the "+1" that ends paragraph dp+1
        dp = conv["discovery_paragraph"]
        para = 0
        post = None
        for i, m in enumerate(msgs):
            if m["role"] == "user" and m["content"].lstrip().startswith("Paragraph"):
                para += 1
            if (m["role"] == "user" and m["content"].strip() == "+1"
                    and para == dp + 1):
                post = i
                break
        if post is None:
            continue
        cid = conv["conversation_id"]
        prefixes.append((cid, "early", msgs[:early + 1] + [{"role": "user", "content": PROBE}]))
        prefixes.append((cid, "post", msgs[:post + 1] + [{"role": "user", "content": PROBE}]))
    print(f"{len(prefixes)} prefixes ({len(prefixes)//2} conversations)", flush=True)

    def make_hook(direction, alpha):
        d = direction.cuda()
        def hook_fn(module, inp, output):
            if isinstance(output, tuple):
                return (output[0] + alpha * d,) + output[1:]
            return output + alpha * d
        return hook_fn

    texts = [tokenizer.apply_chat_template(m, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
             for _, _, m in prefixes]
    # Chunk into length-sorted sub-batches; with dp<=2 prefixes (~3k tokens)
    # batch 4 keeps prefill + KV well inside the 22 GiB A10G.
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    CHUNK = 4
    chunks = [order[i:i + CHUNK] for i in range(0, len(order), CHUNK)]

    rows = []
    conds = [("shipped", a) for a in ALPHAS] + \
            [("corrected", a) for a in ALPHAS if a != 0]
    for ci_, (dname, alpha) in enumerate(conds):
        handle = None
        if alpha != 0:
            handle = model.model.layers[layer - 1].register_forward_hook(
                make_hook(dirs[dname], alpha))
        try:
            for chunk in chunks:
                enc = tokenizer([texts[i] for i in chunk], return_tensors="pt",
                                padding=True, add_special_tokens=False)
                input_ids = enc["input_ids"].cuda()
                attn = enc["attention_mask"].cuda()
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                with torch.no_grad():
                    out = model.generate(
                        input_ids, attention_mask=attn,
                        max_new_tokens=max_new_tokens, do_sample=True,
                        temperature=temperature, top_p=top_p,
                        pad_token_id=tokenizer.pad_token_id)
                gen = out[:, input_ids.shape[1]:]
                for row_j, i in enumerate(chunk):
                    cid, state, _ = prefixes[i]
                    ids = gen[row_j].tolist()
                    if tokenizer.eos_token_id in ids:
                        ids = ids[:ids.index(tokenizer.eos_token_id)]
                    text = tokenizer.decode(ids, skip_special_tokens=True).strip()
                    mnum = _re.search(r"\b(10|[0-9])\b", text[:80])
                    rows.append({"conv_id": cid, "state": state, "direction": dname,
                                 "alpha": alpha, "n_tokens": len(ids),
                                 "ended": len(ids) < max_new_tokens,
                                 "rating": int(mnum.group(1)) if mnum else -1,
                                 "text": text})
                del out, input_ids, attn
                torch.cuda.empty_cache()
        finally:
            if handle is not None:
                handle.remove()
        done = sum(1 for r in rows if r["rating"] >= 0)
        print(f"  [{ci_+1}/{len(conds)}] {dname} alpha={alpha:+d}  "
              f"parsed so far {done}/{len(rows)}", flush=True)

    return {"rows": rows}


@app.function(image=image, gpu=GPU, volumes={"/cache": hf_cache}, timeout=2 * 60 * 60)
def steering_logits(n_convs=15, layer=21, random_seeds=(0,), only_random=False):
    """Length-free confidence readout under value-axis steering.

    The generation experiment (`steering_probe`) shows alpha moves response
    LENGTH (Spearman -0.96) and not the verbalized confidence RATING (t~0).
    The obvious objection -- and the one the paper's Fig 5a invites -- is that
    the rating was measured inside generated text, so a wrap-up push could
    truncate the reasoning before the rating is "computed", or the parse could
    be confounded by length. This function removes length from the readout
    entirely: ONE forward pass, no sampling, read logits at the single position
    where the answer token would go.

    Same steering paradigm as steering_probe (unit direction at `layer`, added
    as alpha*d by a forward hook on decoder block layer-1, at every position),
    same prefixes (early = after the first "-1"; post = after the "+1" of
    paragraph dp+1), same alphas.

    Three readouts at that one position, per (prefix, direction, alpha):
      binary  : logit(Yes) - logit(No) after "Do you know the hidden criterion?
                Answer with one word: Yes or No." Graded, signed, length-free.
      digit   : distribution over "0".."10" after "Rate 0-10 ... first line",
                renormalized over the 11 digit tokens -> expected rating. This
                is the length-free analogue of the parsed rating in
                steering_probe, so the two are directly comparable.
      closure : logit(<|im_end|>) minus the mean logit over the vocabulary, at
                the same position. If the axis is a closure controller, THIS is
                the channel that should move.

    `closure` is a cheap by-product of the same forward pass, not a substitute
    for the EOS-masked steering experiment -- it shows the end-of-turn token is
    promoted, but it cannot separate "the axis encodes a closure state that
    then promotes EOS" from "the axis promotes EOS directly".
    """
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _setup_paths()
    from common.paths import value_axis
    from shared import load_conversations
    from compute_vector import compute_value_axis

    BINARY = ("Before we continue: do you know the hidden criterion? "
              "Answer with one word: Yes or No.")
    # "0 to 9", not "0 to 10": "10" tokenizes as ["1","0"], so a first-token
    # readout over 0-10 silently merges rating 10 into rating 1 -- and the
    # post-discovery state is exactly where 10s are common.
    DIGIT = ("Before we continue: how confident are you right now that you know "
             "the hidden criterion? Reply with a single digit from 0 to 9 and "
             "nothing else.")
    ALPHAS = [-75, -50, -25, 0, 25, 50, 75]

    ship = np.load(value_axis())[layer]
    means_c = torch.load("/root/activation_means_corrected.pt",
                         map_location="cpu", weights_only=False)
    corr = compute_value_axis(means_c)[layer]
    del means_c
    # Random unit directions at the same layer and the same alpha grid. Large
    # |alpha| pushes the residual stream off-distribution whatever direction you
    # push it in, so a readout that moves under BOTH the value axis and a random
    # direction is measuring steering damage, not the axis's content. These arms
    # are what make the other two interpretable.
    #
    # Use SEVERAL random directions, not one. Hidden states have a large mean
    # component, so a single fixed random vector acquires an effective sign from
    # its chance projection onto that mean: +d and -d are not equivalent
    # perturbations, and its LINEAR coefficient is chance-signed. One vector
    # gives a point estimate that cannot be distinguished from noise; a handful
    # gives a band to compare the axis against.
    dirs = {"shipped": torch.tensor(ship / np.linalg.norm(ship), dtype=torch.bfloat16),
            "corrected": torch.tensor(corr / np.linalg.norm(corr), dtype=torch.bfloat16)}
    for sd in random_seeds:
        r = np.random.default_rng(sd).normal(size=corr.shape)
        dirs[f"random{sd}"] = torch.tensor(r / np.linalg.norm(r), dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    def only_id(s):
        """Token id for `s` iff it is a single token, else None."""
        ids = tokenizer.encode(s, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    # Yes/No: take every single-token surface form and use the max logit, so the
    # result does not hinge on which capitalization/space variant Qwen prefers.
    yes_ids = [i for i in (only_id(s) for s in ("Yes", " Yes", "yes", " yes", "YES")) if i is not None]
    no_ids = [i for i in (only_id(s) for s in ("No", " No", "no", " no", "NO")) if i is not None]
    digit_ids = [only_id(str(k)) for k in range(10)]   # 0-9 only; see DIGIT above
    assert all(d is not None for d in digit_ids), "digits must be single tokens"
    n_digits = len(digit_ids)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    print(f"yes={yes_ids} no={no_ids} digits={n_digits} im_end={im_end}", flush=True)

    # ---- prefixes: identical construction to steering_probe ------------------
    conversations = load_conversations()
    prefixes = []          # (conv_id, state, messages_without_probe)
    for conv in conversations:
        if len(prefixes) >= 2 * n_convs:
            break
        if conv["discovery_paragraph"] > 2:
            continue
        msgs = conv["full_messages"]
        early = next((i for i, m in enumerate(msgs)
                      if m["role"] == "user" and m["content"].strip() == "-1"), None)
        if early is None:
            continue
        dp = conv["discovery_paragraph"]
        para = 0
        post = None
        for i, m in enumerate(msgs):
            if m["role"] == "user" and m["content"].lstrip().startswith("Paragraph"):
                para += 1
            if (m["role"] == "user" and m["content"].strip() == "+1"
                    and para == dp + 1):
                post = i
                break
        if post is None:
            continue
        cid = conv["conversation_id"]
        prefixes.append((cid, "early", msgs[:early + 1]))
        prefixes.append((cid, "post", msgs[:post + 1]))
    print(f"{len(prefixes)} prefixes ({len(prefixes)//2} conversations)", flush=True)

    # One text per (prefix, probe). add_generation_prompt puts us exactly at the
    # position where the model's first answer token goes.
    items = []             # (conv_id, state, probe, text)
    for cid, state, msgs in prefixes:
        for pname, ptext in (("binary", BINARY), ("digit", DIGIT)):
            t = tokenizer.apply_chat_template(
                msgs + [{"role": "user", "content": ptext}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            items.append((cid, state, pname, t))

    def make_hook(direction, alpha):
        d = direction.cuda()
        def hook_fn(module, inp, output):
            if isinstance(output, tuple):
                return (output[0] + alpha * d,) + output[1:]
            return output + alpha * d
        return hook_fn

    order = sorted(range(len(items)), key=lambda i: -len(items[i][3]))
    CHUNK = 4
    chunks = [order[i:i + CHUNK] for i in range(0, len(order), CHUNK)]

    rows = []
    conds = [] if only_random else (
        [("shipped", a) for a in ALPHAS] +
        [("corrected", a) for a in ALPHAS if a != 0])
    if only_random:                    # alpha=0 is direction-independent; carry one copy
        conds += [(f"random{random_seeds[0]}", 0)]
    for sd in random_seeds:
        conds += [(f"random{sd}", a) for a in ALPHAS if a != 0]
    for ci_, (dname, alpha) in enumerate(conds):
        handle = None
        if alpha != 0:
            handle = model.model.layers[layer - 1].register_forward_hook(
                make_hook(dirs[dname], alpha))
        try:
            for chunk in chunks:
                enc = tokenizer([items[i][3] for i in chunk], return_tensors="pt",
                                padding=True, add_special_tokens=False)
                input_ids = enc["input_ids"].cuda()
                attn = enc["attention_mask"].cuda()
                with torch.no_grad():
                    logits = model(input_ids=input_ids, attention_mask=attn).logits
                # left padding -> the answer position is the last column
                last = logits[:, -1, :].float().cpu()
                for row_j, i in enumerate(chunk):
                    cid, state, pname, _ = items[i]
                    lg = last[row_j]
                    lz = lg - torch.logsumexp(lg, dim=0)     # log-probabilities
                    r = {"conv_id": cid, "state": state, "probe": pname,
                         "direction": dname, "alpha": alpha,
                         # log P(end-of-turn), not a difference against the mean
                         # logit -- the mean over a 151k vocab is not a baseline.
                         "closure": float(lz[im_end]),
                         "top1": int(lg.argmax())}
                    if pname == "binary":
                        py = torch.logsumexp(lz[yes_ids], dim=0).exp()
                        pn = torch.logsumexp(lz[no_ids], dim=0).exp()
                        r["yes_minus_no"] = float(lg[yes_ids].max() - lg[no_ids].max())
                        # Bounded in [0,1], so it cannot be dragged by an overall
                        # logit-scale change the way a raw difference can.
                        r["p_yes"] = float(py / (py + pn))
                    else:
                        p = torch.softmax(lg[digit_ids], dim=0)
                        r["exp_rating"] = float((p * torch.arange(n_digits)).sum())
                        # If steering pushes the first token off the digit set,
                        # exp_rating is renormalized noise -- this says when.
                        r["digit_mass"] = float(lz[digit_ids].exp().sum())
                    rows.append(r)
                del logits, last, input_ids, attn
                torch.cuda.empty_cache()
        finally:
            if handle is not None:
                handle.remove()
        print(f"  [{ci_+1}/{len(conds)}] {dname} alpha={alpha:+d}  rows={len(rows)}",
              flush=True)

    return {"rows": rows}


@app.function(image=image, gpu=GPU, volumes={"/cache": hf_cache}, timeout=2 * 60 * 60)
def eos_association(limit=60, layer=21, n_random=8, seed=0):
    """Does a token's projection on the value axis predict P(end-of-turn) there?

    The steering result is causal but heavy-handed: it pushes the residual stream
    far off-distribution. This is the observational version, needing no
    intervention at all. For every assistant token in a real conversation, record
    two things at the same position:

        proj  = cos(h_t at `layer`, unit axis)
        eos   = log P(<|im_end|> | tokens up to t)

    If the axis is a closure signal, these are associated on natural text.

    The obvious confound is position: both the projection and P(end-of-turn) rise
    toward the end of a turn, so a raw correlation would be trivial. `rel_pos`
    (position within the assistant turn, 0-1) is stored per token so the
    association can be recomputed WITHIN position bins -- i.e. at a fixed
    distance from the end of the turn, does a higher projection still mean a
    higher probability of stopping?

    Controls: the shipped axis, and `n_random` random unit directions, projected
    at the same positions so the association can be compared against the spread
    of what random directions give.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _setup_paths()
    from common.paths import value_axis
    from shared import load_conversations
    from compute_vector import compute_value_axis
    from turns import parse_turns

    ship = np.load(value_axis())[layer]
    means_c = torch.load("/root/activation_means_corrected.pt",
                         map_location="cpu", weights_only=False)
    corr = compute_value_axis(means_c)[layer]
    del means_c
    rng = np.random.default_rng(seed)
    rand = rng.normal(size=(n_random, corr.shape[0]))
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)

    D = np.concatenate([[corr / np.linalg.norm(corr)],
                        [ship / np.linalg.norm(ship)], rand], axis=0)
    D = torch.tensor(D, dtype=torch.float32).cuda()          # (2+n_random, d)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")

    convs = load_conversations()[:limit]
    proj_all, eos_all, rel_all, cid_all = [], [], [], []
    for ci, conv in enumerate(convs):
        formatted = tokenizer.apply_chat_template(
            conv["full_messages"], tokenize=False, add_generation_prompt=False,
            enable_thinking=False)
        enc = tokenizer(formatted, return_tensors="pt", add_special_tokens=False,
                        return_offsets_mapping=True, truncation=True,
                        max_length=8192)
        ids = enc["input_ids"].cuda()
        off = enc["offset_mapping"][0].numpy()

        spans = [(t["body_start"], t["body_end"]) for t in parse_turns(formatted)
                 if t["role"] == "assistant"]
        if not spans:
            continue

        with torch.no_grad():
            # Run the trunk only. Materializing (T x 151936) logits for a whole
            # conversation is several GB and OOMs an A10G, so the unembedding is
            # applied in position chunks and reduced to one number per position.
            out = model.model(input_ids=ids, output_hidden_states=True)
            h = out.hidden_states[layer][0].float()                   # (T, d)
            hn = h / h.norm(dim=-1, keepdim=True)
            pr = (hn @ D.T).cpu().numpy()                             # (T, n_dir)
            lhs = out.last_hidden_state[0]                            # already normed
            eos_parts = []
            for i in range(0, lhs.shape[0], 256):
                lg = model.lm_head(lhs[i:i + 256]).float()
                eos_parts.append((lg[:, im_end] - torch.logsumexp(lg, dim=-1)).cpu())
                del lg
            eos_lp = torch.cat(eos_parts).numpy()
        del out, h, hn, lhs, eos_parts
        torch.cuda.empty_cache()

        for s_, e_ in spans:
            idx = [i for i in range(len(off) - 1)
                   if off[i][0] >= s_ and off[i][1] <= e_ and off[i][1] > off[i][0]]
            if len(idx) < 8:
                continue
            for k, i in enumerate(idx):
                proj_all.append(pr[i]); eos_all.append(eos_lp[i])
                rel_all.append(k / (len(idx) - 1)); cid_all.append(conv["conversation_id"])
        if (ci + 1) % 10 == 0:
            print(f"  {ci+1}/{len(convs)} conversations, {len(eos_all)} tokens",
                  flush=True)

    return {"proj": np.asarray(proj_all, dtype=np.float32),
            "eos": np.asarray(eos_all, dtype=np.float32),
            "rel_pos": np.asarray(rel_all, dtype=np.float32),
            "conv_id": np.asarray(cid_all),
            "dir_names": np.asarray(["corrected", "shipped"]
                                    + [f"random{i}" for i in range(n_random)])}


@app.function(image=image, volumes={"/cache": hf_cache}, timeout=30 * 60, memory=32768)
def logit_lens(layer=21, top_k=30, corrected=False):
    """Unembed the value axis: which tokens does the direction promote/suppress?

    The paper reports 'positive encouragement' tokens in the top 30 at layer 21
    (e.g. 想办法 / 进一步 / 加分). This is a sign-and-identity check on the axis.
    corrected=True unembeds the corrected-span axis instead of the shipped one.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _setup_paths()
    from common.paths import value_axis

    if corrected:
        from compute_vector import compute_value_axis
        means_c = torch.load("/root/activation_means_corrected.pt",
                             map_location="cpu", weights_only=False)
        axis = compute_value_axis(means_c)[layer]
        del means_c
    else:
        axis = np.load(value_axis())[layer]
    v = torch.tensor(axis, dtype=torch.float32)
    v = v / v.norm()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # bfloat16 on CPU: float32 weights would be ~32 GB for an 8B model.
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="cpu")
    # Match the logit-lens convention: apply the final norm before unembedding.
    norm = model.model.norm
    with torch.no_grad():
        h = norm(v.unsqueeze(0).to(torch.bfloat16)).squeeze(0).float()
        W = model.get_output_embeddings().weight.detach().float()   # (vocab, hidden)
        logits = W @ h

    order = torch.argsort(logits, descending=True)
    top = [(tokenizer.decode([int(i)]), float(logits[i])) for i in order[:top_k]]
    bottom = [(tokenizer.decode([int(i)]), float(logits[i])) for i in order[-top_k:]]
    return {"layer": layer, "top": top, "bottom": bottom}


@app.local_entrypoint()
def main(limit: int = 0, traces: str = ""):
    import json
    from pathlib import Path

    import numpy as np

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    trace_ids = [t for t in traces.split(",") if t]
    res = compute_projections.remote(limit=limit or None, trace_ids=trace_ids)

    rows, cos = res["rows"], res["cos"]
    np.savez_compressed(
        out_dir / "projections.npz",
        cos=cos,
        cos_random=res["cos_random"],
        cos_split=res["cos_split"],
        label=np.array([r["label"] for r in rows]),
        conv_id=np.array([r["conv_id"] for r in rows]),
        reward_fn=np.array([r["reward_fn"] for r in rows]),
        reward_fn_type=np.array([r["reward_fn_type"] for r in rows]),
        token=np.array([r["token"] for r in rows]),
        token_pos=np.array([r["token_pos"] for r in rows]),
        n_tokens=np.array([r["n_tokens"] for r in rows]),
    )
    (out_dir / "split_test_fns.json").write_text(json.dumps(res["split_test_fns"]))
    print(f"wrote {out_dir/'projections.npz'}: {cos.shape[0]} tokens x {cos.shape[1]} layers "
          f"| cos_split {res['cos_split'].shape}")

    if res["traces"]:
        (out_dir / "traces.json").write_text(json.dumps(res["traces"]))
        print(f"wrote {out_dir/'traces.json'}: {len(res['traces'])} traces")


@app.local_entrypoint()
def attempts_main(limit: int = 0, corrected: bool = False):
    """Lucky-vs-earned test. Writes results/attempts[_corrected].npz."""
    import json
    from pathlib import Path

    import numpy as np

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    res = attempt_projections.remote(limit=limit or None, corrected=corrected)
    rows = res["rows"]
    stem = "attempts_corrected" if corrected else "attempts"

    np.savez_compressed(
        out_dir / f"{stem}.npz",
        fb_cos=res["fb_cos"], fb_split=res["fb_split"],
        fb_random=res["fb_random"], asst_cos=res["asst_cos"],
        **{k: np.array([r[k] for r in rows])
           for k in ("conv_id", "reward_fn", "phase", "cell", "reward", "paragraph",
                     "attempt_in_paragraph", "discovery_paragraph", "fb_token_pos",
                     "n_fb_tokens", "n_asst_tokens", "n_tokens")},
    )
    (out_dir / f"{stem}_split_test_fns.json").write_text(json.dumps(res["split_test_fns"]))

    import collections
    print(f"wrote {out_dir / (stem + '.npz')}: {len(rows)} attempts")
    print("cells:", dict(collections.Counter(r["cell"] for r in rows)))


@app.local_entrypoint()
def divider_main(believed: bool = False):
    """Within-attempt (after - before) contrast. Writes results/divider[_believed].npz."""
    import json
    from pathlib import Path

    import numpy as np

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    res = divider_projections.remote(believed=believed)
    rows = res["rows"]
    tag = "_believed" if believed else ""
    np.savez_compressed(
        out_dir / f"divider{tag}.npz",
        before=res["before"], after=res["after"],
        before_split=res["before_split"], after_split=res["after_split"],
        **{k: np.array([r[k] for r in rows])
           for k in ("conv_id", "reward_fn", "type", "paragraph", "attempt_in_paragraph",
                     "reward", "n_before", "n_after", "n_reward_tokens")},
    )
    (out_dir / f"divider{tag}_split_test_fns.json").write_text(json.dumps(res["split_test_fns"]))
    print(f"wrote {out_dir/f'divider{tag}.npz'}: {len(rows)} attempts")


@app.local_entrypoint()
def discovery_tokens_main(limit: int = 0):
    """Per-token axis alignment over the discovery paragraph and the one after."""
    import collections
    import json
    from pathlib import Path

    import numpy as np

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    res = discovery_token_projections.remote(limit=limit or None)
    rows = res["rows"]

    np.savez_compressed(
        out_dir / "discovery_tokens.npz",
        cos=res["cos"], hnorm=res["hnorm"],
        cos_split21=res["cos_split21"], cos_random21=res["cos_random21"],
        **{k: np.array([r[k] for r in rows])
           for k in ("conv_id", "reward_fn", "reward_fn_type", "region", "role",
                     "paragraph", "attempt_in_paragraph", "discovery_paragraph",
                     "phase", "reward", "cell", "is_last_attempt", "in_thinking",
                     "axis_label", "token", "token_pos", "pos_in_span", "n_span",
                     "n_tokens")},
    )
    (out_dir / "discovery_tokens_split_test_fns.json").write_text(
        json.dumps(res["split_test_fns"]))
    print(f"wrote {out_dir/'discovery_tokens.npz'}: {len(rows)} tokens")
    print("region x role:", dict(collections.Counter(
        (r["region"], r["role"]) for r in rows)))


@app.local_entrypoint()
def prefill_rephrase_main(limit: int = 0):
    """Rephrased replication: 3 phrasings x 2 tails, same 65 contexts."""
    import collections
    from pathlib import Path

    import numpy as np

    out_dir = Path(__file__).parent / "results"
    res = prefill_probe_projections.remote(
        limit=limit or None, json_path="/root/prefill_rephrase.json")
    rows = res["rows"]
    np.savez_compressed(
        out_dir / "prefill_rephrase.npz",
        cos=res["cos"], cos_shipped=res["cos_shipped"],
        cos_random21=res["cos_random21"],
        **{k: np.array([r[k] for r in rows])
           for k in ("conv_id", "base_conv", "family", "cond", "phase",
                     "reward_fn", "segment", "token", "token_pos", "n_tokens")},
    )
    print(f"wrote {out_dir/'prefill_rephrase.npz'}: {len(rows)} tokens")
    print("conds:", len(set(r["cond"] for r in rows)))


@app.local_entrypoint()
def prefill_probes_main(limit: int = 0):
    """Prefill/probe experiment: value vs completion vs predictability."""
    import collections
    from pathlib import Path

    import numpy as np

    out_dir = Path(__file__).parent / "results"
    res = prefill_probe_projections.remote(limit=limit or None)
    rows = res["rows"]
    np.savez_compressed(
        out_dir / "prefill_probes.npz",
        cos=res["cos"], cos_shipped=res["cos_shipped"],
        cos_random21=res["cos_random21"],
        **{k: np.array([r[k] for r in rows])
           for k in ("conv_id", "base_conv", "family", "cond", "phase",
                     "reward_fn", "segment", "token", "token_pos", "n_tokens")},
    )
    print(f"wrote {out_dir/'prefill_probes.npz'}: {len(rows)} tokens")
    print("family x cond x segment:", dict(collections.Counter(
        (r["family"], r["cond"], r["segment"]) for r in rows)))


@app.local_entrypoint()
def steering_probe_main(n_convs: int = 15):
    """Confidence-vs-length dissociation under value-axis steering."""
    import json
    from pathlib import Path

    out_dir = Path(__file__).parent / "results"
    res = steering_probe.remote(n_convs=n_convs)
    with open(out_dir / "steering_probe.jsonl", "w") as f:
        for r in res["rows"]:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out_dir/'steering_probe.jsonl'}: {len(res['rows'])} generations")


@app.local_entrypoint()
def eos_association_main(limit: int = 60):
    """Observational association between axis projection and P(end-of-turn)."""
    import numpy as np
    from pathlib import Path

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    r = eos_association.remote(limit=limit)
    np.savez_compressed(out_dir / "eos_association.npz", **r)
    print(f"wrote {out_dir/'eos_association.npz'}: {len(r['eos'])} tokens")


@app.local_entrypoint()
def steering_logits_main(n_convs: int = 15, random_seeds: str = "0",
                        only_random: bool = False, out: str = "steering_logits.jsonl"):
    """Length-free confidence readout under steering (answers the Fig-5a objection).

    --random-seeds "1,2,3" --only-random --out steering_logits_randband.jsonl
    adds more random control directions without recomputing the axis arms.
    """
    import json
    from pathlib import Path

    seeds = tuple(int(x) for x in random_seeds.split(",") if x.strip())
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    res = steering_logits.remote(n_convs=n_convs, random_seeds=seeds,
                                 only_random=only_random)
    with open(out_dir / out, "w") as f:
        for r in res["rows"]:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out_dir/out}: {len(res['rows'])} rows")


@app.local_entrypoint()
def extended_retries_main():
    """Extended-retry experiment: headers at depths 1-20, two arms."""
    import collections
    from pathlib import Path

    import numpy as np

    out_dir = Path(__file__).parent / "results"
    res = extended_retry_headers.remote()
    rows = res["rows"]
    np.savez_compressed(
        out_dir / "extended_retries.npz",
        hdr_cos=res["hdr_cos"], hdr_shipped=res["hdr_shipped"],
        hdr_random21=res["hdr_random21"],
        **{k: np.array([r[k] for r in rows])
           for k in ("conv_id", "arm", "paragraph_id", "depth", "prev", "n_header")},
    )
    print(f"wrote {out_dir/'extended_retries.npz'}: {len(rows)} headers")
    print("arm x depth range:", dict(collections.Counter(r["arm"] for r in rows)))


@app.local_entrypoint()
def assistant_headers_main(limit: int = 0):
    """Header cosines + raw header means + both candidate header-difference vectors."""
    import collections
    import json
    from pathlib import Path

    import numpy as np

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    res = assistant_headers.remote(limit=limit or None)
    rows = res["rows"]

    np.savez_compressed(
        out_dir / "assistant_headers.npz",
        hdr_cos=res["hdr_cos"], hdr_split=res["hdr_split"], hdr_raw=res["hdr_raw"],
        **{k: np.array([r[k] for r in rows])
           for k in ("conv_id", "reward_fn", "paragraph", "attempt_in_paragraph",
                     "discovery_paragraph", "para_phase", "reward", "cell",
                     "is_discovery_moment", "prev", "has_think_open", "n_header")},
    )
    (out_dir / "assistant_headers_split_test_fns.json").write_text(
        json.dumps(res["split_test_fns"]))

    # Both candidate header-difference vectors, value-axis recipe (per-function
    # cell means, difference per function, average over functions with both cells).
    raw = res["hdr_raw"].astype(np.float64)
    fn_arr = np.array([r["reward_fn"] for r in rows])
    prev = np.array([r["prev"] for r in rows])
    phase = np.array([r["para_phase"] for r in rows])
    disc_m = np.array([r["is_discovery_moment"] == "True" for r in rows])

    def cell_mean(fn, mask):
        m = (fn_arr == fn) & mask
        return raw[m].mean(axis=0) if m.sum() else None

    def build(mask_a, mask_b, name):
        deltas = []
        for fn in sorted(set(fn_arr.tolist())):
            a, b = cell_mean(fn, mask_a), cell_mean(fn, mask_b)
            if a is not None and b is not None:
                deltas.append(b - a)
        axis = np.mean(deltas, axis=0)
        np.save(out_dir / name, axis)
        print(f"wrote {out_dir/name}: {axis.shape} from {len(deltas)} functions")

    # A: matched after "-1" -- discovery attempt header minus pre-discovery retries
    build((prev == "minus1") & (phase == "pre"),
          (prev == "minus1") & disc_m,
          "header_axis_disc_minus_pre.npy")
    # B: matched after "Paragraph N" -- post-discovery headers minus pre first-attempts
    build((prev == "paragraph") & (phase == "pre"),
          (prev == "paragraph") & (phase == "post"),
          "header_axis_post_minus_pre.npy")

    print(f"wrote {out_dir/'assistant_headers.npz'}: {len(rows)} headers")
    print("prev x phase:", dict(collections.Counter(
        (r["prev"], r["para_phase"]) for r in rows)))


@app.local_entrypoint()
def attempt_split_main(limit: int = 0):
    """Per-attempt pre/post-split projections vs the corrected axis."""
    import collections
    import json
    from pathlib import Path

    import numpy as np

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    res = attempt_split_projections.remote(limit=limit or None)
    rows = res["rows"]

    np.savez_compressed(
        out_dir / "attempt_split.npz",
        before=res["before"], after=res["after"],
        before_shipped=res["before_shipped"], after_shipped=res["after_shipped"],
        before_split=res["before_split"], after_split=res["after_split"],
        **{k: np.array([r[k] for r in rows])
           for k in ("conv_id", "reward_fn", "reward_fn_type", "channel",
                     "paragraph", "attempt_in_paragraph", "discovery_paragraph",
                     "para_phase", "reward", "cell", "is_discovery_moment",
                     "believed_confidence", "n_before", "n_after",
                     "n_split_tokens")},
    )
    (out_dir / "attempt_split_test_fns.json").write_text(
        json.dumps(res["split_test_fns"]))
    print(f"wrote {out_dir/'attempt_split.npz'}: {len(rows)} rows")
    print("channel x cell:", dict(collections.Counter(
        (r["channel"], r["cell"]) for r in rows)))


@app.local_entrypoint()
def rebuild_axis_main(limit: int = 0):
    """Corrected-span extraction: regenerate means, rebuild axis, project tokens.

    Writes results/activation_means_corrected.pt, value_axis_corrected.npy,
    projections_corrected.npz, corrected_split_test_fns.json.
    """
    import collections
    import json
    from pathlib import Path

    import numpy as np
    import torch

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    res = rebuild_axis.remote(limit=limit or None)
    rows = res["rows"]

    means = {fn: {"before_mean": torch.tensor(d["before_mean"], dtype=torch.float32),
                  "after_mean": torch.tensor(d["after_mean"], dtype=torch.float32),
                  "before_count": d["before_count"], "after_count": d["after_count"]}
             for fn, d in res["means"].items()}
    torch.save(means, out_dir / "activation_means_corrected.pt")
    np.save(out_dir / "value_axis_corrected.npy", res["axis_corrected"])

    np.savez_compressed(
        out_dir / "projections_corrected.npz",
        cos=res["cos"], cos_shipped=res["cos_shipped"],
        cos_split=res["cos_split"], cos_random=res["cos_random"],
        **{k: np.array([r[k] for r in rows])
           for k in ("conv_id", "reward_fn", "reward_fn_type", "label",
                     "token", "token_pos", "n_tokens")},
    )
    (out_dir / "corrected_split_test_fns.json").write_text(
        json.dumps(res["split_test_fns"]))

    print(f"wrote {out_dir/'activation_means_corrected.pt'}: {len(means)} functions")
    print(f"wrote {out_dir/'value_axis_corrected.npy'}: {res['axis_corrected'].shape}")
    print(f"wrote {out_dir/'projections_corrected.npz'}: {len(rows)} tokens "
          f"{dict(collections.Counter(r['label'] for r in rows))}")
    if res["skipped_convs"]:
        print(f"skipped convs: {res['skipped_convs']}")


@app.local_entrypoint()
def logit_lens_main(layer: int = 21, top_k: int = 30, corrected: bool = False):
    import json
    from pathlib import Path

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    res = logit_lens.remote(layer=layer, top_k=top_k, corrected=corrected)
    res["axis"] = "corrected" if corrected else "shipped"
    name = "logit_lens_corrected.json" if corrected else "logit_lens.json"
    (out_dir / name).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"wrote {out_dir / name}")
    print("top:", [t for t, _ in res["top"][:15]])
