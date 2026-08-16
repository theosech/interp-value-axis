"""One tidy row per attempt, for all 380 conversations.

The 2x2 that matters: whether the ATTEMPT succeeded (the +1/-1 the game returned)
crossed with whether the REASONING named the true criterion (the codex judge).

    code  reward  reason    meaning
    RC      +1    correct   succeeded, and knew why
    RW      +1    wrong     succeeded on a wrong hypothesis      -> "lucky"
    FC      -1    correct   named the rule but botched execution
    FW      -1    wrong     ordinary failing guess
    NF     none   -         last failing attempt of a round; the game moved on
                            without returning feedback (no +1/-1 turn exists)

`is_synth_discovery` marks the scripted discovery attempt: the successful attempt
inside `discovery_paragraph`. A round ends on +1, so there is at most one per
conversation.

Hierarchy reminder:
    conversation = one hidden criterion, 3-8 paragraphs
    paragraph    = one Wikipedia source text, 1-5 attempts, ends on +1
    attempt      = one hypothesis + one rewrite + (usually) one feedback token
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "value-axis"))
sys.path.insert(0, str(ROOT / "value-axis" / "construction"))

CODE = {(True, "correct"): "RC", (True, "wrong"): "RW", (True, "unclear"): "RU",
        (False, "correct"): "FC", (False, "wrong"): "FW", (False, "unclear"): "FU"}


def build(layer=None):
    """Tidy per-attempt DataFrame. Projections merged where a feedback token exists."""
    import numpy as np
    import pandas as pd
    from merge_labels import load_labels
    from shared import load_conversations, load_reward_functions
    from turns import attempt_records
    from transformers import AutoTokenizer

    if layer is None:
        from common.paths import DEFAULT_LAYER as layer

    labels = load_labels()                      # (conv_id, para, att) -> judge row
    rf = {r["name"]: r for r in load_reward_functions()}
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    def render(ms):
        try:
            return tok.apply_chat_template(ms, tokenize=False,
                                           add_generation_prompt=False, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=False)

    rows = []
    for conv in load_conversations():
        cid, dp = conv["conversation_id"], conv["discovery_paragraph"]
        recs = attempt_records(render(conv["full_messages"]), dp)
        # the scripted discovery: the successful attempt inside discovery_paragraph
        synth = next((i for i, r in enumerate(recs)
                      if r["paragraph"] == dp and r["reward"] is True), None)
        for i, r in enumerate(recs):
            key = (cid, r["paragraph"], r["attempt_in_paragraph"])
            lab = labels.get(key)
            reason = lab["label"] if lab else None
            code = "NF" if r["reward"] is None else CODE.get((r["reward"], reason), "??")
            rows.append({
                "conv_id": cid, "reward_fn": conv["reward_fn"],
                "criterion": rf[conv["reward_fn"]]["description"],
                "attempt_idx": i,                      # 0-based, within the conversation
                "paragraph": r["paragraph"],
                "attempt_in_paragraph": r["attempt_in_paragraph"],
                "discovery_paragraph": dp,
                "reward": r["reward"], "reason": reason, "code": code,
                "is_synth_discovery": (i == synth),
                "leak": bool(lab["leak"]) if lab else False,
                "think": lab["think"] if lab else "",
            })

    df = pd.DataFrame(rows)

    # Merge layer-`layer` projections at the feedback token (NaN where no feedback).
    #
    # `proj`         = the RELEASED axis, built from all 48 reward functions -> IN-SAMPLE.
    # `proj_heldout` = for each attempt, the mean projection onto only those split
    #                  directions whose 35 training functions EXCLUDE this attempt's
    #                  reward function. Each function is held out in ~2.7 of the 10 splits.
    A = np.load(ROOT / "results" / "attempts.npz", allow_pickle=False)
    split_test = json.loads((ROOT / "results" / "attempts_split_test_fns.json").read_text())
    npz_fn = A["reward_fn"].astype(str)
    fb_split = A["fb_split"][:, :, layer]                      # (n_attempts, n_splits)

    mask = np.zeros(fb_split.shape, dtype=bool)                # True where held out
    for s, test_fns in enumerate(split_test):
        mask[:, s] = np.isin(npz_fn, list(test_fns))
    n_held = mask.sum(axis=1)
    held = np.where(n_held > 0,
                    np.where(mask, fb_split, 0).sum(axis=1) / np.maximum(n_held, 1),
                    np.nan)

    proj = pd.DataFrame({
        "conv_id": A["conv_id"].astype(str),
        "paragraph": A["paragraph"].astype(int),
        "attempt_in_paragraph": A["attempt_in_paragraph"].astype(int),
        "proj": A["fb_cos"][:, layer],
        "proj_heldout": held,
        "n_heldout_splits": n_held,
        # Whole-attempt span: mean cosine over this attempt's own assistant tokens
        # (think block + rewritten paragraph), median 167 tokens. Closer to how the
        # paper reports projections, and much less noisy than a 2-token span -- but
        # it contains the very reasoning the judge label was derived from, so a
        # difference between RC and RW here may partly reflect the direction reading
        # "this text asserts a rule confidently" rather than any value state.
        # In-sample only; split directions were stored for the feedback token alone.
        "proj_asst": A["asst_cos"][:, layer],
        "n_asst_tokens": A["n_asst_tokens"].astype(int),
    })
    df = df.merge(proj, on=["conv_id", "paragraph", "attempt_in_paragraph"], how="left")

    # Cumulative evidence covariates -- sharper than paragraph index, since a
    # conversation can burn five hypotheses inside one paragraph.
    df = df.sort_values(["conv_id", "attempt_idx"]).reset_index(drop=True)
    g = df.groupby("conv_id", sort=False)
    df["n_attempts_so_far"] = g.cumcount()
    df["n_plus1_so_far"] = g["reward"].transform(
        lambda s: s.fillna(False).astype(bool).cumsum().shift(fill_value=0))
    return df


def sequences(df):
    """One row per conversation: the ordered attempt codes as a compact string.

    The scripted discovery attempt is wrapped in [] e.g. FW FW [RC] RC RC
    """
    out = []
    for cid, g in df.groupby("conv_id", sort=True):
        g = g.sort_values("attempt_idx")
        toks = [f"[{c}]" if s else c for c, s in zip(g.code, g.is_synth_discovery)]
        out.append({"conv_id": cid, "reward_fn": g.reward_fn.iloc[0],
                    "n_paragraphs": int(g.paragraph.max()),
                    "n_attempts": len(g),
                    "discovery_paragraph": int(g.discovery_paragraph.iloc[0]),
                    "synth_idx": int(g.attempt_idx[g.is_synth_discovery].iloc[0])
                                 if g.is_synth_discovery.any() else -1,
                    "sequence": " ".join(toks)})
    import pandas as pd
    return pd.DataFrame(out)


if __name__ == "__main__":
    df = build()
    seq = sequences(df)
    df.to_csv(ROOT / "results" / "attempt_table.csv", index=False)
    seq.to_csv(ROOT / "results" / "attempt_sequences.csv", index=False)
    print(f"{len(df)} attempts, {len(seq)} conversations")
    print(df.code.value_counts().to_string())
    print()
    print(seq.head(10).to_string())
