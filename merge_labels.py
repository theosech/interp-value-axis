"""Join judge labels onto the per-attempt projections and build the SEMANTIC cells.

The judge indexes items by a global counter over assistant messages, in
`load_conversations()` order. `attempts.npz` is keyed by (conv_id, paragraph,
attempt_in_paragraph) and excludes attempts that never received feedback. The
mapping between them is recomputed here deterministically rather than stored,
so the judge run does not need repeating.

Semantic cells (what the positional split got wrong):
  lucky     +1 BEFORE the conversation's first correct rule statement
  discovery the first +1 at or after the first correct statement
  post      later +1s
  fail      -1
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "value-axis"))
sys.path.insert(0, str(ROOT / "value-axis" / "construction"))


def judge_index():
    """{global_id: (conv_id, paragraph, attempt_in_paragraph)} -- same enumeration
    order as judge_rules.load_items."""
    from shared import load_conversations
    idx, gid = {}, 0
    for conv in load_conversations():
        para, att = 0, 0
        for msg in conv["full_messages"]:
            body = msg["content"] or ""
            if msg["role"] == "user" and body.lstrip().startswith("Paragraph"):
                para, att = para + 1, 0
            elif msg["role"] == "assistant":
                att += 1
                idx[gid] = (conv["conversation_id"], para, att)
                gid += 1
    return idx


def load_labels():
    """{(conv_id, paragraph, attempt_in_paragraph): row}"""
    idx = judge_index()
    path = ROOT / "results" / "rule_labels.jsonl"
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = idx.get(r["id"])
        if key is None:
            continue
        assert key[0] == r["conv_id"] and key[1] == r["paragraph"], \
            f"judge index disagrees for id={r['id']}: {key} vs {r['conv_id']}/{r['paragraph']}"
        out[key] = r
    return out


def first_correct_paragraph(labels):
    """{conv_id: (paragraph, attempt)} of the first 'correct' statement, per conversation."""
    best = {}
    for (cid, para, att), r in labels.items():
        if r["label"] != "correct":
            continue
        if cid not in best or (para, att) < best[cid]:
            best[cid] = (para, att)
    return best


def semantic_cell(cid, para, att, reward, first_correct):
    if not reward:
        return "fail"
    fc = first_correct.get(cid)
    if fc is None:
        return "lucky_neverstated"        # +1 in a conversation that never states the rule
    if (para, att) < fc:
        return "lucky"
    if (para, att) == fc:
        return "discovery"
    return "post"


def main():
    import numpy as np
    import pandas as pd

    labels = load_labels()
    fc = first_correct_paragraph(labels)
    print(f"{len(labels)} labelled attempts, {len(fc)} conversations with a 'correct' statement")

    A = np.load(ROOT / "results" / "attempts.npz", allow_pickle=False)
    df = pd.DataFrame({k: A[k] for k in
                       ["conv_id", "reward_fn", "paragraph", "attempt_in_paragraph",
                        "discovery_paragraph", "fb_token_pos"]})
    df["conv_id"] = df["conv_id"].astype(str)
    df["reward_fn"] = df["reward_fn"].astype(str)
    for c in ["paragraph", "attempt_in_paragraph", "discovery_paragraph", "fb_token_pos"]:
        df[c] = df[c].astype(int)
    df["reward"] = A["reward"].astype(str) == "True"

    keys = list(zip(df.conv_id, df.paragraph, df.attempt_in_paragraph))
    df["judge"] = [labels[k]["label"] if k in labels else None for k in keys]
    df["leak"] = [labels[k]["leak"] if k in labels else False for k in keys]
    df["cell"] = [semantic_cell(c, p, a, r, fc)
                  for c, p, a, r in zip(df.conv_id, df.paragraph,
                                        df.attempt_in_paragraph, df.reward)]
    print(f"join coverage: {df.judge.notna().mean():.1%} of {len(df)} attempts")
    return df, A


if __name__ == "__main__":
    df, _ = main()
    print(df.cell.value_counts().to_string())
