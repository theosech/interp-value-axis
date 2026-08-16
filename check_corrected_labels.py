"""Validation of corrected_spans.corrected_classify_tokens (CPU, tokenizer-only).

This is the evidence behind the corrected-axis provenance claims:
  1. every corrected-labeled token sits inside an assistant turn
     (47,797 tokens, 0 role violations)
  2. on the 146 conversations where the UPSTREAM locator happened to land
     correctly (span inside the dp+1 assistant body), corrected labels match
     upstream classify_tokens EXACTLY (146/146) -- the fix is a strict
     relocation, not a redesign
  3. label census + conversations dropped by the min-token filter (0)

Usage: python check_corrected_labels.py            (takes a few minutes)
"""

import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "value-axis", ROOT / "value-axis" / "construction"):
    sys.path.insert(0, str(p))

from transformers import AutoTokenizer  # noqa: E402

from shared import (SYNTACTIC_TOKEN_CHECKERS, load_conversations,  # noqa: E402
                    load_reward_functions, load_reward_labels)
from extract_activations import classify_tokens, find_modified_text_spans  # noqa: E402
from turns import parse_turns, attempt_records  # noqa: E402
from corrected_spans import corrected_classify_tokens  # noqa: E402


def main():
    tk = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    rf_by_name = {r["name"]: r for r in load_reward_functions()}
    reward_labels = load_reward_labels()
    convs = load_conversations()

    census = collections.Counter()
    role_bad = 0
    clean_agree, clean_disagree = 0, 0
    dropped = []
    per_fn = collections.Counter()

    for conv in convs:
        cid, fn = conv["conversation_id"], conv["reward_fn"]
        dp = conv["discovery_paragraph"]
        conv_idx = int(cid.split("__conv")[1])
        formatted = tk.apply_chat_template(conv["full_messages"], tokenize=False,
                                           add_generation_prompt=False,
                                           enable_thinking=False)
        enc = tk(formatted, return_offsets_mapping=True, add_special_tokens=False,
                 truncation=True, max_length=16384)
        offsets = enc["offset_mapping"]
        toks = [tk.decode([t]) for t in enc["input_ids"]]

        corr = corrected_classify_tokens(toks, offsets, formatted, conv, fn,
                                         rf_by_name[fn]["type"], reward_labels,
                                         conv_idx, SYNTACTIC_TOKEN_CHECKERS)
        corr_d = {p: l for p, l in corr if l != "excluded"}
        for l in corr_d.values():
            census[l] += 1
            per_fn[fn] += 1
        if not corr_d:
            dropped.append(cid)

        turns = parse_turns(formatted)
        for p in corr_d:
            ch = offsets[p][0]
            role = next((t["role"] for t in turns
                         if t["body_start"] <= ch < t["body_end"]), "marker")
            if role != "assistant":
                role_bad += 1
                print(f"  ROLE VIOLATION {cid} pos={p} role={role} tok={toks[p]!r}")

        spans = find_modified_text_spans(formatted, conv)
        tgt = next((s for s in spans if s["paragraph"] == dp + 1 and s["reward"]), None)
        rec = next((r for r in attempt_records(formatted, dp)
                    if r["paragraph"] == dp + 1 and r["reward"] is True), None)
        if (tgt and rec and rec["assistant_start"] <= tgt["start"]
                and tgt["end"] <= rec["assistant_end"]):
            ups = {p: l for p, l in classify_tokens(
                toks, offsets, formatted, conv, fn, rf_by_name[fn]["type"],
                reward_labels, conv_idx) if l != "excluded"}
            if ups == corr_d:
                clean_agree += 1
            else:
                clean_disagree += 1

    print()
    print(f"census: {dict(census)}  total={sum(census.values())}")
    print(f"role violations: {role_bad}")
    print(f"convs dropped (no labels): {len(dropped)}  {dropped[:6]}")
    print(f"upstream-clean convs: agree={clean_agree}  disagree={clean_disagree}")
    print(f"reward functions with labeled tokens: {len(set(per_fn))}/48")


if __name__ == "__main__":
    main()
